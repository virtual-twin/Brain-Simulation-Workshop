/*
 * vid-tvb-ext-ontology-simulation.js
 * -----------------------------------
 * Playwright recorder that produces img/videos/tvbo-ext-ontology-simulation.mp4
 * (driven by vid-tvb-ext-ontology-simulation.sh).
 *
 * Records a real JupyterLab session running the `tvb-ext-ontology` extension. It walks
 * the clinical knowledge graph and then runs a simulation:
 *   search a disease -> click through the graph (disease -> study -> the neural-mass model
 *   that study used) -> Add the model to the Workspace (the change is highlighted) ->
 *   choose parcellation + tractogram -> Run Simulation -> BIDS results appear in the file
 *   browser.
 *
 * The clinical layer (tvb-o-clinical.ttl + tvb-o-clinical-nmm.ttl) links each study to the
 * runnable TVB model implementing its dynamics, so the graph reaches an addable NMM.
 *
 * The force graph is an HTML canvas whose backing store is the full window, so
 * react-force-graph's centerAt/zoomToFit aim off the clipped viewport; frameGraph()
 * compensates that offset. Node screen positions come from graphData (React fiber on
 * `.graph-container`) + the live d3-zoom transform (canvas.__zoom).
 *
 * Env: BASE, VIDEO_DIR, MARKS_FILE, PARCELLATION, TRACTOGRAM, SKIP_RUN(=1),
 *      CLICK_PATH (comma-separated node labels to click in order; the LAST is the model
 *      added to the workspace; the FIRST is what's typed into the search box).
 */
const fs = require('fs');
const { chromium } = require('playwright');
const ci = require('./cinematic'); // shared cinematic toolkit (cursor / caption / highlight / click)
const moveClick = ci.moveClick;    // alias: same red-cursor click used by the platform demos

const BASE = process.env.BASE || 'http://127.0.0.1:8889/lab';
const VIDEO_DIR = process.env.VIDEO_DIR || '/tmp/tvbo-rec/video';
const MARKS_FILE = process.env.MARKS_FILE || `${VIDEO_DIR}/marks.json`;
const CLICK_PATH = (process.env.CLICK_PATH || 'Schizophrenia,Yang2015,ReducedWongWang').split(',').map(s => s.trim());
const SEARCH = CLICK_PATH[0];
const MODEL = CLICK_PATH[CLICK_PATH.length - 1];
const PARCELLATION = process.env.PARCELLATION || 'DesikanKilliany';
const TRACTOGRAM = process.env.TRACTOGRAM || 'dTOR';
const SKIP_RUN = process.env.SKIP_RUN === '1';
const W = 1366, H = 1040;
const sleep = ms => new Promise(r => setTimeout(r, ms));

// (the red-circle cursor + caption banner + highlight all come from ./cinematic.js,
//  the shared toolkit, so this video matches the platform screencasts.)

// live react-force-graph data + d3-zoom transform -> nodes in screen px
function readGraph(page) {
  return page.evaluate(() => {
    const root = document.querySelector('.graph-container');
    const canvas = document.querySelector('.graph-container canvas');
    const gbox = document.querySelector('.ontology-graph');
    if (!root || !canvas || !gbox) return null;
    const key = Object.keys(root).find(k => k.startsWith('__reactFiber'));
    let gd = null; const seen = new Set(); const st = [root[key]];
    while (st.length) {
      const f = st.pop(); if (!f || seen.has(f)) continue; seen.add(f);
      const p = f.memoizedProps;
      if (p && p.graphData && Array.isArray(p.graphData.nodes)) gd = p.graphData;
      if (f.child) st.push(f.child);
      if (f.sibling) st.push(f.sibling);
      if (f.return && !seen.has(f.return)) st.push(f.return);
    }
    if (!gd) return null;
    const t = canvas.__zoom, r = canvas.getBoundingClientRect(), g = gbox.getBoundingClientRect();
    const m = { l: g.x + 25, r: g.x + g.width - 25, t: g.y + 60, b: g.y + g.height - 20 };
    const nodes = gd.nodes.map(n => {
      const x = r.x + t.x + n.x * t.k, y = r.y + t.y + n.y * t.k;
      return { label: n.label, type: n.type, x, y, onscreen: x > m.l && x < m.r && y > m.t && y < m.b };
    });
    return { nodes };
  });
}

async function nodeScreen(page, label) {
  const g = await readGraph(page);
  if (!g) return null;
  return g.nodes.find(n => n.label === label) || null;
}

// expose the react-force-graph imperative instance as window.__fg
function exposeFG(page) {
  return page.evaluate(() => {
    const root = document.querySelector('.graph-container'); if (!root) return false;
    const key = Object.keys(root).find(k => k.startsWith('__reactFiber'));
    const seen = new Set(); const st = [root[key]]; let fg = null;
    while (st.length) {
      const f = st.pop(); if (!f || seen.has(f)) continue; seen.add(f);
      let h = f.memoizedState, g = 0;
      while (h && typeof h === 'object' && 'next' in h && g < 40) {
        const v = h.memoizedState;
        if (v && v.current && typeof v.current.zoomToFit === 'function') fg = v.current;
        h = h.next; g++;
      }
      if (f.child) st.push(f.child); if (f.sibling) st.push(f.sibling);
      if (f.return && !seen.has(f.return)) st.push(f.return);
    }
    window.__fg = fg; return !!fg;
  });
}

// smoothly frame the whole graph, compensating the canvas-overflow offset
async function frameGraph(page, fillFrac, ms) {
  await exposeFG(page);
  await page.evaluate(({ fillFrac }) => {
    const fg = window.__fg; if (!fg) return;
    const bb = fg.getGraphBbox(); if (!bb) return;
    const cont = document.querySelector('.graph-container').getBoundingClientRect();
    const bw = Math.max(1, bb.x[1] - bb.x[0]), bh = Math.max(1, bb.y[1] - bb.y[0]);
    let k = Math.min(cont.width * fillFrac / bw, cont.height * fillFrac / bh);
    k = Math.max(2.2, Math.min(6, k));
    fg.zoom(k, 800);
  }, { fillFrac });
  await sleep(1000);
  await page.evaluate(({ ms }) => {
    const fg = window.__fg; if (!fg) return;
    const bb = fg.getGraphBbox(); if (!bb) return;
    const canvas = document.querySelector('.graph-container canvas');
    const cont = document.querySelector('.graph-container').getBoundingClientRect();
    const rect = canvas.getBoundingClientRect(); const k = canvas.__zoom.k;
    const bcx = (bb.x[0] + bb.x[1]) / 2, bcy = (bb.y[0] + bb.y[1]) / 2;
    const compX = ((rect.x + rect.width / 2) - (cont.x + cont.width / 2)) / k;
    const compY = ((rect.y + rect.height / 2) - (cont.y + cont.height / 2)) / k;
    fg.centerAt(bcx + compX, bcy + compY, ms);
  }, { ms });
  await sleep(ms + 400);
}

// click a graph node by label, reframing if it's off the visible area; confirm via the info box
async function clickNodeByLabel(page, label) {
  for (let attempt = 0; attempt < 16; attempt++) {
    const p = await nodeScreen(page, label);
    if (p && Number.isFinite(p.x) && p.onscreen) {
      await moveClick(page, p.x, p.y);
      try {
        await page.waitForFunction(
          m => { const e = document.querySelector('.info-box .node-info'); return e && e.textContent.includes(m); },
          label, { timeout: 3000 }
        );
        return true;
      } catch (e) { await sleep(500); }
    } else if (p && Number.isFinite(p.x)) {
      await frameGraph(page, 0.7, 700); await sleep(400);
    } else { await sleep(600); }
  }
  throw new Error('could not click node: ' + label);
}

async function highlightModel(page) {
  // red-square highlight (shared style) around the model that was just added.
  const box = await page.evaluate(() => {
    const h4 = [...document.querySelectorAll('.workspace h4')].find(e => e.textContent.trim() === 'Model');
    const p = h4 && h4.nextElementSibling;
    if (!p) return null;
    const r = p.getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height };
  });
  await ci.highlightBox(page, box, 2400);
}

(async () => {
  fs.mkdirSync(VIDEO_DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({
    viewport: { width: W, height: H }, deviceScaleFactor: 1,
    recordVideo: { dir: VIDEO_DIR, size: { width: W, height: H } }
  });
  await ci.installCursor(ctx);
  const page = await ctx.newPage();
  const t0 = Date.now();
  const mark = {};
  const stamp = k => { mark[k] = +((Date.now() - t0) / 1000).toFixed(2); };

  await page.goto(BASE, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.jp-LauncherCard', { timeout: 60000 });
  await sleep(1200);
  await page.evaluate(() => {
    document.querySelectorAll('[class*="Toast"],[class*="toast"],[class*="Notification"]').forEach(n => {
      if (/Jupyter news|notified about|privacy policy/i.test(n.textContent || '')) n.remove();
    });
  });
  await sleep(600);

  // open the Ontology Widget, then close the Launcher tab
  await page.locator('.jp-LauncherCard', { hasText: 'Ontology' }).first().click();
  await page.waitForSelector('.ontology-graph', { timeout: 60000 });
  await page.waitForSelector('.workspace', { timeout: 60000 });
  await sleep(800);
  try { await page.locator('.lm-TabBar-tab', { hasText: 'Launcher' }).locator('.lm-TabBar-tabCloseIcon, .jp-icon-hoverShow').first().click({ timeout: 3000 }); } catch (e) {}
  await sleep(1000);
  await ci.ensureCursor(page);
  await ci.caption(page, 'The tvb-ext-ontology extension — browse the clinical knowledge graph in JupyterLab');
  await sleep(1200);

  // 1) search the ontology for the disease
  await ci.caption(page, `Search the ontology for a disease — ${SEARCH}`);
  const search = page.locator('.ontology-graph .search-bar input');
  await search.click();
  await search.pressSequentially(SEARCH, { delay: 70 });
  await sleep(400);
  await search.press('Enter');
  await page.waitForTimeout(4500);

  // 2) walk the clinical graph: disease -> study -> neural-mass model
  await ci.caption(page, 'Follow the links: disease → study → the neural-mass model it used');
  for (let i = 0; i < CLICK_PATH.length; i++) {
    await clickNodeByLabel(page, CLICK_PATH[i]);
    await sleep(i === CLICK_PATH.length - 1 ? 1600 : 2600); // dwell to read info / connections
    if (i < CLICK_PATH.length - 1) { await frameGraph(page, 0.6, 1000); await sleep(1100); }
  }

  // 3) add the model to the workspace and highlight the change
  await ci.caption(page, 'Add that neural-mass model to your workspace');
  const addBtn = page.locator('.add-button');
  await addBtn.scrollIntoViewIfNeeded();
  let b = await addBtn.boundingBox();
  await moveClick(page, b.x + b.width / 2, b.y + b.height / 2);
  await highlightModel(page);
  await sleep(2400);

  // 4) choose connectivity: parcellation + tractogram
  await ci.caption(page, 'Choose a parcellation and a tractogram for the connectome');
  const parc = page.locator('#parcellation');
  await parc.scrollIntoViewIfNeeded();
  b = await parc.boundingBox(); await page.mouse.move(b.x + 30, b.y + b.height / 2, { steps: 18 }); await sleep(300);
  await parc.selectOption(PARCELLATION); await sleep(1300);
  const tract = page.locator('#tractogram');
  b = await tract.boundingBox(); await page.mouse.move(b.x + 30, b.y + b.height / 2, { steps: 14 }); await sleep(300);
  await tract.selectOption(TRACTOGRAM); await sleep(1400);

  if (SKIP_RUN) {
    await page.screenshot({ path: `${VIDEO_DIR}/skiprun-final.png` });
  } else {
    // 5) run the simulation
    await ci.caption(page, 'Run the simulation — TVB integrates the model on the connectome');
    const runBtn = page.getByRole('button', { name: 'Run Simulation' });
    await runBtn.scrollIntoViewIfNeeded();
    b = await runBtn.boundingBox();
    await moveClick(page, b.x + b.width / 2, b.y + b.height / 2);
    stamp('run_click');
    await page.waitForSelector('text=Saved simulation results', { timeout: 240000 });
    stamp('run_done');
    await ci.caption(page, 'Saved — the BIDS simulation results appear in the file browser');
    await sleep(2500);
    try {
      await page.locator('.jp-FileBrowser button[title*="Refresh" i], .jp-FileBrowser button[data-command="filebrowser:refresh"]').first().click({ timeout: 4000 });
    } catch (e) {}
    try { await page.waitForSelector('text=sub-01', { timeout: 8000 }); } catch (e) {}
    await sleep(2600);
    stamp('end');
  }

  const video = page.video();
  await page.close();
  const vpath = await video.path();
  await browser.close();
  mark.video = vpath;
  fs.writeFileSync(MARKS_FILE, JSON.stringify(mark, null, 2));
  console.log('VIDEO_PATH=' + vpath);
  console.log('MARKS=' + JSON.stringify(mark));
})().catch(e => { console.error('SCRIPT ERROR:', (e && e.stack) || e); process.exit(1); });

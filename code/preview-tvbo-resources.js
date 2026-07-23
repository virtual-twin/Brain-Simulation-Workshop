/*
 * preview-tvbo-resources.js
 * Capture clean page-preview screenshots of the TVB-O resources (platform + PyPI pages)
 * into img/qr/preview-*.png, used next to their QR codes on the resources slide.
 *
 * Run: NODE_PATH=<dir with playwright> node code/preview-tvbo-resources.js
 * (e.g. NODE_PATH=~/projects/TVB-O/tvb-ext-ontology/node_modules)
 */
const path = require('path');
const { chromium } = require('playwright');

const OUT = path.join(__dirname, '..', 'img', 'qr');
const TARGETS = [
  { name: 'preview-tvbo-platform', url: 'https://tvbo.charite.de/' },
  { name: 'preview-tvbo-pypi', url: 'https://pypi.org/project/tvbo/' },
  { name: 'preview-tvb-ext-ontology-pypi', url: 'https://pypi.org/project/tvb-ext-ontology/' }
];
const W = 1280, H = 860;
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  for (const t of TARGETS) {
    try {
      await page.goto(t.url, { waitUntil: 'networkidle', timeout: 45000 });
    } catch (e) {
      await page.goto(t.url, { waitUntil: 'domcontentloaded', timeout: 45000 }).catch(() => {});
    }
    await sleep(1500);
    // best-effort dismiss of cookie/consent banners
    for (const label of ['Accept all', 'Accept', 'I agree', 'Alle akzeptieren', 'Akzeptieren', 'Got it', 'OK']) {
      try { const b = page.getByRole('button', { name: label }); if (await b.first().isVisible({ timeout: 600 })) { await b.first().click({ timeout: 1500 }); break; } } catch (e) {}
    }
    await sleep(600);
    await page.evaluate(() => window.scrollTo(0, 0));
    const out = path.join(OUT, t.name + '.png');
    await page.screenshot({ path: out, clip: { x: 0, y: 0, width: W, height: H } });
    console.log('wrote', out);
  }
  await browser.close();
})().catch(e => { console.error('SCRIPT ERROR:', (e && e.stack) || e); process.exit(1); });

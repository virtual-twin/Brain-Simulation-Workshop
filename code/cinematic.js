/*
 * cinematic.js — shared screencast-cinematic helpers (CommonJS / raw Playwright).
 *
 * This is the workshop-side counterpart of tvbo-platform's
 * tests/e2e/helpers/cinematic.ts: the SAME red-circle cursor, narration caption
 * banner, red-square highlight and slow mouse moves, so every demo video — the
 * platform e2e screencasts AND the JupyterLab tvb-ext-ontology walkthrough —
 * shares one visual language. Keep the style constants here in sync with
 * cinematic.ts (cursor 24px / #ff2b2b, the caption banner, the highlight box).
 *
 * cinematic.ts targets @playwright/test; this file targets a raw `playwright`
 * `page` (used by the standalone node recorders), but the on-page primitives
 * (cursor / caption / highlight) are identical.
 */

/** Injected into every document: a red-circle cursor that tracks the mouse.
 *  Idempotent per document (guards against stacked intervals/listeners). */
function CURSOR_INIT() {
  const CID = '__cine_cursor__';
  if (window.__cineInit) return;
  window.__cineInit = true;
  function ensure() {
    if (!document.body && !document.documentElement) return;
    if (document.getElementById(CID)) return;
    const c = document.createElement('div');
    c.id = CID;
    const s = c.style;
    s.position = 'fixed';
    s.left = '50%';
    s.top = '50%';
    s.width = '24px';
    s.height = '24px';
    s.marginLeft = '-12px';
    s.marginTop = '-12px';
    s.borderRadius = '50%';
    s.border = '3px solid #ff2b2b';
    s.background = 'rgba(255,43,43,0.22)';
    s.boxShadow = '0 0 0 2px rgba(255,255,255,0.85), 0 0 14px rgba(255,43,43,0.7)';
    s.zIndex = '2147483647';
    s.pointerEvents = 'none';
    s.transition = 'transform 0.08s ease-out';
    s.willChange = 'left, top, transform';
    (document.body || document.documentElement).appendChild(c);
    document.addEventListener('mousemove', (e) => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    document.addEventListener('mousedown', () => (c.style.transform = 'scale(0.55)'), true);
    document.addEventListener('mouseup', () => (c.style.transform = 'scale(1)'), true);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ensure);
  else ensure();
  setInterval(ensure, 400);
}

/** Install the cursor on a context (runs on every document of every page). */
async function installCursor(ctx) {
  await ctx.addInitScript(CURSOR_INIT);
}

/** Ensure the cursor exists on the current document (after a navigation). */
async function ensureCursor(page) {
  await page.evaluate(CURSOR_INIT).catch(() => {});
}

/** Show / update the narration caption banner (fixed, bottom-centre). */
async function caption(page, text) {
  await page.evaluate((t) => {
    const ID = '__cine_caption__';
    let el = document.getElementById(ID);
    if (!el) {
      el = document.createElement('div');
      el.id = ID;
      const s = el.style;
      s.position = 'fixed';
      s.left = '50%';
      s.bottom = '30px';
      s.transform = 'translateX(-50%)';
      s.maxWidth = '82%';
      s.padding = '13px 24px';
      s.background = 'rgba(17,18,28,0.93)';
      s.color = '#fff';
      s.font = '600 19px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif';
      s.borderRadius = '12px';
      s.zIndex = '2147483646';
      s.pointerEvents = 'none';
      s.boxShadow = '0 8px 28px rgba(0,0,0,0.45)';
      s.border = '1px solid rgba(255,255,255,0.16)';
      s.textAlign = 'center';
      (document.body || document.documentElement).appendChild(el);
    }
    el.textContent = t;
    el.style.opacity = '1';
  }, text).catch(() => {});
}

/** Draw a red-square highlight around a bounding box {x,y,width,height}. */
async function highlightBox(page, box, holdMs = 900) {
  if (!box) return;
  await page.evaluate(({ b, hold }) => {
    const ID = '__cine_hl__';
    let el = document.getElementById(ID);
    if (!el) { el = document.createElement('div'); el.id = ID; (document.body || document.documentElement).appendChild(el); }
    const s = el.style;
    s.position = 'fixed';
    s.left = b.x - 6 + 'px';
    s.top = b.y - 6 + 'px';
    s.width = b.width + 12 + 'px';
    s.height = b.height + 12 + 'px';
    s.border = '3px solid #ff2b2b';
    s.borderRadius = '7px';
    s.boxShadow = '0 0 0 3px rgba(255,43,43,0.22), 0 0 0 9999px rgba(10,10,15,0.06)';
    s.background = 'rgba(255,43,43,0.06)';
    s.zIndex = '2147483645';
    s.pointerEvents = 'none';
    s.transition = 'all 0.18s ease';
    s.opacity = '1';
    clearTimeout(window.__cineHl);
    window.__cineHl = setTimeout(() => (el.style.opacity = '0'), hold);
  }, { b: box, hold: holdMs }).catch(() => {});
}

/** Highlight a Playwright locator (resolves its box, then highlightBox). */
async function highlight(page, locator, holdMs = 900) {
  const box = await locator.boundingBox().catch(() => null);
  await highlightBox(page, box, holdMs);
}

const pause = (page, ms) => page.waitForTimeout(ms);

/** Animate the (real) mouse to a point so the cursor follows (few steps: each
 *  step dispatches a mousemove the page must hit-test). */
async function moveTo(page, x, y, steps = 22) {
  await page.mouse.move(x, y, { steps });
}

/** Slow, narrated click at a point: move cursor → press → release. */
async function moveClick(page, x, y, steps = 28) {
  await moveTo(page, x, y, steps);
  await pause(page, 350);
  await page.mouse.down();
  await pause(page, 90);
  await page.mouse.up();
}

module.exports = { CURSOR_INIT, installCursor, ensureCursor, caption, highlight, highlightBox, moveTo, moveClick };

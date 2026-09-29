// Render individual frames to PNG for inspection.
// usage: node scripts/snapshot.mjs <16x9|9x16> <outdir> t1 t2 ...
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const [aspect = '16x9', outDir = 'build/snapshots', ...times] = process.argv.slice(2);
const [w, h] = aspect === '9x16' ? [1080, 1920] : [1920, 1080];
mkdirSync(outDir, { recursive: true });
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const url = pathToFileURL(path.join(root, 'mv', 'index.html')).href + `?render=1&aspect=${aspect}`;

const browser = await chromium.launch({ args: ['--allow-file-access-from-files'] });
const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
page.on('console', (m) => console.log('[page]', m.text()));
page.on('pageerror', (e) => console.error('[page error]', e.message));
await page.goto(url);
await page.evaluate(() => window.MV_READY);
for (const t of times) {
  const data = await page.evaluate((t) => { MV.renderFrame(t); return document.getElementById('mv').toDataURL('image/png'); }, +t);
  const file = path.join(outDir, `${aspect}_${String(t).padStart(6, '0')}.png`);
  writeFileSync(file, Buffer.from(data.split(',')[1], 'base64'));
  console.log(file);
}
await browser.close();

// Render the music video to MP4, frame by frame, with headless Chromium.
//
// usage: node scripts/render.mjs <16x9|9x16> [--fps 30] [--workers 4] [--crf 20] [--from 0] [--to 192]
//
// Every frame is drawn by MV.renderFrame(t) (a pure function of time), so the
// frame range is split across several browser workers that each encode their
// own H.264 segment; the segments are then concatenated and muxed with the
// soundtrack produced by src/compose.py.
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, existsSync, rmSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import os from 'node:os';

const args = process.argv.slice(2);
const aspect = args[0] && !args[0].startsWith('--') ? args[0] : '16x9';
const opt = (name, def) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : def; };
const fps = +opt('fps', 30);
const workers = +opt('workers', Math.max(1, Math.min(4, os.cpus().length)));
const crf = opt('crf', '20');
const preset = opt('preset', 'slow');
const FFMPEG = process.env.FFMPEG || 'ffmpeg';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const [w, h] = aspect === '9x16' ? [1080, 1920] : [1920, 1080];
const url = pathToFileURL(path.join(root, 'mv', 'index.html')).href + `?render=1&aspect=${aspect}`;
const audio = path.join(root, 'build', 'soundtrack.wav');
const tmp = path.join(root, 'build', `segments_${aspect}`);
const out = path.join(root, 'output', `JEPA-MV_${aspect}.mp4`);
if (!existsSync(audio)) { console.error('missing build/soundtrack.wav — run `python3 src/compose.py` first'); process.exit(1); }
rmSync(tmp, { recursive: true, force: true });
mkdirSync(tmp, { recursive: true });
mkdirSync(path.dirname(out), { recursive: true });

function run(cmd, argv, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, argv, { stdio: ['pipe', 'ignore', 'pipe'], ...opts });
    let err = '';
    p.stderr.on('data', (d) => { err += d; if (err.length > 20000) err = err.slice(-10000); });
    p.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}\n${err.slice(-2000)}`))));
    p.on('error', reject);
    if (opts.onSpawn) opts.onSpawn(p);
  });
}

const t0 = +opt('from', 0);
const browserProbe = await chromium.launch();
const probe = await browserProbe.newPage({ viewport: { width: w, height: h } });
await probe.goto(url);
await probe.evaluate(() => window.MV_READY);
const duration = +opt('to', await probe.evaluate(() => MV.duration));
await browserProbe.close();
const first = Math.round(t0 * fps), last = Math.round(duration * fps);
const total = last - first;
console.log(`rendering ${aspect} ${w}x${h} @${fps}fps, frames ${first}..${last - 1} (${total}) with ${workers} workers`);

let done = 0;
const started = Date.now();
async function worker(k) {
  const a = first + Math.floor((total * k) / workers), b = first + Math.floor((total * (k + 1)) / workers);
  const seg = path.join(tmp, `seg_${String(k).padStart(2, '0')}.mp4`);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error(`[worker ${k}]`, e.message));
  await page.goto(url);
  await page.evaluate(() => window.MV_READY);
  let ff;
  const enc = run(FFMPEG, ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', preset, '-tune', 'animation', '-crf', crf, '-pix_fmt', 'yuv420p', '-profile:v', 'high',
    '-g', String(fps * 2), '-threads', '2', seg], { onSpawn: (p) => (ff = p) });
  for (let f = a; f < b; f++) {
    const data = await page.evaluate((t) => { MV.renderFrame(t); return document.getElementById('mv').toDataURL('image/png'); }, f / fps);
    const buf = Buffer.from(data.slice(data.indexOf(',') + 1), 'base64');
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    done++;
    if (done % 150 === 0) {
      const el = (Date.now() - started) / 1000;
      console.log(`  ${done}/${total} frames  ${(done / el).toFixed(1)} fps  eta ${((total - done) / (done / el)).toFixed(0)}s`);
    }
  }
  ff.stdin.end();
  await enc;
  await browser.close();
  return seg;
}

const segs = await Promise.all(Array.from({ length: workers }, (_, k) => worker(k)));
const list = path.join(tmp, 'list.txt');
writeFileSync(list, segs.map((s) => `file '${s}'`).join('\n') + '\n');
await run(FFMPEG, ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list,
  '-ss', String(t0), '-t', String(duration - t0), '-i', audio,
  '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-shortest',
  '-movflags', '+faststart', '-metadata', 'title=JEPA — Predict What Matters', out]);
console.log(`wrote ${out} in ${((Date.now() - started) / 1000).toFixed(0)}s`);

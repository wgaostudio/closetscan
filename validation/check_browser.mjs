// Run the generated catalogue in an isolated headless Chrome profile.
// node validation/check_browser.mjs validation/full-<timestamp>
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const directory = resolve(process.argv[2]);
const demo = process.argv.includes('--demo');
const dialog = demo ? '#detail' : '#d';
const profile = await mkdtemp(join(tmpdir(), 'closetscan-browser-'));
const chrome = spawn(process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', '--no-first-run', '--no-default-browser-check', '--mute-audio',
  `--user-data-dir=${profile}`, '--remote-debugging-port=0', 'about:blank',
], { stdio: 'ignore' });
let socket;
const pending = new Map();
let nextId = 1;
const errors = [];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = nextId++;
  const timeout = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 30000);
  pending.set(id, { resolve: value => { clearTimeout(timeout); resolve(value); },
                    reject: error => { clearTimeout(timeout); reject(error); } });
  socket.send(JSON.stringify({ id, method, params }));
});
const evaluate = async expression => {
  const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
};
const loadImages = () => evaluate(`(async () => {
  const images = [...document.images];
  images.forEach(image => image.loading = 'eager');
  await Promise.all(images.map(image => image.decode().catch(() => {})));
  return images.filter(image => !image.naturalWidth).map(image => image.src);
})()`);
const screenshot = async name => {
  if (demo) {
    await delay(600); // Let smooth scrolling settle before capturing the viewport.
    await evaluate(`Promise.all(document.getAnimations().map(animation => animation.finished.catch(() => {})))`);
  }
  const { data } = await send('Page.captureScreenshot', { format: 'png' });
  await writeFile(join(directory, name), Buffer.from(data, 'base64'));
};

try {
  let port;
  for (let tries = 0; tries < 100; tries++) {
    try { port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; break; }
    catch { await delay(100); }
  }
  assert(port, 'Chrome did not start');
  const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket = new WebSocket(pages.find(page => page.type === 'page').webSocketDebuggerUrl);
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
    const request = pending.get(message.id);
    if (request) {
      pending.delete(message.id);
      if (message.error) request.reject(new Error(JSON.stringify(message.error)));
      else request.resolve(message.result);
    }
  });
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  const page = demo ? resolve('demo/index.html') : join(directory, 'catalogue/catalogue.html');
  await send('Page.navigate', { url: pathToFileURL(page).href });
  for (let tries = 0; tries < 100; tries++) {
    if (await evaluate(`document.readyState === 'complete' && !!document.querySelector('.card')`)) break;
    await delay(100);
  }
  const total = await evaluate(`DATA.items.length`);
  assert.equal(await evaluate(`document.querySelectorAll('.card').length`), total);
  assert.deepEqual(await loadImages(), []);
  await screenshot('catalogue-desktop.png');
  const categories = demo ? [] : await evaluate(`cats`);
  for (const category of categories) {
    const count = await evaluate(`(() => {
      document.querySelectorAll('.chip[data-v]').forEach(chip => { if (chip.dataset.v === ${JSON.stringify(category)}) chip.click(); });
      return { shown: document.querySelectorAll('.card').length, expected: DATA.items.filter(item => item.attributes.category === ${JSON.stringify(category)}).length };
    })()`);
    assert.equal(count.shown, count.expected);
  }
  if (!demo) await evaluate(`document.querySelector('.chip[data-v="all"]').click()`);
  const ids = await evaluate(`${demo ? 'chronologicalItems' : 'DATA.items'}.map(item => item.id)`);
  if (demo) {
    assert.equal(total, 34);
    assert.equal(await evaluate(`document.querySelector('#count').textContent`), '34 pieces');
    assert(await evaluate(`chronologicalItems.every((item, index, items) => !index || item.first_seen_s >= items[index-1].first_seen_s)`));
    await evaluate(`new Promise((resolve, reject) => {
      const video = document.querySelector('#video');
      if (video.readyState >= 1) return resolve(true);
      const timer = setTimeout(() => reject(new Error('Video metadata did not load')), 10000);
      video.addEventListener('loadedmetadata', () => { clearTimeout(timer); resolve(true); }, { once: true });
    })`);
  }
  for (let i = 0; i < ids.length; i++) {
    await evaluate(`document.querySelector('.card[data-id="${ids[i]}"]').click()`);
    assert.equal(await evaluate(`document.querySelector('${dialog}').open`), true);
    assert.equal(await evaluate(`document.querySelectorAll('${demo ? '.photos.plates img' : '.d-plates img'}').length`), 2);
    assert((await evaluate(`document.querySelectorAll('${demo ? '#detail-body .photos:not(.plates) img' : '.strip img'}').length`)) > 0);
    assert.deepEqual(await loadImages(), []);
    if (i === 0) await screenshot('catalogue-detail.png');
    if (demo) {
      assert(await evaluate(`[...document.querySelectorAll('#detail [data-time]')].every(button => {
        const time = Number(button.dataset.time);
        return Number.isFinite(time) && time >= 0 && time <= document.querySelector('#video').duration;
      })`), 'Invalid video timestamp');
      if (i === 0) {
        await evaluate(`document.querySelector('[data-step="1"]').click()`);
        assert.equal(await evaluate(`selected`), ids[1]);
        await evaluate(`document.querySelector('[data-step="-1"]').click()`);
        assert.equal(await evaluate(`selected`), ids[0]);
      }
      const seek = await evaluate(`new Promise((resolve, reject) => {
        const video = document.querySelector('#video');
        const button = document.querySelector('#detail [data-time]');
        const expected = Number(button.dataset.time);
        const timer = setTimeout(() => reject(new Error('Video seek timed out')), 10000);
        video.addEventListener('seeked', () => {
          clearTimeout(timer); video.pause();
          resolve({ expected, actual: video.currentTime, closed: !document.querySelector('#detail').open });
        }, { once: true });
        button.click();
      })`);
      assert(Math.abs(seek.actual - seek.expected) < 0.5, 'Video seek did not reach its timestamp');
      assert(seek.closed);
    }
    await evaluate(`document.querySelector('#close').click()`);
  }
  await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  if (demo) await evaluate(`document.querySelector('#demo').scrollIntoView()`);
  await loadImages();
  assert(await evaluate(`document.documentElement.scrollWidth <= window.innerWidth`), 'Mobile page overflows horizontally');
  await screenshot('catalogue-mobile.png');
  assert.deepEqual(errors, []);
  const report = { passed: true, garments_opened: total, categories_tested: categories, runtime_errors: errors,
                   checked: ['desktop cards', ...(demo ? ['chronological order', 'previous/next navigation', 'video seeking for every garment', 'all source/narration timestamps'] : ['category filters']), 'every garment detail', 'generated and source images', 'mobile overflow'] };
  await writeFile(join(directory, 'browser-audit.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
} finally {
  socket?.close();
  chrome.kill();
  await new Promise(resolve => chrome.exitCode !== null ? resolve() : chrome.once('exit', resolve));
  await rm(profile, { recursive: true, force: true });
}

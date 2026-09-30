// usage: node tools/shot.mjs <절대경로.html> <out.png> [폭=900]
// headless Chrome을 DevTools 프로토콜로 띄워 MathJax 조판(MathJax.startup.promise)이 끝난 뒤 전체 페이지를 캡처한다.
// 넘치는 디스플레이 수식이 있으면 경고를 출력한다. (GUIDELINES.md 부록 E)
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";

const [file, out, widthArg] = process.argv.slice(2);
const width = Number(widthArg || 900);
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn("google-chrome", [
  "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
  `--remote-debugging-port=${port}`, `--window-size=${width},1200`, "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function getJSON(path) {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(`http://127.0.0.1:${port}${path}`); return await r.json(); } catch { await sleep(100); }
  }
  throw new Error("chrome did not start");
}

const targets = await getJSON("/json/list");
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map();
ws.addEventListener("message", (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
});
const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });

await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width, height: 1200, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: "file://" + file });
let ok = false;
for (let i = 0; i < 200; i++) {
  await sleep(250);
  const r = await send("Runtime.evaluate", {
    expression: "(window.MathJax && MathJax.startup && MathJax.startup.promise) ? MathJax.startup.promise.then(()=>'done') : 'wait'",
    awaitPromise: true, returnByValue: true,
  });
  if (r.result && r.result.result && r.result.result.value === "done") { ok = true; break; }
}
await sleep(500);
const errs = await send("Runtime.evaluate", { expression: "document.querySelectorAll('mjx-merror, [data-mjx-error]').length", returnByValue: true });
const cnt = await send("Runtime.evaluate", { expression: "document.querySelectorAll('mjx-container').length", returnByValue: true });
const ovExpr = `JSON.stringify(Array.from(document.querySelectorAll('mjx-container[display="true"]')).filter(e => {
  const r = e.getBoundingClientRect();
  if (r.width === 0) return false;
  const main = e.closest('main') || document.body;
  const mr = main.getBoundingClientRect();
  let maxR = -1e9;
  e.querySelectorAll('*').forEach(c => { const cr = c.getBoundingClientRect(); if (cr.width > 0) maxR = Math.max(maxR, cr.right); });
  return maxR > Math.min(r.right, mr.right) + 1 || e.scrollWidth > e.clientWidth + 2;
}).map(e => { const p = e.closest('[id]'); return (p ? p.id : '?'); }))`;
const ov = await send("Runtime.evaluate", { expression: ovExpr, returnByValue: true });
console.log("overflowing displays:", ov.result.result.value);
const m = await send("Page.getLayoutMetrics");
const h = Math.ceil(m.result.cssContentSize.height);
await send("Emulation.setDeviceMetricsOverride", { width, height: h, deviceScaleFactor: 1, mobile: false });
await sleep(300);
const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: { x: 0, y: 0, width, height: h, scale: 1 } });
writeFileSync(out, Buffer.from(shot.result.data, "base64"));
console.log(`mathjax ${ok ? "done" : "TIMEOUT"}, containers ${cnt.result.result.value}, errors ${errs.result.result.value}, height ${h}`);
ws.close();
chrome.kill("SIGKILL");
process.exit(0);

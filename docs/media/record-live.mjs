// Records a live session of the web UI (docs/media/ui-live.webp) as PNG frames with
// headless Chromium: questions are typed and answered by the running API (LLM + verifier).
// Needs the API on http://127.0.0.1:8000 (CORS_ORIGINS=http://127.0.0.1:3100), a static
// build pointing at it, served on 127.0.0.1:3100, and puppeteer-core:
//   (cd apps/web && NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 pnpm build && python3 -m http.server 3100 -d out)
//   npm install puppeteer-core && node record-live.mjs frames/ [questions.json]   (then encode_demo.py)
// questions.json: [{"text": ..., "format": "free" | "yes_no" | "multiple_choice", "unverified": bool}]
import puppeteer from "puppeteer-core";
import fs from "node:fs";

const out = process.argv[2];
const DEFAULT_QUESTIONS = [
  {
    text: "Does increased use of private health care reduce the demand for NHS care?",
    format: "yes_no",
  },
  {
    text: "Is semaglutide effective for weight loss in adults without diabetes?",
    format: "free",
    unverified: true,
  },
];
const QUESTIONS = process.argv[3]
  ? JSON.parse(fs.readFileSync(process.argv[3], "utf-8"))
  : DEFAULT_QUESTIONS;

const browser = await puppeteer.launch({
  executablePath: process.env.CHROMIUM ?? "/snap/bin/chromium",
  headless: true,
  args: ["--no-sandbox", "--disable-gpu"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1000, height: 720, deviceScaleFactor: 1 });
await page.goto("http://127.0.0.1:3100/", { waitUntil: "networkidle0" });

// Sequential capture with a playback clock: each frame is taken after the page settled.
const frames = [];
let clock = 0;
let n = 0;
const settle = () =>
  page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
async function snap(showMs) {
  await settle();
  const file = `${out}/f${String(n++).padStart(4, "0")}.png`;
  await page.screenshot({ path: file, type: "png" });
  frames.push({ file, at: clock });
  clock += showMs;
}
async function scrollTo(y) {
  const from = await page.evaluate(() => window.scrollY);
  const steps = Math.max(1, Math.round(Math.abs(y - from) / 90));
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    await page.evaluate((v) => window.scrollTo(0, v), from + (y - from) * (1 - (1 - t) ** 2));
    await snap(45);
  }
}
const top = (sel) =>
  page.$eval(sel, (el) => el.getBoundingClientRect().top + window.scrollY - 12);

/** Waits for the real answer, capturing the streamed progress (played back faster). */
async function waitForAnswer() {
  const started = Date.now();
  while (!(await page.$("article.answer, .card.error"))) {
    if (Date.now() - started > 180_000) throw new Error("no answer within 3 minutes");
    await snap(350);
    await new Promise((r) => setTimeout(r, 600));
  }
}

async function ask({ text, format, unverified }) {
  await scrollTo(0);
  await page.$eval("#question", (el) => {
    el.value = "";
  });
  await page.click("#question", { clickCount: 3 });
  for (const word of text.split(" ")) {
    await page.type("#question", word + " ");
    await snap(90);
  }
  await page.select("select", format);
  const box = await page.$("input[type=checkbox]");
  if ((await box.evaluate((el) => el.checked)) !== Boolean(unverified)) await box.click();
  await snap(900);
  await page.click("button[type=submit]");
  await snap(300);
  await page.waitForSelector("section.progress");
  await scrollTo(await top("section.progress"));
  await waitForAnswer();
  await snap(900);
  await scrollTo(await top("article.answer"));
  await snap(2600);
  if (unverified && (await page.$("details.unverified summary"))) {
    await page.click("details.unverified summary");
    await snap(2600);
  }
  if (await page.$("ol.statements")) {
    await scrollTo(await top("ol.statements"));
    await snap(3200);
  }
  if (await page.$("ol.citations")) {
    await scrollTo(await top("ol.citations"));
    await snap(3400);
  }
}

for (const question of QUESTIONS) await ask(question);
fs.writeFileSync(`${out}/frames.json`, JSON.stringify(frames));
await browser.close();
console.log(frames.length, "frames,", (clock / 1000).toFixed(1), "s playback");

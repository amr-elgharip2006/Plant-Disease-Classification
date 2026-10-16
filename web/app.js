import { UI, TEAM, PLANTS_AR, CONDITIONS_AR } from "./i18n.js";
import { adviceFor } from "./recommendations.js";

const $ = (id) => document.getElementById(id);
const LOW_CONFIDENCE = 0.6;
const STEP_ICONS = ["camera", "brain", "search", "doc"];
const FIELD_ICONS = ["eye", "algo", "search"];
const BENEFIT_ICONS = ["globe", "leaf", "coins", "plant"];

let lang = "ar";
try { lang = localStorage.getItem("lang") || "ar"; } catch {}

let model = null;
let meta = null;
let results = null;
let lastPreds = null;

// ---------- i18n ----------

function t(key) { return UI[lang][key]; }

function plantName(c) { return lang === "ar" ? PLANTS_AR[c.plant] || c.plant : c.plant; }

function conditionName(c) {
  if (c.healthy) return t("healthy");
  return lang === "ar" ? CONDITIONS_AR[c.id] || c.condition : c.condition;
}

const pct = (p, digits = 1) => `${(p * 100).toFixed(digits)}%`;

// ---------- DOM helpers ----------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "ico");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

// ---------- static sections ----------

function renderStatic() {
  $("solution-points").replaceChildren(...t("solutionPoints").map((text, i) => {
    const li = el("li");
    li.append(el("span", "num", String(i + 1)), el("span", "", text));
    return li;
  }));

  $("steps").replaceChildren(...t("steps").map(([title, body], i) => {
    const li = el("li");
    const circle = el("div", "step-icon");
    circle.append(icon(STEP_ICONS[i]), el("span", "step-num", String(i + 1)));
    li.append(circle, el("strong", "", title), el("p", "", body));
    return li;
  }));

  $("fields").replaceChildren(...t("fields").map(([title, body], i) => {
    const li = el("li");
    const circle = el("span", "icon-circle lavender");
    circle.append(icon(FIELD_ICONS[i]));
    const text = el("div");
    text.append(el("strong", "", title), el("span", "desc", body));
    li.append(circle, text);
    return li;
  }));

  $("benefits").replaceChildren(...t("benefits").map((text, i) => {
    const li = el("li");
    const circle = el("span", "icon-circle sky");
    circle.append(icon(BENEFIT_ICONS[i]));
    li.append(circle, el("span", "", text));
    return li;
  }));

  $("team").replaceChildren(...TEAM.map((m) => {
    const li = el("li");
    const name = m[lang];
    li.append(el("span", "avatar", name.trim()[0]), el("span", "member", name));
    return li;
  }));

  renderStats();
  renderResults();
}

function renderStats() {
  const acc = meta?.test_accuracy ? pct(meta.test_accuracy, 2) : "99%+";
  $("acc-pill").textContent = t("accuracyPill").replace("{acc}", acc);
  $("stats").replaceChildren(...t("stats").map(([value, label], i) => {
    const div = el("div", `stat stat-${i}`);
    div.append(el("strong", "", value === "accuracy" ? acc : value), el("span", "", label));
    return div;
  }));
}

function renderResults() {
  if (!results) return;
  const labels = t("metricLabels");
  const items = [
    ["accuracy", pct(results.accuracy, 2)],
    ["top3", pct(results.top3_accuracy, 2)],
    ["macroF1", pct(results.macro_f1, 2)],
    ["testImages", results.test_images.toLocaleString(lang === "ar" ? "en-US" : "en-US")],
  ];
  $("metrics").replaceChildren(...items.map(([key, value]) => {
    const li = el("li");
    li.append(el("strong", "", value), el("span", "", labels[key]));
    return li;
  }));
}

function renderMeta() {
  const plants = [...new Set(meta.classes.map((c) => c.plant))];
  $("plants").replaceChildren(...plants.map((p) => el("li", "", plantName({ plant: p }))));
  renderStats();
}

function applyLanguage() {
  document.documentElement.lang = lang;
  document.documentElement.dir = t("dir");
  document.title = t("pageTitle");
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  $("lang").textContent = t("lang");
  renderStatic();
  if (meta) renderMeta();
  if (lastPreds) renderResult(lastPreds);
  renderBatch();
  setStatus(model ? "ready" : $("status").classList.contains("error") ? "error" : "loading");
}

$("lang").addEventListener("click", () => {
  lang = lang === "ar" ? "en" : "ar";
  try { localStorage.setItem("lang", lang); } catch {}
  applyLanguage();
});

// ---------- model ----------

function setStatus(state) {
  $("status").className = `pill status ${state}`;
  $("status-text").textContent = t({ loading: "loadingModel", ready: "modelReady", error: "modelError", busy: "analyzing" }[state]);
  document.querySelectorAll(".picker .btn, .batch-btn").forEach((b) => b.setAttribute("aria-disabled", state !== "ready"));
}

async function loadModel() {
  setStatus("loading");
  try {
    [meta, results] = await Promise.all([
      fetch("model/metadata.json").then((r) => r.json()),
      fetch("results.json").then((r) => (r.ok ? r.json() : null)).catch(() => null),
    ]);
    renderMeta();
    renderResults();
    model = await tf.loadGraphModel("model/model.json");
    // Warm-up so the first real prediction is fast.
    tf.tidy(() => model.predict(tf.zeros([1, meta.img_size, meta.img_size, 3])));
    setStatus("ready");
  } catch (err) {
    console.error(err);
    setStatus("error");
  }
}

// ---------- inference ----------

function drawCenterCrop(img, canvas = $("canvas")) {
  const size = meta.img_size;
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  const side = Math.min(img.width, img.height);
  ctx.drawImage(img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, size, size);
  return canvas;
}

async function predictFile(file, canvas = $("canvas"), topK = 4) {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  drawCenterCrop(bitmap, canvas);
  bitmap.close();
  // Model expects raw 0..255 RGB; EfficientNet normalizes internally.
  // Test-time augmentation: average the image with its horizontal and vertical flips.
  const probs = tf.tidy(() => {
    const x = tf.browser.fromPixels(canvas).toFloat().expandDims(0);
    const batch = tf.concat([x, tf.reverse(x, 2), tf.reverse(x, 1)]);
    return model.predict(batch).mean(0);
  });
  const data = await probs.data();
  probs.dispose();
  return [...data]
    .map((p, i) => ({ ...meta.classes[i], p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, topK);
}

async function classify(file) {
  if (!model) return;
  if (!file.type.startsWith("image/")) { alert(t("invalidFile")); return; }
  setStatus("busy");
  try {
    renderResult(await predictFile(file));
    $("dropzone").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    console.error(err);
    alert(t("invalidFile"));
  } finally {
    setStatus("ready");
  }
}

function renderResult(preds) {
  lastPreds = preds;
  const [best, ...rest] = preds;

  $("picker").hidden = true;
  $("result").hidden = false;
  $("badge").className = `badge ${best.healthy ? "ok" : "bad"}`;
  $("badge").textContent = best.healthy ? t("healthy") : t("diseased");
  $("plant").textContent = plantName(best);
  $("condition").textContent = conditionName(best);
  $("confidence").textContent = pct(best.p);
  $("meter-fill").style.width = pct(best.p);
  $("low").hidden = best.p >= LOW_CONFIDENCE;
  $("tips").replaceChildren(...adviceFor(best, lang).map((tip) => el("li", "", tip)));
  $("others").replaceChildren(...rest.map((c) => {
    const li = el("li");
    li.append(el("span", "", `${plantName(c)}: ${conditionName(c)}`), el("span", "", pct(c.p)));
    return li;
  }));
}

// ---------- input ----------

for (const id of ["file", "camera"]) {
  $(id).addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) classify(file);
    e.target.value = "";
  });
}

$("again").addEventListener("click", () => {
  lastPreds = null;
  $("result").hidden = true;
  $("picker").hidden = false;
});

const zone = $("dropzone");
zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
zone.addEventListener("drop", (e) => {
  e.preventDefault();
  zone.classList.remove("drag");
  const file = e.dataTransfer.files[0];
  if (file) classify(file);
});
document.addEventListener("paste", (e) => {
  const file = [...e.clipboardData.files].find((f) => f.type.startsWith("image/"));
  if (file) classify(file);
});

// ---------- batch test ----------

// Minimal CSV parser (handles quoted fields such as "Pepper,_bell___...").
function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  text = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((f) => f !== "")) rows.push(row);
      row = [];
    } else field += ch;
  }
  row.push(field);
  if (row.some((f) => f !== "")) rows.push(row);
  return rows;
}

async function readAnswers(file) {
  const [header, ...rows] = parseCsv(await file.text());
  const fileCol = header.indexOf("file");
  const idCol = header.indexOf("class_id");
  return new Map(rows.map((r) => [r[fileCol], r[idCol]]));
}

const fill = (key, vars) => Object.entries(vars).reduce((s, [k, v]) => s.replace(`{${k}}`, v), t(key));

let batchState = null;

// <bdi> keeps "0001.jpg" in order inside the right-to-left table.
function fileCell(name) {
  const td = el("td", "file");
  td.append(el("bdi", "", name));
  return td;
}

function renderBatch() {
  if (!batchState) return;
  const { rows, total, answers } = batchState;
  const checked = rows.filter((r) => r.truth);
  const correct = checked.filter((r) => r.truth.id === r.pred.id).length;
  const lines = [rows.length < total ? fill("batchProgress", { done: rows.length, total }) : fill("batchDone", { total })];
  if (checked.length) lines.push(fill("batchAccuracy", { acc: pct(correct / checked.length, 2), correct, checked: checked.length }));
  else if (!answers) lines.push(t("batchNoAnswers"));
  $("batch-summary").textContent = lines.join(" · ");
  $("batch-progress").style.width = pct(rows.length / total);
  $("batch-rows").replaceChildren(...rows.map((r) => {
    const tr = el("tr", r.truth ? (r.truth.id === r.pred.id ? "ok" : "bad") : "");
    const label = (c) => `${plantName(c)}: ${conditionName(c)}`;
    tr.append(
      fileCell(r.name),
      el("td", "", label(r.pred)),
      el("td", "num", pct(r.pred.p)),
      el("td", "", r.truth ? `${r.truth.id === r.pred.id ? "✓" : "✗"} ${label(r.truth)}` : "—"),
    );
    return tr;
  }));
}

$("batch").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  e.target.value = "";
  if (!model || !files.length) return;
  const csv = files.find((f) => f.name.toLowerCase().endsWith(".csv"));
  const images = files.filter((f) => f.type.startsWith("image/")).sort((a, b) => a.name.localeCompare(b.name));
  if (!images.length) { alert(t("invalidFile")); return; }
  const answers = csv ? await readAnswers(csv) : null;
  const byId = new Map(meta.classes.map((c) => [c.id, c]));
  batchState = { rows: [], total: images.length, answers };
  $("batch-out").hidden = false;
  setStatus("busy");
  const canvas = document.createElement("canvas");
  try {
    for (const file of images) {
      const [pred] = await predictFile(file, canvas, 1);
      const truthId = answers?.get(file.name);
      batchState.rows.push({ name: file.name, pred, truth: truthId ? byId.get(truthId) : null });
      if (batchState.rows.length % 10 === 0 || batchState.rows.length === images.length) {
        renderBatch();
        await new Promise((r) => setTimeout(r));  // let the page repaint
      }
    }
  } catch (err) {
    console.error(err);
    alert(t("invalidFile"));
  } finally {
    renderBatch();
    setStatus("ready");
  }
});

applyLanguage();
loadModel();

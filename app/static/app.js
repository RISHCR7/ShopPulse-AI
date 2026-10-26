(() => {
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const META = {
  overview: ["Overview", "Late-delivery prediction and review sentiment for e-commerce orders."],
  predict: ["Delivery Prediction", "Estimate whether an order will arrive late."],
  sentiment: ["Review Sentiment", "Classify the sentiment of a customer review."],
  models: ["Model Information", "The models behind ShopPulse-AI."],
  about: ["About Project", "Objective, data and technology."]
};
const pct = v => (v * 100).toFixed(1) + "%";

// ---- navigation
function show(view) {
  if (!META[view]) view = "overview";
  $$(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + view));
  $$(".nav").forEach(n => n.classList.toggle("active", n.dataset.view === view));
  $("#pageTitle").textContent = META[view][0];
  $("#pageDesc").textContent = META[view][1];
  history.replaceState(null, "", "#" + view);
  closeMenu(); window.scrollTo(0, 0);
}
$$(".nav").forEach(n => n.addEventListener("click", () => show(n.dataset.view)));
$$("[data-go]").forEach(b => b.addEventListener("click", () => show(b.dataset.go)));
const sb = $("#sidebar"), scrim = $("#scrim"), menuBtn = $("#menuBtn");
function closeMenu() { sb.classList.remove("open"); scrim.classList.remove("show"); menuBtn.setAttribute("aria-expanded", "false"); }
menuBtn.addEventListener("click", () => { sb.classList.add("open"); scrim.classList.add("show"); menuBtn.setAttribute("aria-expanded", "true"); });
scrim.addEventListener("click", closeMenu);
document.addEventListener("keydown", e => e.key === "Escape" && closeMenu());

// ---- theme
const root = document.documentElement;
try { if (localStorage.getItem("sp-theme") === "dark") root.dataset.theme = "dark"; } catch {}
$("#themeBtn").addEventListener("click", () => {
  const d = root.dataset.theme === "dark"; if (d) delete root.dataset.theme; else root.dataset.theme = "dark";
  try { localStorage.setItem("sp-theme", d ? "light" : "dark"); } catch {}
});

// ---- toast
function toast(msg, type = "") {
  const t = document.createElement("div"); t.className = "toast " + type; t.textContent = msg;
  $("#toasts").append(t); setTimeout(() => t.remove(), 4500);
}

// ---- health (real GET /health)
async function health() {
  const el = $("#health"), txt = $("span", el), det = $("#healthDetail");
  try {
    const r = await fetch("/health"); if (!r.ok) throw new Error("HTTP " + r.status);
    const j = await r.json();
    el.className = "status ok"; txt.textContent = "API connected";
    det.textContent = "GET /health responded: " + JSON.stringify(j);
  } catch (e) {
    el.className = "status bad"; txt.textContent = "API unreachable";
    det.textContent = "GET /health failed (" + e.message + ").";
  }
}
health(); setInterval(health, 30000);

// ---- helpers
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  let data = null; try { data = await r.json(); } catch {}
  if (!r.ok) {
    let d = data && data.detail;
    if (Array.isArray(d)) d = d.map(x => (x.loc || []).slice(1).join(".") + ": " + x.msg).join("; ");
    throw new Error(d || "Request failed (HTTP " + r.status + ")");
  }
  return data;
}
function busy(btn, on, label) { btn.disabled = on; btn.classList.toggle("loading", on); if (label) btn.firstChild.textContent = label; }
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

// ---- selects
const MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"];
const DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];
function fill(id, items, valueFn) {
  const s = $("#" + id); s.append(new Option("Select…", ""));
  items.forEach((t, i) => s.append(new Option(t, valueFn(i))));
}
fill("purchase_month", MONTHS, i => i + 1);
fill("purchase_weekday", DAYS, i => i);   // pandas convention: Monday = 0
fill("purchase_hour", Array.from({ length: 24 }, (_, h) => String(h).padStart(2, "0") + ":00"), i => i);

// ---- prediction
const FIELDS = ["item_count","order_value","freight_value","payment_total","payment_installments_max","estimated_delivery_days","purchase_month","purchase_weekday","purchase_hour"];
let predicting = false;
$("#predForm").addEventListener("submit", async e => {
  e.preventDefault(); if (predicting) return;
  const body = {}; let bad = [];
  FIELDS.forEach(id => {
    const i = $("#" + id); i.classList.remove("bad");
    const raw = i.value.trim(); const n = Number(raw);
    const min = i.min !== "" ? Number(i.min) : null, max = i.max !== "" ? Number(i.max) : null;
    const isInt = i.step === "1" || i.tagName === "SELECT";
    if (raw === "" || !Number.isFinite(n) || (min !== null && n < min) || (max !== null && n > max) || (isInt && !Number.isInteger(n))) { bad.push(i); i.classList.add("bad"); }
    else body[id] = n;
  });
  const err = $("#formErr");
  if (bad.length) { err.textContent = "Please correct the highlighted fields."; bad[0].focus(); return; }
  err.textContent = "";
  const dl = $("input[name=model]:checked").value === "dl";
  const btn = $("#predBtn"); predicting = true; busy(btn, true);
  try {
    const res = await post(dl ? "/predict/dl" : "/predict", body);
    renderPred(res, dl); saveHist(res, dl, body);
  } catch (ex) { err.textContent = ex.message; toast(ex.message, "error"); }
  finally { predicting = false; busy(btn, false); }
});

function renderPred(r, dl) {
  const box = $("#predResult"); box.replaceChildren();
  const p = typeof r.late_probability === "number" ? r.late_probability : null;
  let late = typeof r.late_delivery_prediction !== "undefined" ? !!Number(r.late_delivery_prediction) : (p !== null && r.threshold != null ? p >= r.threshold : null);
  const label = r.label || (late === null ? "Result" : late ? "Late delivery" : "On-time delivery");
  const cls = late === null ? "neutral" : late ? "late" : "ontime";
  box.className = "card result " + cls;
  const v = el("div", "verdict");
  if (p !== null) { const ring = el("div", "ring"); ring.style.setProperty("--p", (p * 100).toFixed(1)); ring.append(el("span", "", pct(p))); v.append(ring); }
  const t = el("div"); t.append(el("div", "badge", (late ? "⚠ " : late === false ? "✓ " : "") + label));
  t.append(el("p", "muted", late === null ? "The API did not return a prediction flag." : late ? "This order is predicted to arrive after the estimated date." : "This order is predicted to arrive on or before the estimated date."));
  v.append(t); box.append(v);
  if (p !== null) {
    box.append(el("div", "lbl", "Late-delivery probability")); box.lastChild.append(el("span", "", pct(p)));
    const bar = el("div", "bar"); bar.append(document.createElement("i"));
    if (typeof r.threshold === "number") { const u = document.createElement("u"); u.style.left = r.threshold * 100 + "%"; u.title = "Threshold " + r.threshold; bar.append(u); }
    box.append(bar); requestAnimationFrame(() => bar.firstChild.style.width = p * 100 + "%");
  }
  const kv = el("dl", "kv");
  const add = (k, val) => { if (val != null) { kv.append(el("dt", "", k), el("dd", "", String(val))); } };
  add("Model", r.model || (dl ? "Neural Network (Deep Learning)" : "Random Forest (Machine Learning)"));
  add("Endpoint", dl ? "POST /predict/dl" : "POST /predict");
  add("Threshold", typeof r.threshold === "number" ? r.threshold : null);
  box.append(kv);
}

// ---- history (browser only)
const HK = "sp-history";
const loadHist = () => { try { return JSON.parse(localStorage.getItem(HK)) || []; } catch { return []; } };
function saveHist(r, dl, body) {
  const h = loadHist(); h.unshift({ t: Date.now(), dl, label: r.label || "", p: r.late_probability, v: body.order_value, d: body.estimated_delivery_days });
  try { localStorage.setItem(HK, JSON.stringify(h.slice(0, 8))); } catch {}
  drawHist();
}
function drawHist() {
  const ul = $("#hist"); ul.replaceChildren(); const h = loadHist();
  if (!h.length) { ul.append(el("li", "muted", "No predictions yet.")); return; }
  h.forEach(x => {
    const li = el("li"); const a = el("span", "", (x.label || "Prediction") + (typeof x.p === "number" ? " · " + pct(x.p) : ""));
    const b = el("span", "muted", (x.dl ? "Neural Net" : "Random Forest") + " · " + new Date(x.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    li.append(a, b); ul.append(li);
  });
}
$("#clearHist").addEventListener("click", () => { try { localStorage.removeItem(HK); } catch {} drawHist(); });
drawHist();

// ---- sentiment
const ta = $("#review_text");
ta.addEventListener("input", () => $("#count").textContent = ta.value.length + " / 2000");
let analyzing = false;
$("#senForm").addEventListener("submit", async e => {
  e.preventDefault(); if (analyzing) return;
  const text = ta.value.trim(), err = $("#senErr");
  if (!text) { err.textContent = "Please enter a review."; ta.focus(); return; }
  err.textContent = ""; analyzing = true; const btn = $("#senBtn"); busy(btn, true);
  try { renderSen(await post("/sentiment", { review_text: text })); }
  catch (ex) { err.textContent = ex.message; toast(ex.message, "error"); }
  finally { analyzing = false; busy(btn, false); }
});
function renderSen(r) {
  const box = $("#senResult"); box.replaceChildren();
  const s = String(r.sentiment ?? r.sentiment_prediction ?? "Unknown");
  const neg = /neg/i.test(s), pos = /pos/i.test(s);
  box.className = "card result " + (neg ? "late" : pos ? "ontime" : "neutral");
  box.append(el("div", "badge", (pos ? "✓ " : neg ? "⚠ " : "") + s.charAt(0).toUpperCase() + s.slice(1)));
  let cp = r.class_probabilities, entries = [];
  if (Array.isArray(cp)) entries = cp.map((v, i) => ["Class " + i, v]);
  else if (cp && typeof cp === "object") entries = Object.entries(cp);
  const wrap = el("div"); wrap.style.marginTop = "16px";
  entries.forEach(([k, v]) => {
    if (typeof v !== "number") return;
    const l = el("div", "lbl"); l.append(el("span", "", k), el("span", "", pct(v)));
    const bar = el("div", "bar"); bar.append(document.createElement("i")); wrap.append(l, bar);
    requestAnimationFrame(() => bar.firstChild.style.width = v * 100 + "%");
  });
  box.append(wrap, el("p", "muted small", "Probabilities are the sentiment model's class probabilities for this text."));
}

show((location.hash || "#overview").slice(1));
})();

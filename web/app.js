/* Spare parts dashboard.  Vanilla JS + hand-drawn SVG charts: no CDN, so the
   page works on an air-gapped ops box. */

const state = {
  overview: null,
  site: "",
  part: "",
  detail: null,
};

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const fmt = (n, d = 0) =>
  n == null || !isFinite(n) ? "–" :
  Number(n).toLocaleString("zh-CN", { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (n, d = 1) => (n == null ? "–" : (n * 100).toFixed(d) + "%");
const cny = (n) => {
  if (n == null || !isFinite(n)) return "–";
  if (Math.abs(n) >= 1e8) return (n / 1e8).toFixed(2) + " 亿";
  if (Math.abs(n) >= 1e4) return (n / 1e4).toFixed(1) + " 万";
  return fmt(n, 0);
};
const policyColor = { fixed: "#5aa9ff", hist: "#f59e0b", ours: "#34d399" };

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " → " + r.status);
  return r.json();
}

/* ------------------------------------------------------------------ charts */
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function svgRoot(host) {
  host.innerHTML = "";
  const box = host.getBoundingClientRect();
  const w = Math.max(box.width, 240), h = Math.max(box.height, 120);
  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", height: "100%" }, host);
  return { svg, w, h };
}
function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const f = v / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}
function tooltip() {
  let t = $(".tooltip");
  if (!t) { t = document.createElement("div"); t.className = "tooltip"; document.body.appendChild(t); }
  return t;
}
function bindTip(node, html) {
  const t = tooltip();
  node.addEventListener("mousemove", (e) => {
    t.innerHTML = html; t.style.opacity = 1;
    t.style.left = Math.min(e.clientX + 14, innerWidth - 250) + "px";
    t.style.top = (e.clientY + 16) + "px";
  });
  node.addEventListener("mouseleave", () => (t.style.opacity = 0));
}

function lineChart(host, series, opts = {}) {
  const { svg, w, h } = svgRoot(host);
  const plot = { l: 48, t: 18, r: w - 14, b: h - 26 };
  plot.w = plot.r - plot.l; plot.h = plot.b - plot.t;
  const n = Math.max(...series.map(s => s.y.length));
  const all = series.flatMap(s => s.y).filter(v => v != null && isFinite(v));
  const ymin = opts.ymin != null ? opts.ymin
    : (opts.zeroBased === false ? Math.max(0, Math.min(...all) * 0.94) : 0);
  const rawMax = Math.max(...all, 1) * 1.06;
  const ymax = opts.zeroBased === false
    ? ymin + niceMax(rawMax - ymin) : niceMax(rawMax);
  const x = i => plot.l + (n <= 1 ? plot.w / 2 : (i / (n - 1)) * plot.w);
  const y = v => plot.b - ((v - ymin) / Math.max(ymax - ymin, 1e-9)) * plot.h;
  for (let k = 0; k <= 4; k++) {
    const v = ymin + ((ymax - ymin) * k) / 4, yy = y(v);
    el("line", { x1: plot.l, x2: plot.r, y1: yy, y2: yy, stroke: "#232833" }, svg);
    el("text", { x: plot.l - 8, y: yy + 4, fill: "#8b93a3", "font-size": 10.5, "text-anchor": "end" }, svg)
      .textContent = opts.axisFmt ? opts.axisFmt(v) : fmt(v, ymax < 10 ? 1 : 0);
  }
  series.forEach((s, si) => {
    const color = s.color || ["#5aa9ff", "#34d399", "#f59e0b", "#ef4444"][si % 4];
    if (s.band) {
      const [lo, hi] = s.band;
      let d = "";
      for (let i = 0; i < n; i++) d += `${i ? "L" : "M"}${x(i)},${y(hi[i] ?? 0)}`;
      for (let i = n - 1; i >= 0; i--) d += `L${x(i)},${y(lo[i] ?? 0)}`;
      el("path", { d: d + "Z", fill: color, opacity: 0.13 }, svg);
    }
    if (s.bars) {
      const bw = Math.max(2, (plot.w / n) * 0.6);
      for (let i = 0; i < n; i++) {
        const r = el("rect", {
          x: x(i) - bw / 2, y: y(s.y[i] || 0), width: bw,
          height: Math.max(plot.b - y(s.y[i] || 0), 0.5),
          fill: color, opacity: 0.85, rx: 1.5,
        }, svg);
        bindTip(r, `<b>${s.label}</b> ${fmt(s.y[i], 1)}${opts.unit || ""}`);
      }
    } else {
      let d = "";
      for (let i = 0; i < n; i++) d += `${i ? "L" : "M"}${x(i)},${y(s.y[i] || 0)}`;
      el("path", {
        d, fill: "none", stroke: color, "stroke-width": 2,
        "stroke-linejoin": "round", "stroke-dasharray": s.dashed ? "5 4" : null,
      }, svg);
      if (opts.dots !== false && n <= 40) {
        for (let i = 0; i < n; i++) {
          const c = el("circle", { cx: x(i), cy: y(s.y[i] || 0), r: 2.3, fill: color }, svg);
          if (opts.tip) bindTip(c, opts.tip(i, s));
        }
      }
    }
  });
  const labels = opts.labels || [];
  const step = Math.max(1, Math.ceil(n / 9));
  for (let i = 0; i < n; i += step)
    el("text", { x: x(i), y: plot.b + 16, fill: "#8b93a3", "font-size": 10.5, "text-anchor": "middle" }, svg)
      .textContent = labels[i] ?? i;
  if (opts.legend !== false) {
    let lx = plot.l;
    series.forEach((s, si) => {
      const color = s.color || ["#5aa9ff", "#34d399", "#f59e0b", "#ef4444"][si % 4];
      el("rect", { x: lx, y: 2, width: 10, height: 10, rx: 2, fill: color }, svg);
      el("text", { x: lx + 14, y: 11, fill: "#8b93a3", "font-size": 10.5 }, svg).textContent = s.label;
      lx += 22 + s.label.length * 10.5;
    });
  }
}

function hBarChart(host, rows, opts = {}) {
  const { svg, w, h } = svgRoot(host);
  const padL = Math.min(180, Math.max(70, opts.labelWidth || 110)), padR = 60;
  const max = niceMax(Math.max(...rows.map(r => Math.abs(r.v)), 1));
  const bh = (h - 16) / Math.max(rows.length, 1);
  rows.forEach((r, i) => {
    const yy = 8 + i * bh;
    const bw = ((w - padL - padR) * Math.abs(r.v)) / max;
    el("text", { x: padL - 8, y: yy + bh * 0.62, fill: "#cbd5e1", "font-size": 11, "text-anchor": "end" }, svg)
      .textContent = r.k;
    const rect = el("rect", {
      x: padL, y: yy + bh * 0.18, width: Math.max(bw, 1),
      height: Math.max(bh * 0.62, 2), rx: 3, fill: r.color || "#5aa9ff", opacity: 0.9,
    }, svg);
    el("text", { x: padL + Math.max(bw, 1) + 6, y: yy + bh * 0.62, fill: "#8b93a3", "font-size": 10.5 }, svg)
      .textContent = opts.fmt ? opts.fmt(r.v) : fmt(r.v, 1);
    if (r.tip) bindTip(rect, r.tip);
  });
}

/* ------------------------------------------------------------------- tabs */
function initTabs() {
  $$("#tabs button").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("#tabs button").forEach(b => b.classList.toggle("active", b === btn));
      $$(".tab").forEach(t => t.classList.toggle("active", t.id === "tab-" + btn.dataset.tab));
      renderTab(btn.dataset.tab);
    });
  });
}
function renderTab(tab) {
  if (!state.overview) return;
  if (tab === "overview") renderOverview();
  if (tab === "demand") { if (!$("#sel-site").options.length) fillSelects(); loadForecast(); fillForecastTable(); }
  if (tab === "life") renderLife();
  if (tab === "network") renderNetwork();
  if (tab === "lastbuy") renderLastBuy();
  if (tab === "logs") loadLogs();
}
let rid = 0;
window.addEventListener("resize", () => {
  clearTimeout(rid);
  rid = setTimeout(() => renderTab($(".tab.active").id.replace("tab-", "")), 160);
});

/* --------------------------------------------------------------- overview */
function card(k, v, s, cls = "") {
  return `<div class="card ${cls}"><div class="k">${k}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
}
function renderOverview() {
  const b = state.overview;
  const m = b.meta;
  const best = b.results.find(r => r.policy === b.best) || b.results[0];
  const c = b.calibration.overall;
  const worst = b.results.reduce((a, r) => (r.total_cost > a.total_cost ? r : a), b.results[0]);
  $("#subtitle").textContent =
    `${fmt(m.total_units_installed)} 台服务器 · ${m.sites} 机房 · ${m.hubs} 中心仓 · ${m.parts} 种零件 · 数据截止 ${m.as_of_year}-01-01 · 仿真 ${m.sim_years.toFixed(0)} 年`;
  $("#footer-note").textContent =
    `结果指纹 ${m.fingerprint} · 生成于 ${m.generated} · 仿真 ${fmt(m.sim_days)} 天`
    + (b.run_seconds ? ` · 计算 ${b.run_seconds}s` : "");

  $("#headline").innerHTML = [
    card("推荐策略", "两级网络 + 调拨",
      `运营成本 ${cny(best.total_cost)} · 比最差省 ${pct(1 - best.total_cost / worst.total_cost, 0)}`, "good"),
    card("运营成本（最优）", cny(best.total_cost),
      `固定 ${cny(b.results[0].total_cost)} · 历史均值 ${cny(b.results[1].total_cost)}`),
    card("缺货次数（最优）", fmt(best.stockout_events),
      `未满足 ${fmt(best.unfilled || 0)} 次 · 网络满足率 ${pct(best.csl_network)}`),
    card("平均等待备件", best.wait_mean.toFixed(2) + " 天",
      `P95 ${best.wait_p95.toFixed(0)} 天 · 对比 ${b.results[0].wait_mean.toFixed(2)} 天`, "warn"),
    card("可用性（最优）", pct(best.availability, 3), "按零件停机权重折算"),
    card("预测 80% 区间覆盖", pct(c.cov80, 1),
      `名义 80% · ${fmt(c.n)} 个观测格`, c.cov80 >= 0.75 ? "good" : "warn"),
  ].join("");

  const cols = [
    ["策略", r => r.policy],
    ["运营成本(元)", r => fmt(r.total_cost)],
    ["持有", r => fmt(r.holding)],
    ["订货", r => fmt(r.ordering)],
    ["调拨", r => fmt(r.transfer)],
    ["缺货损失", r => fmt(r.shortage)],
    ["缺货次数", r => fmt(r.stockout_events)],
    ["未满足", r => fmt(r.unfilled)],
    ["平均等待(天)", r => r.wait_mean.toFixed(2)],
    ["P95等待", r => r.wait_p95.toFixed(0)],
    ["可用性", r => pct(r.availability, 3)],
    ["网络满足率", r => pct(r.csl_network)],
    ["调拨次数", r => fmt(r.transfers)],
    ["平均库存值", r => cny(r.mean_inv_value)],
  ];
  $("#policy-table").innerHTML =
    `<thead><tr>${cols.map(([h]) => `<th>${h}</th>`).join("")}</tr></thead><tbody>` +
    b.results.map(r => `<tr class="${r.policy === b.best ? "best" : ""}">` +
      cols.map(([, f]) => `<td>${f(r)}</td>`).join("") + "</tr>").join("") + "</tbody>";

  hBarChart($("#chart-cost"), b.results.map(r => ({
    k: r.policy, v: r.total_cost, color: policyColor[r.kind],
    tip: `<b>${r.policy}</b><br>运营成本 ${cny(r.total_cost)}<br>持有 ${cny(r.holding)} · 缺货 ${cny(r.shortage)}`,
  })), { fmt: cny });

  const { svg, w, h } = svgRoot($("#chart-outcome"));
  const names = b.results.map(r => r.policy);
  const maxStock = Math.max(...b.results.map(r => r.stockout_events), 1);
  const maxWait = Math.max(...b.results.map(r => r.wait_mean), 1);
  const gw = (w - 20) / names.length;
  names.forEach((nm, i) => {
    const r = b.results[i];
    const xc = 10 + gw * i + gw / 2;
    const bh1 = (r.stockout_events / maxStock) * (h - 74);
    const bh2 = (r.wait_mean / maxWait) * (h - 74);
    const r1 = el("rect", { x: xc - 48, y: h - 34 - bh1, width: 40, height: Math.max(bh1, 1), fill: "#ef4444", rx: 3 }, svg);
    const r2 = el("rect", { x: xc + 8, y: h - 34 - bh2, width: 40, height: Math.max(bh2, 1), fill: "#f59e0b", rx: 3 }, svg);
    bindTip(r1, `<b>${nm}</b><br>缺货 ${fmt(r.stockout_events)} 次`);
    bindTip(r2, `<b>${nm}</b><br>平均等待 ${r.wait_mean.toFixed(2)} 天`);
    el("text", { x: xc - 28, y: h - 38 - bh1, fill: "#ef4444", "font-size": 11, "text-anchor": "middle" }, svg).textContent = fmt(r.stockout_events);
    el("text", { x: xc + 28, y: h - 38 - bh2, fill: "#f59e0b", "font-size": 11, "text-anchor": "middle" }, svg).textContent = r.wait_mean.toFixed(1) + "d";
    el("text", { x: xc, y: h - 14, fill: "#cbd5e1", "font-size": 11.5, "text-anchor": "middle" }, svg).textContent = nm;
  });
  el("rect", { x: 8, y: 2, width: 10, height: 10, fill: "#ef4444", rx: 2 }, svg);
  el("text", { x: 22, y: 11, fill: "#8b93a3", "font-size": 10.5 }, svg).textContent = "缺货次数";
  el("rect", { x: 86, y: 2, width: 10, height: 10, fill: "#f59e0b", rx: 2 }, svg);
  el("text", { x: 100, y: 11, fill: "#8b93a3", "font-size": 10.5 }, svg).textContent = "平均等待(天)";

  const cal = b.calibration;
  $("#calibration").innerHTML = [
    card("80% 区间实际覆盖", pct(cal.overall.cov80, 1), "名义 80%", cal.overall.cov80 >= 0.75 ? "good" : "warn"),
    card("50% 区间实际覆盖", pct(cal.overall.cov50, 1), "名义 50%（区间偏宽）", "warn"),
    card("整体偏差", pct(cal.overall.aggregate_bias, 1), "预测总量 vs 实际",
      Math.abs(cal.overall.aggregate_bias) < 0.08 ? "good" : "warn"),
    card("区间平均宽度", fmt(cal.overall.mean_width80, 1), "80% 预测带"),
  ].join("");
  lineChart($("#chart-calib"), [
    { label: "预测", y: cal.monthly_pred, color: "#5aa9ff" },
    { label: "实际", y: cal.monthly_actual, color: "#f59e0b", dashed: true },
  ], { labels: cal.monthly_pred.map((_, i) => "M" + (i + 1)) });
  hBarChart($("#chart-calib-part"),
    cal.by_part.slice().sort((a, b) => b.actual - a.actual).slice(0, 12).map(p => ({
      k: p.part, v: p.ratio,
      color: Math.abs(p.ratio - 1) < 0.15 ? "#34d399" : "#f59e0b",
      tip: `<b>${p.part}</b><br>预测 ${fmt(p.predicted)} / 实际 ${fmt(p.actual)}`,
    })), { fmt: v => v.toFixed(2) + "×", labelWidth: 130 });
}

/* ----------------------------------------------------------------- demand */
function fillSelects() {
  const sites = state.overview.site_info.sites.map(s => s.code);
  const parts = state.overview.site_info.parts.map(p => p.code);
  $("#sel-site").innerHTML = `<option value="">全部机房</option>` +
    sites.map(s => `<option value="${s}">${s}</option>`).join("");
  $("#sel-part").innerHTML = `<option value="">全部零件</option>` +
    parts.map(p => `<option value="${p}">${p}</option>`).join("");
}
async function loadForecast() {
  const q = new URLSearchParams();
  if (state.site) q.set("site", state.site);
  if (state.part) q.set("part", state.part);
  const fc = await getJSON("/api/forecast?" + q.toString());
  $("#fc-scope").textContent = fc.scope;
  lineChart($("#chart-forecast"), [
    { label: "预测均值", y: fc.mean, color: "#5aa9ff", band: [fc.q05, fc.q95] },
    { label: "90% 区间下界", y: fc.q05, color: "#34d399", dashed: true },
    { label: "90% 区间上界", y: fc.q95, color: "#f59e0b", dashed: true },
  ], {
    labels: fc.labels, dots: false,
    tip: (i) => `<b>${fc.labels[i]}</b><br>均值 ${fmt(fc.mean[i], 1)}<br>区间 ${fmt(fc.q05[i], 1)} – ${fmt(fc.q95[i], 1)}`,
  });
}
function fillForecastTable() {
  const b = state.overview;
  const fc = b.forecast;
  const sites = b.site_info.sites.map(s => s.code);
  const parts = b.site_info.parts.map(p => p.code);
  const H = fc.months.length;
  const ann = [];
  for (let si = 0; si < sites.length; si++) {
    for (let pi = 0; pi < parts.length; pi++) {
      let sum = 0, lo = 0, hi = 0, installed = 0;
      for (let k = 0; k < H; k++) {
        sum += fc.mean[si][pi][k];
        lo += fc.q05[si][pi][k];
        hi += fc.q95[si][pi][k];
        if (k < 12) installed += fc.installed[si][pi][k];
      }
      const a = (sum * 12) / H;
      if (a < 0.5) continue;
      ann.push([sites[si], parts[pi], a, (lo * 12) / H, (hi * 12) / H, installed / Math.min(12, H)]);
    }
  }
  ann.sort((x, y) => y[2] - x[2]);
  $("#fc-table").innerHTML =
    `<thead><tr><th>机房</th><th>零件</th><th>年化需求(件)</th><th>年化 5% 下界</th><th>年化 95% 上界</th><th>在装量</th></tr></thead><tbody>` +
    ann.slice(0, 200).map(([s, p, a, lo, hi, inst]) =>
      `<tr data-site="${s}" data-part="${p}" style="cursor:pointer"><td>${s}</td><td>${p}</td>
       <td>${fmt(a, 1)}</td><td>${fmt(lo, 1)}</td><td>${fmt(hi, 1)}</td><td>${fmt(inst)}</td></tr>`
    ).join("") + "</tbody>";
  $$("#fc-table tbody tr").forEach(tr => tr.addEventListener("click", () => {
    state.site = tr.dataset.site; state.part = tr.dataset.part;
    $("#sel-site").value = state.site; $("#sel-part").value = state.part;
    loadForecast();
  }));
}

/* ------------------------------------------------------------------- life */
function renderLife() {
  const b = state.overview;
  const curves = b.curves;
  const parts = Object.keys(curves);
  $("#life-table").innerHTML =
    `<thead><tr><th>零件</th><th>拟合形状 k</th><th>真值 k</th><th>拟合寿命(年)</th><th>真值</th><th>底噪声/年</th><th>真值</th><th>失效样本</th><th>暴露(件年)</th></tr></thead><tbody>` +
    parts.map(p => {
      const c = curves[p];
      return `<tr data-part="${p}" style="cursor:pointer"><td>${p}</td>
        <td>${c.shape.toFixed(2)}</td><td>${c.true_shape.toFixed(2)}</td>
        <td>${c.mean_life.toFixed(1)}</td><td>${c.true_life.toFixed(1)}</td>
        <td>${(c.floor_hazard * 100).toFixed(2)}%</td><td>${(c.true_floor * 100).toFixed(2)}%</td>
        <td>${fmt(c.n_failures)}</td><td>${fmt(c.exposure)}</td></tr>`;
    }).join("") + "</tbody>";
  const first = parts.includes("HDD-2T-SATA") ? "HDD-2T-SATA" : parts[0];
  drawHazard(first);
  $$("#life-table tbody tr").forEach(tr => tr.addEventListener("click", () => drawHazard(tr.dataset.part)));
  drawSiteMult();
  drawLot();
}
function drawHazard(part) {
  const c = state.overview.curves[part];
  if (!c) return;
  $("#life-part").textContent = part;
  const curve = c.curve;
  lineChart($("#chart-hazard"), [
    { label: "拟合(平滑)", y: curve.map(r => r.smoothed * 100), color: "#5aa9ff" },
    { label: "参数曲线", y: curve.map(r => r.fitted * 100), color: "#34d399", dashed: true },
    { label: "经验值", y: curve.map(r => r.empirical * 100), color: "#f59e0b", bars: true },
  ], {
    labels: curve.map(r => r.bucket), unit: "%/年", axisFmt: v => v.toFixed(0) + "%",
    tip: (i) => `<b>${curve[i].bucket}</b><br>暴露 ${fmt(curve[i].exposure)} 件年<br>失效 ${fmt(curve[i].failures)}`,
  });
}
function drawSiteMult() {
  const eff = state.overview.effects;
  const parts = state.overview.site_info.parts.map(p => p.code);
  const pi = parts.includes("FAN-40MM") ? parts.indexOf("FAN-40MM") : 0;
  const rows = eff.sites.map((s, i) => ({
    k: s, v: eff.site_mult[pi][i],
    color: eff.site_mult[pi][i] > 1.15 ? "#ef4444" : eff.site_mult[pi][i] < 0.85 ? "#34d399" : "#5aa9ff",
    tip: `<b>${s}</b><br>${parts[pi]} 故障率乘数 ${eff.site_mult[pi][i].toFixed(2)}`,
  })).sort((a, b) => b.v - a.v);
  hBarChart($("#chart-site-mult"), rows, { fmt: v => v.toFixed(2) + "×", labelWidth: 54 });
}
function drawLot() {
  const eff = state.overview.effects;
  const sample = eff.lot_mult_sample;
  const bins = 26;
  const lo = 0.3, hi = Math.max(1.8, Math.max(...sample));
  const hist = new Array(bins).fill(0);
  sample.forEach(v => {
    const i = Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)));
    hist[i]++;
  });
  const { svg, w, h } = svgRoot($("#chart-lot"));
  const padL = 40, padB = 22, padT = 8;
  const ymax = niceMax(Math.max(...hist, 1));
  hist.forEach((v, i) => {
    const bw = (w - padL - 8) / bins;
    const bh = (v / ymax) * (h - padT - padB);
    const r = el("rect", { x: padL + i * bw, y: h - padB - bh, width: bw * 0.86, height: Math.max(bh, 0.5), fill: "#5aa9ff", opacity: 0.85, rx: 1 }, svg);
    const mid = lo + ((hi - lo) * (i + 0.5)) / bins;
    bindTip(r, `<b>质量乘数≈${mid.toFixed(2)}×</b><br>${v} 个批次`);
  });
  for (let k = 0; k <= 3; k++) {
    const v = (ymax * k) / 3, yy = h - padB - (v / ymax) * (h - padT - padB);
    el("text", { x: padL - 6, y: yy + 4, fill: "#8b93a3", "font-size": 10, "text-anchor": "end" }, svg).textContent = fmt(v);
    el("line", { x1: padL, x2: w - 8, y1: yy, y2: yy, stroke: "#232833" }, svg);
  }
  el("text", { x: padL, y: h - 6, fill: "#8b93a3", "font-size": 10, "text-anchor": "start" }, svg).textContent = lo.toFixed(1) + "×";
  el("text", { x: w - 8, y: h - 6, fill: "#8b93a3", "font-size": 10, "text-anchor": "end" }, svg).textContent = hi.toFixed(1) + "×";
  el("text", { x: padL, y: 12, fill: "#8b93a3", "font-size": 10.5 }, svg)
    .textContent = `${fmt(eff.lot_count)} 个批次 · 抽样 ${sample.length} · 收缩后乘数 CV ${eff.lot_cv}`;
}

/* ---------------------------------------------------------------- network */
async function renderNetwork() {
  const b = state.overview;
  const det = state.detail || (state.detail = await getJSON("/api/detail"));
  const sites = b.site_info.sites;
  const hubs = b.site_info.hubs;
  const parts = b.site_info.parts.map(p => p.code);
  const inv = det.inventory_end.site;
  const hubInv = det.inventory_end.hub;
  const hubCodes = det.inventory_end.hubs;

  const totals = hubInv.map(r => r.reduce((a, c) => a + c, 0));
  const maxUnits = Math.max(1, ...totals);
  $("#network").innerHTML = hubs.map(h => {
    const hi = hubCodes.indexOf(h.code);
    const units = totals[hi];
    const own = sites.filter(s => s.hub === h.code);
    const siteTotal = own.reduce((acc, s) => acc + inv[sites.indexOf(s)].reduce((a, c) => a + c, 0), 0);
    return `<div class="hub-card">
      <h4>${h.name}</h4>
      <div class="muted">${own.length} 个机房 · 中心仓 ${fmt(units)} 件 · 前置仓合计 ${fmt(siteTotal)} 件</div>
      <div class="hubbar"><div style="width:${(units / maxUnits) * 100}%"></div></div>
      <div class="sites">${own.map(s => {
        const u = inv[sites.indexOf(s)].reduce((a, c) => a + c, 0);
        return `<span class="site-chip">${s.code} <b>${fmt(u)}</b></span>`;
      }).join("")}</div>
    </div>`;
  }).join("");

  const hm = $("#heatmap");
  const maxV = Math.max(1, ...inv.flat());
  // a red border means "this cell cannot serve demand": zero stock while the
  // part still has forecast demand at that site, and it is not an EOL part
  // (EOL parts are served from the central last-time-buy reserve instead)
  const eol = new Set(Object.keys(b.meta.discontinued || {}));
  const demand = b.forecast.mean;
  const H = b.forecast.months.length;
  let html = `<div class="heat-row"><div class="heat-cell lbl"></div>` +
    parts.map(p => `<div class="heat-cell hdr" title="${p}">${p}</div>`).join("") + `</div>`;
  sites.forEach((s, si) => {
    html += `<div class="heat-row"><div class="heat-cell lbl">${s.code}</div>`;
    parts.forEach((p, pi) => {
      const v = inv[si][pi];
      const t = Math.min(1, Math.log1p(v) / Math.log1p(maxV));
      const col = `rgba(90,169,255,${0.08 + t * 0.85})`;
      let monthly = 0;
      if (demand && demand[si] && demand[si][pi])
        for (let k = 0; k < H; k++) monthly += demand[si][pi][k];
      monthly /= Math.max(H, 1);
      const stockout = (v < 1 && !eol.has(p) && monthly > 0.2) ? " stockout" : "";
      html += `<div class="heat-cell${stockout}" style="background:${col}" title="${s.code} / ${p}: ${fmt(v)} 件">${v < 100 ? fmt(v) : ""}</div>`;
    });
    html += "</div>";
  });
  hm.innerHTML = html;
  $("#heat-legend").innerHTML = `
    <div><b>期末现货</b></div>
    <div><span class="sw" style="background:rgba(90,169,255,.15)"></span>低水位</div>
    <div><span class="sw" style="background:rgba(90,169,255,.85)"></span>高水位</div>
    <div><span class="sw" style="border:1px solid #ef4444"></span>缺货（&lt;1 件）</div>
    <div style="margin-top:6px">对数刻度。</div>`;

  const weeks = det.inventory_weeks.weeks;
  lineChart($("#chart-inv"), [
    { label: "库存价值(万元)", y: det.inventory_weeks.value.map(v => v / 1e4), color: "#34d399" },
  ], {
    labels: weeks.map(w => "W" + w), dots: false, zeroBased: false,
    axisFmt: v => v.toFixed(0) + "万", unit: " 万元",
    tip: (i) => `<b>第 ${weeks[i]} 周</b><br>${cny(det.inventory_weeks.value[i])}`,
  });

  const pp = b.policy_params;
  if (pp) {
    $("#param-scope").textContent =
      `两级网络 · 风险占比 ${pp.risk_share.toFixed(2)} · 安全系数 ${pp.buffer.toFixed(1)} · 周检`;
    const rows = [];
    pp.sites.forEach((site, si) => {
      pp.parts.forEach((part, pi) => {
        const S = pp.S[si][pi], rop = pp.rop[si][pi], safety = pp.safety[si][pi];
        if (S <= 0) return;
        const hub = sites[si].hub;
        const hi = pp.hubs.indexOf(hub);
        rows.push({ site, part, S, rop, safety,
                    target: pp.targets[part], hub_S: pp.hub_S[hi][pi], hub_rop: pp.hub_rop[hi][pi] });
      });
    });
    rows.sort((a, b) => b.safety - a.safety);
    $("#param-table").innerHTML =
      `<thead><tr><th>机房</th><th>零件</th><th>补货点 rop</th><th>目标水位 S</th><th>安全库存</th>
        <th>缺货率目标</th><th>中心仓 S</th><th>中心仓 rop</th></tr></thead><tbody>` +
      rows.slice(0, 400).map(r => `<tr><td>${r.site}</td><td>${r.part}</td>
        <td>${fmt(r.rop, 1)}</td><td>${fmt(r.S, 1)}</td><td>${fmt(r.safety, 1)}</td>
        <td>${pct(1 - r.target)}</td><td>${fmt(r.hub_S, 1)}</td><td>${fmt(r.hub_rop, 1)}</td></tr>`
      ).join("") + "</tbody>";
  }

  const sw = b.split_sweep || [];
  if (sw.length) {
    $("#split-table").innerHTML =
      `<thead><tr><th>前置仓风险占比</th><th>安全库存系数</th><th>前置仓件数</th><th>中心仓件数</th><th>运营成本</th><th>网络满足率</th><th>达标</th></tr></thead><tbody>` +
      sw.map(r => `<tr class="${r.risk_share === b.chosen_risk_share && r.buffer === b.chosen_buffer ? "best" : ""}">
        <td>${r.risk_share.toFixed(2)}</td><td>${r.buffer.toFixed(1)}</td>
        <td>${fmt(r.site_units)}</td><td>${fmt(r.hub_units)}</td>
        <td>${fmt(r.total_cost)}</td><td>${pct(r.csl_network)}</td>
        <td>${r.meets_target ? "✓" : "–"}</td></tr>`).join("") + "</tbody>";
  }
}

/* ---------------------------------------------------------------- lastbuy */
function renderLastBuy() {
  const b = state.overview;
  const check = b.last_buy_check || [];
  $("#lastbuy-table").innerHTML =
    `<thead><tr><th>零件</th><th>停产年</th><th>尾段年数</th><th>在装量</th><th>到退役总需求(预测)</th>
      <th>P10</th><th>P90</th><th>建议采购</th><th>实际尾段需求</th><th>预测误差</th>
      <th>剩余</th><th>短缺</th><th>剩余金额</th></tr></thead><tbody>` +
    check.map(c => {
      const plan = b.last_buy.find(p => p.part === c.part) || {};
      return `<tr><td>${c.part}</td><td>${plan.eol_year}</td><td>${c.tail_years}</td>
        <td>${fmt(plan.installed)}</td><td>${fmt(plan.expected)}</td>
        <td>${fmt(plan.q10)}</td><td>${fmt(plan.q90)}</td><td><b>${fmt(c.recommended)}</b></td>
        <td>${fmt(c.actual_tail_demand)}</td>
        <td style="color:${Math.abs(c.estimate_error) < 0.2 ? "var(--good)" : "var(--warn)"}">${pct(c.estimate_error, 1)}</td>
        <td>${fmt(c.surplus)}</td><td>${fmt(c.shortfall)}</td><td>${cny(c.surplus_value)}</td></tr>`;
    }).join("") + "</tbody>";
  hBarChart($("#chart-lastbuy"), check.flatMap(c => [
    { k: c.part + " 实际", v: c.actual_tail_demand, color: "#5aa9ff",
      tip: `<b>${c.part}</b><br>实际尾段需求 ${fmt(c.actual_tail_demand)}<br>模型预测 ${fmt(c.estimate)}` },
    { k: c.part + " 采购", v: c.recommended, color: "#34d399",
      tip: `<b>${c.part}</b><br>建议采购 ${fmt(c.recommended)}` },
  ]), { fmt: fmt, labelWidth: 176 });
  const tgt = b.last_buy_targets || [];
  if (tgt.length) {
    $("#lastbuy-targets").innerHTML =
      `<thead><tr><th>采购分位</th><th>总采购件数</th><th>采购金额</th><th>剩余件数</th><th>短缺件数</th><th>剩余金额</th></tr></thead><tbody>` +
      tgt.map(t => `<tr class="${Math.abs(t.target - 0.9) < 1e-9 ? "best" : ""}">
        <td>${pct(t.target, 1)}</td><td>${fmt(t.units)}</td><td>${cny(t.cost)}</td>
        <td>${fmt(t.surplus)}</td><td>${fmt(t.shortfall)}</td><td>${cny(t.surplus_value)}</td></tr>`).join("") +
      "</tbody>";
  }
  hBarChart($("#chart-lastbuy-resid"), check.map(c => ({
    k: c.part, v: c.surplus - c.shortfall,
    color: c.shortfall > 0 ? "#ef4444" : "#f59e0b",
    tip: `<b>${c.part}</b><br>剩余 ${fmt(c.surplus)} · 短缺 ${fmt(c.shortfall)}`,
  })), { fmt: v => (v >= 0 ? "+" : "") + fmt(v), labelWidth: 130 });
}

/* ------------------------------------------------------------------- logs */
async function loadLogs() {
  const policy = $("#sel-policy").value;
  const ev = await getJSON("/api/events?policy=" + policy);
  $("#log-count").textContent = `缺货 ${fmt(ev.n_stockouts)} 条 · 调拨 ${fmt(ev.n_transfers)} 条`;
  const rows = (ev.stockouts || []).slice(0, 500);
  $("#stockout-table").innerHTML =
    `<thead><tr><th>天</th><th>机房</th><th>零件</th><th>服务器</th><th>类型</th></tr></thead><tbody>` +
    (rows.length
      ? rows.map(e => `<tr><td>${fmt(e.day)}</td><td>${e.site}</td><td>${e.part}</td>
          <td>${e.server}</td><td>${e.kind}</td></tr>`).join("")
      : `<tr><td colspan="5" style="text-align:center;color:var(--good)">
           该策略在整个仿真期内没有出现备件缺货（本地即时满足率 ${pct(
             (state.overview.results.find(r => r.kind === policy) || {}).csl_site)}）</td></tr>`)
    + "</tbody>";
  $("#transfer-table").innerHTML =
    `<thead><tr><th>天</th><th>零件</th><th>件数</th><th>类型</th><th>来源</th><th>去向</th><th>运输(天)</th></tr></thead><tbody>` +
    (ev.transfers || []).slice(0, 500).map(e =>
      `<tr><td>${fmt(e.day)}</td><td>${e.part}</td><td>${e.qty}</td><td>${e.kind}</td><td>${e.source}</td><td>${e.target}</td><td>${Number(e.transit).toFixed(2)}</td></tr>`
    ).join("") + "</tbody>";
}

/* ------------------------------------------------------------------- boot */
(async function boot() {
  initTabs();
  try {
    state.overview = await getJSON("/api/overview");
  } catch (e) {
    document.body.innerHTML = `<div style="padding:40px;font-family:sans-serif;color:#eee">
      <h2>结果尚未生成</h2><p>先运行：<code>python3 -m spareparts.acceptance</code></p><pre>${e}</pre></div>`;
    return;
  }
  // header text is shared by all tabs, so set it once on boot
  const m = state.overview.meta;
  $("#subtitle").textContent =
    `${fmt(m.total_units_installed)} 台服务器 · ${m.sites} 机房 · ${m.hubs} 中心仓 · ${m.parts} 种零件 · 数据截止 ${m.as_of_year}-01-01 · 仿真 ${m.sim_years.toFixed(0)} 年`;
  $("#footer-note").textContent =
    `结果指纹 ${m.fingerprint} · 生成于 ${m.generated} · 仿真 ${fmt(m.sim_days)} 天`
    + (state.overview.run_seconds ? ` · 计算 ${state.overview.run_seconds}s` : "");
  const want = new URLSearchParams(location.search).get("tab");
  const btn = want ? $(`#tabs button[data-tab="${want}"]`) : null;
  if (btn) btn.click();
  else renderOverview();

  $("#sel-site").addEventListener("change", e => { state.site = e.target.value; loadForecast(); });
  $("#sel-part").addEventListener("change", e => { state.part = e.target.value; loadForecast(); });
  $("#btn-refresh").addEventListener("click", loadForecast);
  $("#sel-policy").addEventListener("change", loadLogs);
})();

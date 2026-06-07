/* Focus Detector — Web
   Captura webcam, envia frames ao backend (mesmo focus_model.keras),
   desenha o HUD ao vivo e gera o relatório da sessão. */

// ── Paleta (igual ao HUD do desktop) ──────────────────────────────────────
const FOCUSED   = "rgb(110,230,80)";
const UNFOCUSED = "rgb(240,60,60)";
const CYAN      = "rgb(120,210,220)";
const TEXT_DIM  = "rgb(130,128,122)";
const TEXT      = "rgb(240,238,235)";
const PANEL_BG  = "rgba(16,18,22,0.88)";

// ── Estado da sessão ───────────────────────────────────────────────────────
const state = {
  threshold: 0.5,
  inverter: true,
  prob: 0.5,
  confianca: 0.5,
  focado: false,
  rotulo: "DESFOCADO",
  running: false,
  historico: [],   // {t, focado, prob, confianca}
};

const els = {
  startScreen:  document.getElementById("start-screen"),
  liveScreen:   document.getElementById("live-screen"),
  reportScreen: document.getElementById("report-screen"),
  btnStart:     document.getElementById("btn-start"),
  camError:     document.getElementById("cam-error"),
  video:        document.getElementById("video"),
  hud:          document.getElementById("hud"),
  btnInvert:    document.getElementById("btn-invert"),
  btnThrDown:   document.getElementById("btn-thr-down"),
  btnThrUp:     document.getElementById("btn-thr-up"),
  thrLabel:     document.getElementById("thr-label"),
  btnStop:      document.getElementById("btn-stop"),
  btnRestart:   document.getElementById("btn-restart"),
  kpis:         document.getElementById("kpis"),
};

let stream = null;
const grabCanvas = document.createElement("canvas"); // offscreen p/ enviar frames
grabCanvas.width = 320;
grabCanvas.height = 240;
const grabCtx = grabCanvas.getContext("2d");

// ── Inicialização da webcam ──────────────────────────────────────────────
async function startSession() {
  els.camError.classList.add("hidden");
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  } catch (e) {
    els.camError.textContent = "Não foi possível acessar a webcam: " + e.message;
    els.camError.classList.remove("hidden");
    return;
  }
  els.video.srcObject = stream;
  await els.video.play();

  state.running = true;
  state.historico = [];
  els.startScreen.classList.add("hidden");
  els.reportScreen.classList.add("hidden");
  els.liveScreen.classList.remove("hidden");

  resizeHud();
  requestAnimationFrame(renderLoop);  // HUD animado
  inferLoop();                        // predições (sequencial)
}

function stopSession() {
  state.running = false;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  els.liveScreen.classList.add("hidden");
  els.reportScreen.classList.remove("hidden");
  buildReport();
}

// ── Loop de inferência (sequencial, sem afogar o servidor) ─────────────────
async function inferLoop() {
  while (state.running) {
    const t0 = performance.now();
    try {
      grabCtx.drawImage(els.video, 0, 0, grabCanvas.width, grabCanvas.height);
      const dataUrl = grabCanvas.toDataURL("image/jpeg", 0.6);
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
      });
      if (res.ok) {
        const { prob } = await res.json();
        applyPrediction(prob);
      }
    } catch (_) { /* ignora frame ruim */ }

    // ~8 fps no máximo; respeita o tempo da predição
    const elapsed = performance.now() - t0;
    if (elapsed < 120) await sleep(120 - elapsed);
  }
}

function applyPrediction(prob) {
  // Mesma lógica do testar_modelo.py
  state.prob = prob;
  state.focado = (prob >= state.threshold) !== state.inverter; // XOR
  state.confianca = prob >= state.threshold ? prob : 1 - prob;
  state.rotulo = state.focado ? "FOCADO" : "DESFOCADO";
  state.historico.push({ t: Date.now() / 1000, focado: state.focado, prob, confianca: state.confianca });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── Desenho do HUD ──────────────────────────────────────────────────────────
function resizeHud() {
  const r = els.hud.getBoundingClientRect();
  els.hud.width = r.width;
  els.hud.height = r.height;
}
window.addEventListener("resize", () => { if (state.running) resizeHud(); });

function renderLoop() {
  if (!state.running) return;
  drawHud(els.hud.getContext("2d"));
  requestAnimationFrame(renderLoop);
}

function drawHud(ctx) {
  const w = els.hud.width, h = els.hud.height;
  ctx.clearRect(0, 0, w, h);

  const accent = state.focado ? FOCUSED : UNFOCUSED;
  const TOP_H = Math.max(86, h * 0.16);
  const BOT_H = 42;
  const pulse = (Math.sin(performance.now() / 1000 * 3) + 1) / 2;

  // Painel superior
  ctx.fillStyle = PANEL_BG;
  ctx.fillRect(0, 0, w, TOP_H);
  ctx.fillStyle = accent;
  ctx.fillRect(0, TOP_H - 2, w, 2);

  // Orb de status
  drawOrb(ctx, 48, TOP_H / 2, 18, accent, pulse);

  // Rótulo
  ctx.textBaseline = "alphabetic";
  ctx.fillStyle = accent;
  ctx.font = "700 30px 'Segoe UI', sans-serif";
  ctx.fillText(state.rotulo, 82, TOP_H / 2 + 4);
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "12px 'Segoe UI', sans-serif";
  ctx.fillText("ESTADO DE FOCO", 84, TOP_H / 2 + 24);

  // Barra de confiança
  const barX = Math.max(300, w - 320);
  const barW = w - barX - 20;
  const barY = 30;
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "11px 'Segoe UI', sans-serif";
  ctx.fillText("CONFIANÇA", barX, 22);
  const pctLabel = (state.confianca * 100).toFixed(1) + "%";
  ctx.fillStyle = TEXT;
  ctx.textAlign = "right";
  ctx.fillText(pctLabel, barX + barW, 22);
  ctx.textAlign = "left";

  drawBar(ctx, barX, barY, barW, 14, state.confianca, accent);

  // Marcador de threshold
  const thrX = barX + barW * state.threshold;
  ctx.strokeStyle = CYAN;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(thrX, barY - 3); ctx.lineTo(thrX, barY + 17); ctx.stroke();
  ctx.fillStyle = CYAN;
  ctx.font = "10px 'Segoe UI', sans-serif";
  ctx.fillText("thr " + state.threshold.toFixed(2), thrX - 18, barY + 32);

  ctx.fillStyle = TEXT_DIM;
  ctx.font = "11px 'Segoe UI', sans-serif";
  ctx.fillText("prob bruta: " + state.prob.toFixed(3), barX, TOP_H - 12);

  // Brackets HUD
  const M = 26;
  drawBrackets(ctx, M, TOP_H + M, w - M, h - BOT_H - M, accent, 40);

  // Linha de scan
  const scanRange = h - TOP_H - BOT_H - 2 * M;
  if (scanRange > 0) {
    const scanY = TOP_H + M + ((performance.now() / 1000 * 60) % scanRange);
    ctx.strokeStyle = "rgba(120,120,120,0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(M, scanY); ctx.lineTo(w - M, scanY); ctx.stroke();
  }

  // Painel inferior
  ctx.fillStyle = PANEL_BG;
  ctx.fillRect(0, h - BOT_H, w, BOT_H);
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "11px 'Segoe UI', sans-serif";
  ctx.fillText("FOCUS DETECTOR  v1.0 — web", 14, h - BOT_H - 8);
}

function drawOrb(ctx, cx, cy, r, color, pulse) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.4;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, r + 5 + 3 * pulse, 0, Math.PI * 2); ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.beginPath(); ctx.arc(cx - r / 4, cy - r / 4, r / 4, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function drawBar(ctx, x, y, w, h, progress, fg) {
  ctx.fillStyle = "rgb(35,40,45)";
  ctx.fillRect(x, y, w, h);
  const fill = w * Math.max(0, Math.min(1, progress));
  ctx.fillStyle = fg;
  ctx.fillRect(x, y, fill, h);
  ctx.strokeStyle = "rgba(60,55,50,1)";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);
}

function drawBrackets(ctx, x1, y1, x2, y2, color, size) {
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 2;
  const corners = [
    [[x1, y1 + size], [x1, y1], [x1 + size, y1]],
    [[x2 - size, y1], [x2, y1], [x2, y1 + size]],
    [[x1, y2 - size], [x1, y2], [x1 + size, y2]],
    [[x2 - size, y2], [x2, y2], [x2, y2 - size]],
  ];
  for (const pts of corners) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    ctx.lineTo(pts[1][0], pts[1][1]);
    ctx.lineTo(pts[2][0], pts[2][1]);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

// ── Controles ────────────────────────────────────────────────────────────
function updateThrLabel() { els.thrLabel.textContent = "thr " + state.threshold.toFixed(2); }

els.btnStart.addEventListener("click", startSession);
els.btnStop.addEventListener("click", stopSession);
els.btnRestart.addEventListener("click", () => {
  els.reportScreen.classList.add("hidden");
  els.startScreen.classList.remove("hidden");
});
els.btnInvert.addEventListener("click", () => { state.inverter = !state.inverter; });
els.btnThrUp.addEventListener("click", () => {
  state.threshold = Math.min(0.95, Math.round((state.threshold + 0.05) * 100) / 100);
  updateThrLabel();
});
els.btnThrDown.addEventListener("click", () => {
  state.threshold = Math.max(0.05, Math.round((state.threshold - 0.05) * 100) / 100);
  updateThrLabel();
});

// Atalhos de teclado equivalentes ao desktop
document.addEventListener("keydown", (e) => {
  if (!state.running) return;
  const k = e.key.toLowerCase();
  if (k === "q" || e.key === "Escape") stopSession();
  else if (k === "i") state.inverter = !state.inverter;
  else if (e.key === "+" || e.key === "=") els.btnThrUp.click();
  else if (e.key === "-" || e.key === "_") els.btnThrDown.click();
});

// ── Relatório da sessão (replica mostrar_dashboard do desktop) ─────────────
function fmt(seg) {
  const m = Math.floor(seg / 60), s = Math.floor(seg % 60);
  return m ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
}

function computeStats(hist) {
  const t0 = hist[0].t;
  const tempos = hist.map((h) => h.t - t0);
  const focados = hist.map((h) => h.focado);

  let tempoFocado = 0, tempoDesf = 0, transicoes = 0;
  const segmentos = [];
  let segIni = tempos[0], segEstado = focados[0];
  const duracoesFoco = [];
  let durAtual = 0;

  for (let i = 0; i < hist.length - 1; i++) {
    const dt = tempos[i + 1] - tempos[i];
    if (focados[i]) { tempoFocado += dt; durAtual += dt; }
    else { tempoDesf += dt; }

    if (focados[i + 1] !== segEstado) {
      segmentos.push([segIni, tempos[i + 1], segEstado]);
      segIni = tempos[i + 1];
      if (segEstado && !focados[i + 1]) {
        transicoes++;
        if (durAtual > 0) duracoesFoco.push(durAtual);
        durAtual = 0;
      }
      segEstado = focados[i + 1];
    }
  }
  segmentos.push([segIni, tempos[tempos.length - 1], segEstado]);
  if (segEstado && durAtual > 0) duracoesFoco.push(durAtual);

  const tempoTotal = tempoFocado + tempoDesf;
  const pctFoco = tempoTotal > 0 ? (tempoFocado / tempoTotal) * 100 : 0;
  const maiorFoco = duracoesFoco.length ? Math.max(...duracoesFoco) : 0;

  // Produtividade: média móvel de % focado
  const janela = Math.max(5, Math.floor(focados.length / 30));
  const prod = [];
  const buf = [];
  for (const f of focados) {
    buf.push(f ? 1 : 0);
    if (buf.length > janela) buf.shift();
    prod.push((buf.reduce((a, b) => a + b, 0) / buf.length) * 100);
  }

  return { tempos, focados, tempoFocado, tempoDesf, tempoTotal, pctFoco,
           maiorFoco, transicoes, segmentos, prod };
}

function buildReport() {
  if (state.historico.length < 2) {
    els.kpis.innerHTML = "<p style='grid-column:1/-1;color:var(--text-dim)'>Sessão muito curta para gerar relatório.</p>";
    return;
  }
  const s = computeStats(state.historico);

  const kpis = [
    ["TEMPO TOTAL",     fmt(s.tempoTotal),     CYAN],
    ["TEMPO FOCADO",    fmt(s.tempoFocado),    FOCUSED],
    ["TEMPO DESFOCADO", fmt(s.tempoDesf),      UNFOCUSED],
    ["FOCO MÉDIO",      Math.round(s.pctFoco) + "%", FOCUSED],
    ["MAIOR SEQUÊNCIA", fmt(s.maiorFoco),      CYAN],
    ["DISTRAÇÕES",      String(s.transicoes),  UNFOCUSED],
  ];
  els.kpis.innerHTML = kpis.map(([label, val, cor]) =>
    `<div class="kpi"><div class="value" style="color:${cor}">${val}</div>
     <div class="label">${label}</div></div>`).join("");

  drawDonut(s);
  drawProd(s);
  drawTimeline(s);
}

function setupCanvas(canvas, cssHeight) {
  const w = canvas.parentElement.clientWidth - 36;
  canvas.style.width = w + "px";
  canvas.style.height = cssHeight + "px";
  canvas.width = w;
  canvas.height = cssHeight;
  return canvas.getContext("2d");
}

function drawDonut(s) {
  const c = document.getElementById("chart-donut");
  const ctx = setupCanvas(c, 220);
  const cx = c.width / 2, cy = c.height / 2, R = Math.min(cx, cy) - 10, r = R * 0.58;
  const frac = s.tempoTotal > 0 ? s.tempoFocado / s.tempoTotal : 0;

  const ring = (start, end, color) => {
    ctx.beginPath();
    ctx.arc(cx, cy, R, start, end);
    ctx.arc(cx, cy, r, end, start, true);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  };
  const a0 = -Math.PI / 2;
  ring(a0, a0 + frac * 2 * Math.PI, FOCUSED);
  ring(a0 + frac * 2 * Math.PI, a0 + 2 * Math.PI, UNFOCUSED);

  ctx.fillStyle = TEXT;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.font = "700 24px 'Segoe UI'";
  ctx.fillText(Math.round(s.pctFoco) + "%", cx, cy - 6);
  ctx.font = "11px 'Segoe UI'";
  ctx.fillStyle = TEXT_DIM;
  ctx.fillText("FOCO", cx, cy + 16);
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
}

function drawProd(s) {
  const c = document.getElementById("chart-prod");
  const ctx = setupCanvas(c, 220);
  const padL = 36, padB = 24, padT = 12, padR = 12;
  const W = c.width - padL - padR, H = c.height - padT - padB;
  const tMax = s.tempos[s.tempos.length - 1] || 1;
  const xOf = (t) => padL + (t / tMax) * W;
  const yOf = (v) => padT + (1 - v / 100) * H;

  // Eixos / grade
  ctx.strokeStyle = "rgba(80,80,90,0.4)";
  ctx.lineWidth = 1;
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "9px 'Segoe UI'";
  for (let v = 0; v <= 100; v += 25) {
    const y = yOf(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + W, y); ctx.stroke();
    ctx.fillText(v + "%", 6, y + 3);
  }

  // Linha da média móvel
  ctx.strokeStyle = CYAN;
  ctx.lineWidth = 2;
  ctx.beginPath();
  s.prod.forEach((v, i) => {
    const x = xOf(s.tempos[i]), y = yOf(v);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();

  // Preenchimento
  ctx.lineTo(xOf(tMax), yOf(0));
  ctx.lineTo(padL, yOf(0));
  ctx.closePath();
  ctx.fillStyle = "rgba(120,210,220,0.15)";
  ctx.fill();

  // Linha da média
  const ym = yOf(s.pctFoco);
  ctx.strokeStyle = FOCUSED;
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(padL, ym); ctx.lineTo(padL + W, ym); ctx.stroke();
  ctx.setLineDash([]);
}

function drawTimeline(s) {
  const c = document.getElementById("chart-timeline");
  const ctx = setupCanvas(c, 70);
  const padL = 10, padR = 10;
  const W = c.width - padL - padR, H = c.height - 26;
  const tMax = s.tempos[s.tempos.length - 1] || 1;
  const y = 8;
  for (const [ini, fim, estado] of s.segmentos) {
    const x = padL + (ini / tMax) * W;
    const wseg = ((fim - ini) / tMax) * W;
    ctx.fillStyle = estado ? FOCUSED : UNFOCUSED;
    ctx.fillRect(x, y, Math.max(1, wseg), H);
  }
  ctx.fillStyle = TEXT_DIM;
  ctx.font = "9px 'Segoe UI'";
  ctx.fillText("0s", padL, c.height - 6);
  ctx.textAlign = "right";
  ctx.fillText(Math.round(tMax) + "s", padL + W, c.height - 6);
  ctx.textAlign = "left";
}

updateThrLabel();

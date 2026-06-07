"""
Focus Detector — MVP Interface
Detecção de foco em tempo real com HUD moderno.

Teclas:
  q / ESC   -> sair
  i         -> inverter rótulos das classes
  + / -     -> ajustar threshold de decisão
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import numpy as np
import keras
import time

MODELO     = "focus_model.keras"
TAM        = 224

# ── Paleta BGR ────────────────────────────────────────────────────────────────
PANEL_BG      = (22, 18, 16)
FOCUSED_CLR   = (80, 230, 110)     # verde neon
UNFOCUSED_CLR = (60,  60, 240)     # vermelho vivo
CYAN          = (210, 200, 120)
TEXT_DIM      = (130, 125, 120)
TEXT_BRIGHT   = (240, 238, 235)
BAR_BG        = (45,  40,  35)


# ── Helpers de desenho ────────────────────────────────────────────────────────

def blend_rect(img, x1, y1, x2, y2, color, alpha=0.85):
    """Pinta um retângulo semitransparente sobre img (in-place)."""
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    layer = np.full(roi.shape, color, dtype=np.uint8)
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def draw_bar(img, x, y, w, h, progress, fg, bg=BAR_BG):
    """Barra de progresso com glow na ponta."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    fill = int(w * max(0.0, min(1.0, progress)))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), fg, -1)
        gx = x + fill - 3
        if gx + 6 <= x + w:
            roi = img[y:y+h, gx:gx+6]
            white = np.full(roi.shape, (255, 255, 255), dtype=np.uint8)
            cv2.addWeighted(white, 0.45, roi, 0.55, 0, roi)
            img[y:y+h, gx:gx+6] = roi
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 55, 50), 1)


def draw_orb(img, cx, cy, r, color, pulse):
    """Orb pulsante de status."""
    B, G, R = color
    dark   = (B // 4, G // 4, R // 4)
    bright = (min(255, B + 90), min(255, G + 90), min(255, R + 90))
    glow_r = r + 4 + int(3 * pulse)
    glow_c = (B // 3, G // 3, R // 3)
    cv2.circle(img, (cx, cy), glow_r + 5, glow_c,  3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), glow_r,     color,    1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r,          dark,    -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r - 3,      color,   -1, cv2.LINE_AA)
    cv2.circle(img, (cx - r//4, cy - r//4), r // 4, bright, -1, cv2.LINE_AA)


def draw_brackets(img, x1, y1, x2, y2, color, size=36, thick=2):
    """Cantos estilo HUD / targeting."""
    corners = [
        [(x1, y1 + size), (x1, y1), (x1 + size, y1)],
        [(x2 - size, y1), (x2, y1), (x2, y1 + size)],
        [(x1, y2 - size), (x1, y2), (x1 + size, y2)],
        [(x2 - size, y2), (x2, y2), (x2, y2 - size)],
    ]
    for pts in corners:
        cv2.polylines(img, [np.array(pts, np.int32)], False,
                      color, thick, cv2.LINE_AA)


def vignette(img, strength=0.40):
    """Vinheta suave nas bordas."""
    h, w = img.shape[:2]
    Y = np.linspace(-1, 1, h)[:, None]
    X = np.linspace(-1, 1, w)[None, :]
    mask = np.clip(1.0 - np.sqrt(X**2 + Y**2) * strength, 0, 1).astype(np.float32)
    mask3 = np.stack([mask] * 3, axis=-1)
    return (img.astype(np.float32) * mask3).astype(np.uint8)


def mostrar_dashboard(historico):
    """
    Abre uma janela com gráficos da sessão de foco.
    `historico` é uma lista de tuplas (timestamp, focado(bool), prob, confianca).
    Totalmente opcional: qualquer erro aqui é silenciado para não afetar o programa.
    """
    if not historico or len(historico) < 2:
        print("Sessão muito curta para gerar relatório.")
        return

    import matplotlib
    matplotlib.use("TkAgg")  # janela interativa
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Cores (RGB normalizado) coerentes com o HUD
    VERDE    = (110 / 255, 230 / 255,  80 / 255)
    VERMELHO = (240 / 255,  60 / 255,  60 / 255)
    CIANO    = (120 / 255, 210 / 255, 220 / 255)
    FUNDO    = (0.07, 0.07, 0.09)
    PAINEL   = (0.11, 0.11, 0.14)
    TXT      = (0.90, 0.90, 0.90)
    TXT_DIM  = (0.55, 0.55, 0.58)

    t0 = historico[0][0]
    tempos   = [h[0] - t0 for h in historico]      # segundos relativos
    focados  = [h[1] for h in historico]
    confs    = [h[3] for h in historico]

    # ── Integra o tempo gasto em cada estado (dt entre frames) ──────────────────
    tempo_focado = 0.0
    tempo_desf   = 0.0
    transicoes_para_desf = 0      # número de "distrações" (focado -> desfocado)
    segmentos = []                # (inicio, fim, focado) para o gantt
    seg_ini   = tempos[0]
    seg_estado = focados[0]
    duracoes_foco = []            # durações de cada sequência focada
    dur_atual = 0.0

    for i in range(len(historico) - 1):
        dt = tempos[i + 1] - tempos[i]
        if focados[i]:
            tempo_focado += dt
            dur_atual    += dt
        else:
            tempo_desf += dt

        if focados[i + 1] != seg_estado:
            segmentos.append((seg_ini, tempos[i + 1], seg_estado))
            seg_ini = tempos[i + 1]
            if seg_estado and not focados[i + 1]:
                transicoes_para_desf += 1
                if dur_atual > 0:
                    duracoes_foco.append(dur_atual)
                dur_atual = 0.0
            seg_estado = focados[i + 1]
    segmentos.append((seg_ini, tempos[-1], seg_estado))
    if seg_estado and dur_atual > 0:
        duracoes_foco.append(dur_atual)

    tempo_total = tempo_focado + tempo_desf
    pct_foco = (tempo_focado / tempo_total * 100) if tempo_total > 0 else 0.0
    maior_foco = max(duracoes_foco) if duracoes_foco else 0.0
    conf_media = (sum(confs) / len(confs) * 100) if confs else 0.0

    def fmt(seg):
        m, s = divmod(int(seg), 60)
        return f"{m}m {s:02d}s" if m else f"{s}s"

    # ── Produtividade ao longo do tempo (média móvel de % focado) ───────────────
    janela = max(5, len(focados) // 30)
    prod = []
    soma = 0.0
    from collections import deque
    buf = deque(maxlen=janela)
    for f in focados:
        buf.append(1.0 if f else 0.0)
        prod.append(sum(buf) / len(buf) * 100)

    # ── Figura ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 8), facecolor=FUNDO)
    fig.canvas.manager.set_window_title("Focus Detector — Relatório da Sessão")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.7, 1.1, 1.0],
                  hspace=0.45, wspace=0.30,
                  left=0.06, right=0.96, top=0.90, bottom=0.08)

    fig.suptitle("RELATÓRIO DA SESSÃO DE FOCO", color=TXT,
                 fontsize=18, fontweight="bold", y=0.965)

    def estilo_eixo(ax, titulo=""):
        ax.set_facecolor(PAINEL)
        for spine in ax.spines.values():
            spine.set_color((0.3, 0.3, 0.33))
        ax.tick_params(colors=TXT_DIM, labelsize=8)
        if titulo:
            ax.set_title(titulo, color=TXT, fontsize=11, fontweight="bold", pad=10)

    # ── KPIs (linha superior) ─────────────────────────────────────────────────
    kpis = [
        ("TEMPO TOTAL",      fmt(tempo_total),     CIANO),
        ("TEMPO FOCADO",     fmt(tempo_focado),    VERDE),
        ("TEMPO DESFOCADO",  fmt(tempo_desf),      VERMELHO),
        ("FOCO MÉDIO",       f"{pct_foco:.0f}%",   VERDE),
        ("MAIOR SEQUÊNCIA",  fmt(maior_foco),      CIANO),
        ("DISTRAÇÕES",       str(transicoes_para_desf), VERMELHO),
    ]
    # Painel de KPIs desenhado num único eixo
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_facecolor(FUNDO)
    ax_kpi.axis("off")
    n = len(kpis)
    for i, (titulo, valor, cor) in enumerate(kpis):
        cx = (i + 0.5) / n
        ax_kpi.text(cx, 0.78, valor, color=cor, fontsize=20, fontweight="bold",
                    ha="center", va="center", transform=ax_kpi.transAxes)
        ax_kpi.text(cx, 0.28, titulo, color=TXT_DIM, fontsize=9,
                    ha="center", va="center", transform=ax_kpi.transAxes)

    # ── Donut: Focado x Desfocado ──────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[1, 0])
    ax_pie.set_facecolor(FUNDO)
    if tempo_total > 0:
        wedges, _ = ax_pie.pie(
            [tempo_focado, tempo_desf],
            colors=[VERDE, VERMELHO], startangle=90,
            wedgeprops=dict(width=0.42, edgecolor=FUNDO, linewidth=3),
        )
        ax_pie.text(0, 0, f"{pct_foco:.0f}%\nFOCO", color=TXT, fontsize=15,
                    fontweight="bold", ha="center", va="center")
    ax_pie.set_title("FOCADO  x  DESFOCADO", color=TXT, fontsize=11,
                     fontweight="bold", pad=10)
    ax_pie.legend(["Focado", "Desfocado"], loc="lower center",
                  bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False,
                  labelcolor=TXT_DIM, fontsize=9)

    # ── Produtividade ao longo do tempo ────────────────────────────────────────
    ax_prod = fig.add_subplot(gs[1, 1:])
    estilo_eixo(ax_prod, "PRODUTIVIDADE AO LONGO DO TEMPO")
    ax_prod.plot(tempos, prod, color=CIANO, linewidth=2)
    ax_prod.fill_between(tempos, prod, color=CIANO, alpha=0.15)
    ax_prod.axhline(pct_foco, color=VERDE, linestyle="--", linewidth=1,
                    alpha=0.7, label=f"média {pct_foco:.0f}%")
    ax_prod.set_ylim(0, 100)
    ax_prod.set_xlabel("tempo (s)", color=TXT_DIM, fontsize=9)
    ax_prod.set_ylabel("% foco (móvel)", color=TXT_DIM, fontsize=9)
    ax_prod.legend(loc="lower right", frameon=False, labelcolor=TXT_DIM, fontsize=8)
    ax_prod.grid(True, color=(0.2, 0.2, 0.23), linewidth=0.5)

    # ── Linha do tempo (gantt) dos estados ─────────────────────────────────────
    ax_tl = fig.add_subplot(gs[2, :])
    estilo_eixo(ax_tl, "LINHA DO TEMPO DOS ESTADOS")
    for ini, fim, estado in segmentos:
        ax_tl.barh(0, fim - ini, left=ini, height=0.6,
                   color=VERDE if estado else VERMELHO)
    ax_tl.set_yticks([])
    ax_tl.set_xlim(0, tempos[-1])
    ax_tl.set_xlabel("tempo (s)", color=TXT_DIM, fontsize=9)
    ax_tl.set_ylim(-0.5, 0.5)
    from matplotlib.patches import Patch
    ax_tl.legend(handles=[Patch(color=VERDE, label="Focado"),
                          Patch(color=VERMELHO, label="Desfocado")],
                 loc="upper right", frameon=False, labelcolor=TXT_DIM, fontsize=8,
                 ncol=2)

    print("\nRelatório gerado. Feche a janela do gráfico para finalizar.")
    plt.show()


def render_hud(frame, rotulo, confianca, prob, threshold, inverter, pulse):
    h, w = frame.shape[:2]
    accent  = FOCUSED_CLR if rotulo == "FOCADO" else UNFOCUSED_CLR
    dim_acc = tuple(c // 3 for c in accent)

    TOP_H = 96
    BOT_H = 48

    # ── Vinheta ──────────────────────────────────────────────────────────────
    frame[:] = vignette(frame)

    # ── Painel superior ──────────────────────────────────────────────────────
    blend_rect(frame, 0, 0, w, TOP_H, PANEL_BG, alpha=0.88)
    cv2.line(frame, (0, TOP_H), (w, TOP_H), accent, 2)

    # Orb de status
    draw_orb(frame, 52, TOP_H // 2, 20, accent, pulse)

    # Rótulo principal
    cv2.putText(frame, rotulo, (88, 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, accent, 2, cv2.LINE_AA)
    cv2.putText(frame, "ESTADO DE FOCO", (90, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1, cv2.LINE_AA)

    # Barra de confiança (lado direito)
    BAR_X = max(300, w - 310)
    BAR_W = w - BAR_X - 15

    cv2.putText(frame, "CONFIANCA", (BAR_X, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)
    pct_label = f"{confianca * 100:.1f}%"
    (lw, _), _ = cv2.getTextSize(pct_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(frame, pct_label, (BAR_X + BAR_W - lw, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_BRIGHT, 1, cv2.LINE_AA)

    draw_bar(frame, BAR_X, 30, BAR_W, 14, confianca, accent)

    # Marcador de threshold
    thr_x = BAR_X + int(BAR_W * threshold)
    cv2.line(frame, (thr_x, 27), (thr_x, 47), (120, 210, 220), 2)
    cv2.putText(frame, f"thr {threshold:.2f}", (thr_x - 22, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 210, 220), 1, cv2.LINE_AA)

    cv2.putText(frame, f"prob bruta: {prob:.3f}", (BAR_X, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, TEXT_DIM, 1, cv2.LINE_AA)

    # ── Brackets HUD na área de vídeo ─────────────────────────────────────────
    M = 28
    bracket_clr = tuple(c // 2 for c in accent)
    draw_brackets(frame, M, TOP_H + M, w - M, h - BOT_H - M,
                  bracket_clr, size=42, thick=2)

    # Linha de scan animada (sutil)
    scan_y = TOP_H + M + int(((time.time() * 60) % (h - TOP_H - BOT_H - 2 * M)))
    cv2.line(frame, (M, scan_y), (w - M, scan_y), (40, 38, 35), 1)

    # ── Painel inferior ──────────────────────────────────────────────────────
    blend_rect(frame, 0, h - BOT_H, w, h, PANEL_BG, alpha=0.88)
    cv2.line(frame, (0, h - BOT_H), (w, h - BOT_H), (50, 45, 40), 1)

    controls = "[Q] Sair    [I] Inverter classes    [+] Aumentar limiar    [-] Diminuir limiar"
    cv2.putText(frame, controls, (15, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_DIM, 1, cv2.LINE_AA)

    # Branding
    cv2.putText(frame, "FOCUS DETECTOR  v1.0", (15, h - BOT_H - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (50, 46, 44), 1, cv2.LINE_AA)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

print("Carregando modelo...")
model = keras.models.load_model(MODELO)
print("Modelo carregado!\n")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a webcam.")

threshold = 0.5
inverter  = True
historico = []          # registro da sessão: (timestamp, focado, prob, confianca)

cv2.namedWindow("Focus Detector", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Focus Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Focus Detector iniciado. Pressione [Q] para sair.\n")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Falha ao ler frame.")
        break

    pulse = (np.sin(time.time() * 3.0) + 1.0) / 2.0

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (TAM, TAM))
    entrada = np.expand_dims(resized.astype("float32"), axis=0)
    prob    = float(model.predict(entrada, verbose=0)[0][0])

    focado    = (prob >= threshold) ^ inverter
    confianca = prob if prob >= threshold else 1.0 - prob
    rotulo    = "FOCADO" if focado else "DESFOCADO"

    historico.append((time.time(), bool(focado), prob, confianca))

    frame = render_hud(frame, rotulo, confianca, prob, threshold, inverter, pulse)

    cv2.imshow("Focus Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    elif key == ord("i"):
        inverter = not inverter
        print(f"Inversão de classes: {inverter}")
    elif key in (ord("+"), ord("=")):
        threshold = min(0.95, round(threshold + 0.05, 2))
        print(f"Threshold: {threshold:.2f}")
    elif key in (ord("-"), ord("_")):
        threshold = max(0.05, round(threshold - 0.05, 2))
        print(f"Threshold: {threshold:.2f}")

cap.release()
cv2.destroyAllWindows()
print("Encerrado.")

# ── Relatório da sessão (opcional, não interfere no detector) ───────────────────
try:
    mostrar_dashboard(historico)
except Exception as e:
    print(f"Não foi possível gerar o relatório: {e}")

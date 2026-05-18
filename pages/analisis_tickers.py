import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("🔍 Análisis de Tickers")
st.caption(
    "Análisis fundamental, consenso de analistas y posición técnica "
    "para apoyar decisiones de inversión. No es asesoramiento financiero."
)


# ── FUNCIONES ──────────────────────────────────────────────────────────────────

def fmt_large(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e12: return f"{sign}${a/1e12:.2f}T"
    if a >= 1e9:  return f"{sign}${a/1e9:.2f}B"
    if a >= 1e6:  return f"{sign}${a/1e6:.1f}M"
    return f"{sign}${a:,.0f}"

def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.1f}%"

def safe(v, fmt=None, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if fmt:
        return f"{v:{fmt}}{suffix}"
    return str(v) + suffix

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return (100 - (100 / (1 + rs))).round(1)

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    tkr  = yf.Ticker(ticker)
    info = tkr.info or {}
    hist = tkr.history(period="1y")
    try:
        targets = tkr.analyst_price_targets or {}
    except Exception:
        targets = {}
    try:
        recs = tkr.recommendations_summary
        if recs is None:
            recs = pd.DataFrame()
    except Exception:
        recs = pd.DataFrame()
    return info, hist, targets, recs


# ── INPUT ──────────────────────────────────────────────────────────────────────

col_in, _ = st.columns([1, 3])
with col_in:
    ticker_input = st.text_input(
        "Ticker", placeholder="AAPL, NVDA, JNJ, VOO…"
    ).upper().strip()

if not ticker_input:
    st.info("Introduce un ticker para comenzar el análisis.")
    st.stop()

with st.spinner(f"Cargando datos de {ticker_input}…"):
    try:
        info, hist, targets, recs = fetch_data(ticker_input)
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        st.stop()

if not info or hist is None or hist.empty:
    st.error("Ticker no encontrado o sin datos. Verifica el símbolo.")
    st.stop()


# ── CABECERA ───────────────────────────────────────────────────────────────────

name       = info.get("longName") or info.get("shortName") or ticker_input
sector     = info.get("sector", "—")
industry   = info.get("industry", "—")
country    = info.get("country", "—")
quote_type = info.get("quoteType", "EQUITY").upper()
price_cur  = info.get("currentPrice") or info.get("regularMarketPrice") or 0
mkt_cap    = info.get("marketCap")

st.markdown(f"## {name}")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Precio actual", f"${price_cur:.2f}" if price_cur else "—")
h2.metric("Market Cap",    fmt_large(mkt_cap))
h3.metric("Sector",        sector)
h4.metric("País",          country)
st.caption(f"Ticker: **{ticker_input}** · Tipo: {quote_type} · Industria: {industry}")

st.divider()


# ── BLOQUE 1: FUNDAMENTAL SNAPSHOT ────────────────────────────────────────────

st.subheader("📊 Bloque 1 — Fundamental Snapshot")

is_etf = quote_type in ("ETF", "MUTUALFUND")
if is_etf:
    st.info(
        "Este ticker es un ETF/Fondo. Los ratios fundamentales de valoración "
        "no aplican directamente. Se muestran los datos disponibles."
    )

# Variables
pe_ttm   = info.get("trailingPE")
pe_fwd   = info.get("forwardPE")
pb       = info.get("priceToBook")
ev_ebit  = info.get("enterpriseToEbitda")
ps       = info.get("priceToSalesTrailing12Months")
peg      = info.get("trailingPegRatio") or info.get("pegRatio")
roe      = info.get("returnOnEquity")
roa      = info.get("returnOnAssets")
gross_m  = info.get("grossMargins")
oper_m   = info.get("operatingMargins")
net_m    = info.get("profitMargins")
rev_gr   = info.get("revenueGrowth")
earn_gr  = info.get("earningsGrowth")
de_ratio = info.get("debtToEquity")
curr_r   = info.get("currentRatio")
fcf      = info.get("freeCashflow")
div_y    = info.get("dividendYield")

# Corrección div yield: yfinance a veces devuelve 0.69 en lugar de 0.0069
if div_y and not np.isnan(div_y) and div_y > 1:
    div_y_display = f"{div_y:.2f}%"
else:
    div_y_display = fmt_pct(div_y)

# --- Valoración ---
st.markdown("**Valoración**")
v1, v2, v3, v4, v5, v6 = st.columns(6)
v1.metric("P/E TTM",     safe(pe_ttm,  ".1f") if pe_ttm  else "—",
    help="Precio / Beneficio últimos 12m.\nCuánto pagas por $1 de beneficio actual.\n<15 barato · 15–25 razonable · >25 caro")
v2.metric("P/E Forward", safe(pe_fwd,  ".1f") if pe_fwd  else "—",
    help="P/E con beneficios estimados del próximo año.\nSi es menor que TTM → mercado espera crecimiento fuerte.")
v3.metric("P/B",         safe(pb,      ".2f") if pb       else "—",
    help="Precio / Valor contable de activos.\n<1 muy barato · 1–3 razonable · >3 premium (normal en pharma/tech)")
v4.metric("EV/EBITDA",   safe(ev_ebit, ".1f") if ev_ebit else "—",
    help="Valor empresa (incl. deuda) / EBITDA.\nMás robusto que P/E.\n<10 barato · 10–20 razonable · >20 caro")
v5.metric("P/S",         safe(ps,      ".2f") if ps       else "—",
    help="Precio / Ventas. Útil cuando los beneficios son volátiles.\n<2 barato · 2–5 razonable · >5 caro")
v6.metric("PEG",         safe(peg,     ".2f") if peg      else "—",
    help="P/E ajustado por crecimiento esperado.\n<1 barato para su crecimiento · 1–2 razonable · >2 caro")

# --- Rentabilidad ---
st.markdown("**Rentabilidad**")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("ROE",          fmt_pct(roe),
    help="Beneficio / Capital propio. Eficiencia del negocio.\n>15% bueno · >25% excelente.\nMuy alto puede deberse a deuda elevada.")
r2.metric("ROA",          fmt_pct(roa),
    help="Beneficio / Activos totales. Más fiable que ROE con mucha deuda.\n>5% bueno · >10% excelente")
r3.metric("Margen bruto", fmt_pct(gross_m),
    help="(Ventas – Coste directo) / Ventas. Mide el poder de pricing.\n>40% muy bueno · >70% excepcional (pharma/software)")
r4.metric("Margen oper.", fmt_pct(oper_m),
    help="EBIT / Ventas. Eficiencia operativa después de gastos de estructura.\n>15% bueno · >25% excelente")
r5.metric("Margen neto",  fmt_pct(net_m),
    help="Beneficio neto / Ventas. El margen final tras impuestos e intereses.\n>10% bueno · >20% excelente")

# --- Crecimiento y salud ---
st.markdown("**Crecimiento y salud financiera**")
g1, g2, g3, g4, g5, g6 = st.columns(6)
g1.metric("Crec. ingresos",  fmt_pct(rev_gr),
    help="Crecimiento de ventas respecto al año anterior.\n>10% fuerte · >20% excepcional · <0% señal de alerta")
g2.metric("Crec. beneficio", fmt_pct(earn_gr),
    help="Crecimiento del beneficio neto YoY.\nPuede ser volátil por extraordinarios — contrasta con crecimiento de ingresos.")
g3.metric("Deuda/Equity",    f"{de_ratio:.1f}" if de_ratio else "—",
    help="Deuda total / Capital propio.\n<100 conservador · 100–200 moderado · >200 elevado")
g4.metric("Current ratio",   f"{curr_r:.2f}" if curr_r else "—",
    help="Activos corrientes / Pasivos corrientes. Liquidez a corto plazo.\n>1.5 cómodo · 1–1.5 ajustado · <1 alerta")
g5.metric("Free Cash Flow",  fmt_large(fcf),
    help="Caja generada tras inversiones en el negocio. Difícil de manipular.\nPositivo y creciente = señal de calidad muy sólida.")
g6.metric("Div. Yield",      div_y_display,
    help="Dividendo anual / Precio acción.\n>6% puede indicar riesgo de recorte.\n0% normal en empresas de alto crecimiento.")

# --- Señal fundamental ---
f_signals = []
f_notes   = []

if pe_ttm and not np.isnan(pe_ttm):
    if pe_ttm < 15:
        f_signals.append(2); f_notes.append(f"P/E {pe_ttm:.1f} — barato históricamente")
    elif pe_ttm < 28:
        f_signals.append(1); f_notes.append(f"P/E {pe_ttm:.1f} — valoración razonable")
    else:
        f_signals.append(0); f_notes.append(f"P/E {pe_ttm:.1f} — múltiplo elevado")

if roe and not np.isnan(roe):
    if roe > 0.20:
        f_signals.append(2); f_notes.append(f"ROE {roe*100:.0f}% — alta rentabilidad")
    elif roe > 0.10:
        f_signals.append(1); f_notes.append(f"ROE {roe*100:.0f}% — rentabilidad aceptable")
    else:
        f_signals.append(0); f_notes.append(f"ROE {roe*100:.0f}% — baja rentabilidad")

if net_m and not np.isnan(net_m):
    if net_m > 0.15:
        f_signals.append(2); f_notes.append(f"Margen neto {net_m*100:.0f}% — muy sano")
    elif net_m > 0.05:
        f_signals.append(1); f_notes.append(f"Margen neto {net_m*100:.0f}% — aceptable")
    else:
        f_signals.append(0); f_notes.append(f"Margen neto {net_m*100:.1f}% — ajustado")

if rev_gr and not np.isnan(rev_gr):
    if rev_gr > 0.15:
        f_signals.append(2); f_notes.append(f"Crecimiento ingresos {rev_gr*100:.0f}% — fuerte")
    elif rev_gr > 0.03:
        f_signals.append(1); f_notes.append(f"Crecimiento ingresos {rev_gr*100:.0f}% — moderado")
    else:
        f_signals.append(0); f_notes.append(f"Crecimiento ingresos {rev_gr*100:.0f}% — débil")

fund_score = round(np.mean(f_signals), 2) if f_signals else None

if not is_etf and fund_score is not None:
    with st.expander("📌 Interpretación fundamental", expanded=False):
        for note in f_notes:
            st.caption(f"• {note}")

st.divider()


# ── BLOQUE 2: CONSENSO DE ANALISTAS ───────────────────────────────────────────

st.subheader("🎯 Bloque 2 — Consenso de analistas")

if is_etf:
    st.info("Los ETFs no tienen cobertura de analistas individuales.")
    analyst_score = 1
else:
    target_mean = info.get("targetMeanPrice") or (targets.get("mean") if targets else None)
    target_low  = info.get("targetLowPrice")  or (targets.get("low")  if targets else None)
    target_high = info.get("targetHighPrice") or (targets.get("high") if targets else None)
    n_analysts  = info.get("numberOfAnalystOpinions") or (targets.get("numberOfAnalysts") if targets else None)
    rec_key     = info.get("recommendationKey", "—")
    rec_mean    = info.get("recommendationMean")

    upside = ((target_mean / price_cur) - 1) * 100 if (target_mean and price_cur) else None

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Precio objetivo (media)", f"${target_mean:.2f}" if target_mean else "—",
              f"{upside:+.1f}% upside" if upside is not None else None)
    a2.metric("Rango analistas",
              f"${target_low:.0f} – ${target_high:.0f}" if (target_low and target_high) else "—")
    a3.metric("Nº analistas", str(n_analysts) if n_analysts else "—")
    a4.metric("Recomendación media",
              rec_key.replace("_", " ").title() if rec_key != "—" else "—",
              f"{rec_mean:.1f} / 5.0" if rec_mean else None)

    if target_mean and price_cur and target_low and target_high:
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = price_cur,
            delta = {"reference": target_mean, "valueformat": ".2f"},
            gauge = {
                "axis":  {"range": [target_low * 0.85, target_high * 1.05]},
                "bar":   {"color": "#FF6B35"},
                "steps": [
                    {"range": [target_low * 0.85, target_low],    "color": "#A32D2D"},
                    {"range": [target_low,          target_mean],  "color": "#FFA726"},
                    {"range": [target_mean,          target_high], "color": "#3B6D11"},
                    {"range": [target_high, target_high * 1.05],   "color": "#43A047"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": target_mean,
                },
            },
            number = {"prefix": "$"},
            title  = {"text": "Precio actual vs rango de analistas"},
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=50, b=20, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

    if not recs.empty:
        latest = recs.iloc[0]
        try:
            strong_buy  = int(latest.get("strongBuy",  0))
            buy         = int(latest.get("buy",        0))
            hold        = int(latest.get("hold",       0))
            sell        = int(latest.get("sell",       0))
            strong_sell = int(latest.get("strongSell", 0))

            fig_recs = go.Figure(go.Bar(
                x            = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
                y            = [strong_buy, buy, hold, sell, strong_sell],
                marker_color = ["#1B5E20","#388E3C","#FFA726","#E64A19","#B71C1C"],
                text         = [strong_buy, buy, hold, sell, strong_sell],
                textposition = "outside",
            ))
            fig_recs.update_layout(
                title      = "Distribución de recomendaciones",
                height     = 280,
                yaxis      = dict(title="Nº analistas"),
                margin     = dict(t=40, b=20, l=20, r=20),
                showlegend = False,
            )
            st.plotly_chart(fig_recs, use_container_width=True)
        except Exception:
            pass

    if upside is not None:
        if upside > 20:   analyst_score = 2
        elif upside > 5:  analyst_score = 1
        else:             analyst_score = 0
    elif rec_mean is not None:
        if rec_mean < 2:    analyst_score = 2
        elif rec_mean < 3.5:analyst_score = 1
        else:               analyst_score = 0
    else:
        analyst_score = 1

st.divider()


# ── BLOQUE 3: POSICIÓN TÉCNICA ─────────────────────────────────────────────────

st.subheader("📈 Bloque 3 — Posición técnica")

close     = hist["Close"]
ma50      = close.rolling(50).mean()
ma200     = close.rolling(200).mean()
rsi       = compute_rsi(close)

high52    = close.max()
low52     = close.min()
price_now = float(close.iloc[-1])
pos_52w   = (price_now - low52) / (high52 - low52) * 100 if high52 != low52 else 50
rsi_now   = float(rsi.iloc[-1])   if not rsi.empty   else 50
ma50_now  = float(ma50.iloc[-1])  if not ma50.empty  else None
ma200_now = float(ma200.iloc[-1]) if not ma200.empty else None

t1, t2, t3, t4 = st.columns(4)
t1.metric("Posición 52 semanas", f"{pos_52w:.0f}%",
          help=f"Dónde está el precio dentro del rango anual.\n0% = mínimo · 100% = máximo\nMín: ${low52:.2f}  ·  Máx: ${high52:.2f}")
t2.metric("RSI (14 días)", f"{rsi_now:.1f}",
          "Sobrecomprado" if rsi_now > 70 else ("Sobrevendido" if rsi_now < 30 else "Zona neutral"),
          help="Índice de Fuerza Relativa. Mide si el activo está sobrecomprado o sobrevendido.\n<30 sobrevendido (posible oportunidad) · 30–70 neutral · >70 sobrecomprado (precaución)")
t3.metric("Media móvil 50d",  f"${ma50_now:.2f}"  if ma50_now  else "—",
          f"{((price_now/ma50_now)-1)*100:+.1f}% vs precio"  if ma50_now  else None,
          help="Tendencia de corto/medio plazo.\nPrecio por encima → tendencia alcista de corto plazo.")
t4.metric("Media móvil 200d", f"${ma200_now:.2f}" if ma200_now else "—",
          f"{((price_now/ma200_now)-1)*100:+.1f}% vs precio" if ma200_now else None,
          help="Tendencia de largo plazo. La más importante.\nPrecio por encima → tendencia alcista. MA50>MA200 = 'Golden Cross' (señal positiva).")

fig_tech = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.70, 0.30],
    vertical_spacing=0.04,
)
fig_tech.add_trace(go.Scatter(
    x=close.index, y=close.values, name="Precio",
    line=dict(color="#FF6B35", width=1.8)
), row=1, col=1)
fig_tech.add_trace(go.Scatter(
    x=ma50.index, y=ma50.values, name="MA 50d",
    line=dict(color="#26C6DA", width=1.2, dash="dot")
), row=1, col=1)
fig_tech.add_trace(go.Scatter(
    x=ma200.index, y=ma200.values, name="MA 200d",
    line=dict(color="#FFA726", width=1.2, dash="dot")
), row=1, col=1)
fig_tech.add_trace(go.Scatter(
    x=rsi.index, y=rsi.values, name="RSI",
    line=dict(color="#534AB7", width=1.5),
    fill="tonexty", fillcolor="rgba(83,74,183,0.08)"
), row=2, col=1)
fig_tech.add_hline(y=70, line_dash="dash", line_color="red",   opacity=0.4, row=2, col=1)
fig_tech.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.4, row=2, col=1)
fig_tech.add_hline(y=50, line_dash="dot",  line_color="gray",  opacity=0.3, row=2, col=1)
fig_tech.update_layout(
    height     = 500,
    showlegend = True,
    legend     = dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    hovermode  = "x unified",
    margin     = dict(l=40, r=20, t=40, b=20),
)
fig_tech.update_yaxes(title_text="Precio ($)", row=1, col=1)
fig_tech.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
st.plotly_chart(fig_tech, use_container_width=True)

tech_signals = []
if rsi_now < 30:       tech_signals.append(2)
elif rsi_now < 45:     tech_signals.append(2)
elif rsi_now < 60:     tech_signals.append(1)
elif rsi_now < 70:     tech_signals.append(1)
else:                  tech_signals.append(0)

if ma200_now:
    tech_signals.append(2 if price_now > ma200_now else 0)
if ma50_now and ma200_now:
    tech_signals.append(2 if ma50_now > ma200_now else 0)

tech_score = round(np.mean(tech_signals), 2) if tech_signals else 1

st.divider()


# ── BLOQUE 4: VEREDICTO DE INVERSIÓN ──────────────────────────────────────────

st.subheader("🚦 Bloque 4 — Veredicto de inversión")
st.caption(
    "Semáforo agregado basado en los tres bloques anteriores. "
    "Indicativo — no es recomendación de compra ni venta."
)

scores = {}
if not is_etf and fund_score is not None:
    scores["Fundamentales"] = fund_score
scores["Analistas"] = analyst_score
scores["Técnico"]   = tech_score

avg_score = np.mean(list(scores.values()))

if avg_score >= 1.5:
    verdict_color = "#1B5E20"
    verdict_bg    = "#c6efce"
    verdict_emoji = "🟢"
    verdict_text  = "Señal positiva"
    verdict_desc  = "La mayoría de indicadores apuntan favorablemente. Analiza con más detalle antes de decidir."
elif avg_score >= 0.8:
    verdict_color = "#7f5700"
    verdict_bg    = "#fff2cc"
    verdict_emoji = "🟡"
    verdict_text  = "Señal mixta"
    verdict_desc  = "Algunos indicadores positivos, otros negativos. Requiere análisis adicional."
else:
    verdict_color = "#7f1d1d"
    verdict_bg    = "#ffc7ce"
    verdict_emoji = "🔴"
    verdict_text  = "Señal negativa"
    verdict_desc  = "La mayoría de indicadores apuntan desfavorablemente en el momento actual."

st.markdown(
    f"""
    <div style="padding:1.5rem; border-radius:12px; background:{verdict_bg};
                border-left:6px solid {verdict_color}; margin-bottom:1rem">
        <div style="font-size:2rem">{verdict_emoji} {verdict_text}</div>
        <div style="color:{verdict_color}; margin-top:0.5rem">{verdict_desc}</div>
    </div>
    """,
    unsafe_allow_html=True
)

COLORS = {0: "#A32D2D", 1: "#BA7517", 2: "#3B6D11"}
LABELS = {0: "🔴 Negativo", 1: "🟡 Neutral", 2: "🟢 Positivo"}

v1, v2, v3 = st.columns(3)
for col, (bloque, score) in zip([v1, v2, v3], scores.items()):
    col.metric(bloque, LABELS[round(score)])

st.markdown("**Puntuaciones por bloque** (0 = negativo · 1 = neutral · 2 = positivo)")
fig_scores = go.Figure(go.Bar(
    x            = list(scores.keys()),
    y            = list(scores.values()),
    marker_color = [COLORS[round(v)] for v in scores.values()],
    text         = [f"{v:.1f}" for v in scores.values()],
    textposition = "outside",
))
fig_scores.add_hline(y=1, line_dash="dot", line_color="gray", opacity=0.5)
fig_scores.update_layout(
    height     = 260,
    yaxis      = dict(range=[0, 2.4], title="Señal"),
    showlegend = False,
    margin     = dict(t=20, b=20, l=20, r=20),
)
st.plotly_chart(fig_scores, use_container_width=True)
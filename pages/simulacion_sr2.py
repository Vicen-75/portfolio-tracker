import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize


# ── FUNCIONES ──────────────────────────────────────────────────────────────────

def load_positions():
    try:
        return pd.read_csv("data/positions.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["ticker","shares","avg_price","date_added","category"])

@st.cache_data(ttl=300)
def get_prices_today(tickers):
    try:
        raw = yf.download(list(tickers), period="2d", auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            return {tickers[0]: float(raw.dropna().iloc[-1])}
        return {t: float(raw[t].dropna().iloc[-1]) for t in tickers if t in raw.columns}
    except:
        return {}

@st.cache_data(ttl=300)
def load_price_history(tickers, period):
    try:
        raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=tickers[0])
        return raw.dropna()
    except:
        return pd.DataFrame()


# ── CARGA DE POSICIONES ────────────────────────────────────────────────────────

positions = load_positions()
if positions.empty:
    st.title("🔬 Análisis")
    st.warning("No hay posiciones. Ve al Dashboard y añade posiciones primero.")
    st.stop()

tickers_tuple = tuple(positions["ticker"].tolist())
prices_today  = get_prices_today(tickers_tuple)

positions["current_price"] = positions["ticker"].map(prices_today)
positions["market_value"]  = positions["shares"] * positions["current_price"]
total_mv = positions["market_value"].sum()
positions["actual_weight"] = positions["market_value"] / total_mv

ticker_list    = positions["ticker"].tolist()
actual_weights = positions["actual_weight"].to_numpy()


# ── SESSION STATE ──────────────────────────────────────────────────────────────

for t, w in zip(ticker_list, actual_weights):
    if f"s_{t}" not in st.session_state:
        st.session_state[f"s_{t}"]    = round(float(w) * 100, 1)
    if f"lock_{t}" not in st.session_state:
        st.session_state[f"lock_{t}"] = True


# ── REDISTRIBUCIÓN ─────────────────────────────────────────────────────────────

def redistribute(changed_t):
    new_val    = st.session_state[f"s_{changed_t}"] / 100.0
    locked_sum = sum(
        st.session_state[f"s_{x}"] / 100.0
        for x in ticker_list
        if st.session_state.get(f"lock_{x}", False) and x != changed_t
    )
    available_w = max(0.0, 1.0 - locked_sum)
    new_val     = min(new_val, available_w)
    others      = [x for x in ticker_list
                   if not st.session_state.get(f"lock_{x}", False) and x != changed_t]
    others_sum  = sum(st.session_state[f"s_{x}"] / 100.0 for x in others)
    remaining   = available_w - new_val
    st.session_state[f"s_{changed_t}"] = round(new_val * 100, 1)
    if others:
        if others_sum > 0.001:
            for x in others:
                old = st.session_state[f"s_{x}"] / 100.0
                st.session_state[f"s_{x}"] = round(old / others_sum * remaining * 100, 1)
        else:
            for x in others:
                st.session_state[f"s_{x}"] = round(remaining / len(others) * 100, 1)

def make_cb(t):
    def cb(): redistribute(t)
    return cb


# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuración")
    period       = st.selectbox("Período histórico", ["3mo","6mo","1y","2y"], index=2)
    n_portfolios = st.slider("Portfolios simulados", 500, 5000, 2000, step=500)

    st.divider()
    if st.button("🔄 Actualizar datos de mercado"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Fuerza una nueva descarga desde Yahoo Finance.")

    st.divider()
    st.subheader("🎛️ Pesos hipotéticos")
    total_pct = sum(st.session_state[f"s_{t}"] for t in ticker_list)
    st.caption(f"Total: **{total_pct:.1f}%** {'✅' if abs(total_pct - 100) < 0.6 else '⚠️'}")

    if st.button("↺ Resetear a portfolio real"):
        for t, w in zip(ticker_list, actual_weights):
            st.session_state[f"s_{t}"]    = round(float(w) * 100, 1)
            st.session_state[f"lock_{t}"] = True
        st.rerun()

    all_locked = all(st.session_state.get(f"lock_{t}", True) for t in ticker_list)
    if st.button("🔓 Desbloquear todos" if all_locked else "🔒 Bloquear todos"):
        for t in ticker_list:
            st.session_state[f"lock_{t}"] = not all_locked
        st.rerun()

    st.caption("🔒 Bloquea posiciones que no quieres tocar.")
    st.markdown("---")

    for _, row in positions.sort_values("actual_weight", ascending=False).iterrows():
        t         = row["ticker"]
        is_locked = st.session_state.get(f"lock_{t}", True)
        c1, c2    = st.columns([1, 5])
        with c1:
            st.checkbox("lock", value=is_locked, key=f"lock_{t}",
                        label_visibility="collapsed",
                        help=f"{'Desbloquear' if is_locked else 'Bloquear'} {t}")
        with c2:
            st.slider(f"{'🔒 ' if is_locked else ''}{t}",
                      min_value=0.0, max_value=100.0, step=0.5,
                      key=f"s_{t}", disabled=is_locked,
                      on_change=make_cb(t),
                      help=f"Peso real actual: {row['actual_weight']:.1%}")


# ── PESOS HIPOTÉTICOS ──────────────────────────────────────────────────────────

hypo_raw     = np.array([st.session_state[f"s_{t}"] for t in ticker_list])
hypo_weights = hypo_raw / hypo_raw.sum()


# ── DATOS HISTÓRICOS ───────────────────────────────────────────────────────────

raw          = load_price_history(tickers_tuple, period)
log_returns  = np.log(raw / raw.shift(1)).dropna()
mean_returns = log_returns.mean() * 252
cov_matrix   = log_returns.cov()   * 252

# CORRECCIÓN: yfinance devuelve columnas en orden alfabético, no en orden del CSV.
# Siempre reordenamos por ticker_list para garantizar alineación correcta.
available = [t for t in ticker_list if t in mean_returns.index]
mr = mean_returns[available]               # Serie reordenada por ticker_list
cm = cov_matrix.loc[available, available]  # Matriz reordenada por ticker_list

# Pesos reales e hipotéticos en el mismo orden que available
aw = np.array([actual_weights[ticker_list.index(t)] for t in available])
aw = aw / aw.sum()
hw = np.array([hypo_weights[ticker_list.index(t)] for t in available])
hw = hw / hw.sum()


# ── ESTADÍSTICAS DE PORTFOLIOS ─────────────────────────────────────────────────
# mr y cm.values garantizan alineación ticker a ticker

act_r = float(np.dot(aw, mr))
act_v = float(np.sqrt(aw @ cm.values @ aw))
act_s = act_r / act_v

hyp_r = float(np.dot(hw, mr))
hyp_v = float(np.sqrt(hw @ cm.values @ hw))
hyp_s = hyp_r / hyp_v


# ── MONTE CARLO ────────────────────────────────────────────────────────────────

all_w_mc, port_r, port_v, port_s = [], [], [], []
for _ in range(n_portfolios):
    w = np.random.dirichlet(np.ones(len(available)))
    r = float(np.dot(w, mr))
    v = float(np.sqrt(w @ cm.values @ w))
    all_w_mc.append(w); port_r.append(r); port_v.append(v); port_s.append(r / v)

port_r     = np.array(port_r)
port_v     = np.array(port_v)
port_s     = np.array(port_s)
minvol_idx = int(np.argmin(port_v))


# ── SCIPY: PESOS ÓPTIMOS EXACTOS ──────────────────────────────────────────────

def neg_sharpe_opt(w):
    r = float(np.dot(w, mr))
    v = float(np.sqrt(w @ cm.values @ w))
    return -(r / v)

n_ast   = len(available)
opt_res = minimize(
    neg_sharpe_opt,
    x0          = np.ones(n_ast) / n_ast,
    method      = "SLSQP",
    bounds      = [(0, 1)] * n_ast,
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
)
best_w_scipy = opt_res.x if opt_res.success else np.ones(n_ast) / n_ast
opt_r_scipy  = float(np.dot(best_w_scipy, mr))
opt_v_scipy  = float(np.sqrt(best_w_scipy @ cm.values @ best_w_scipy))
opt_s_scipy  = opt_r_scipy / opt_v_scipy


# ── CABECERA ───────────────────────────────────────────────────────────────────

st.title("🔬 Análisis del portfolio")
st.caption("Explora cómo cambia el riesgo y retorno al ajustar los pesos con los sliders.")


# ── RETORNOS ACUMULADOS ────────────────────────────────────────────────────────

st.subheader("Retornos acumulados")
cum    = (raw / raw.iloc[0] - 1) * 100
melted = cum.reset_index().melt("Date", var_name="Ticker", value_name="Retorno (%)")

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=melted, x="Date", y="Retorno (%)", hue="Ticker", ax=ax1)
ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax1.set_title(f"Retornos acumulados — {period}")
sns.despine()
plt.tight_layout()
st.pyplot(fig1)

summary = cum.iloc[-1].reset_index()
summary.columns = ["Ticker","Retorno (%)"]
summary["Retorno (%)"] = summary["Retorno (%)"].round(2)
st.dataframe(summary.sort_values("Retorno (%)", ascending=False),
             use_container_width=True, hide_index=True)

st.divider()


# ── FRONTERA EFICIENTE ─────────────────────────────────────────────────────────

st.subheader("Frontera eficiente — Simulación Monte Carlo")
st.caption(
    "**◆ azul** = portfolio real.  **⭐ rojo** = hipotético (sliders).  "
    "**★ dorado** = Max Sharpe exacto (scipy)."
)

fig2, ax2 = plt.subplots(figsize=(10, 5))
sc = ax2.scatter(port_v, port_r, c=port_s, cmap="viridis", alpha=0.4, s=10)
plt.colorbar(sc, ax=ax2, label="Sharpe Ratio")

ax2.scatter(opt_v_scipy, opt_r_scipy,
            color="gold", s=180, zorder=6,
            label=f"Max Sharpe — scipy (Sharpe {opt_s_scipy:.2f})")
ax2.scatter(port_v[minvol_idx], port_r[minvol_idx],
            color="cyan", s=150, zorder=5, label="Min Volatilidad")
ax2.scatter(hyp_v, hyp_r,
            color="red", s=200, marker="*", zorder=6,
            label=f"Hipotético (Sharpe {hyp_s:.2f})")
ax2.scatter(act_v, act_r,
            color="#00BFFF", s=200, marker="D", zorder=6,
            label=f"Real actual (Sharpe {act_s:.2f})")

ax2.set_xlabel("Volatilidad anualizada (Riesgo)")
ax2.set_ylabel("Retorno anualizado")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax2.set_title("Frontera eficiente")
ax2.legend(fontsize=8)
sns.despine()
plt.tight_layout()
st.pyplot(fig2)

st.divider()


# ── COMPARATIVA ────────────────────────────────────────────────────────────────

st.subheader("Comparativa de portfolios")
col_real, col_hypo = st.columns(2)

with col_real:
    st.markdown("**◆ Portfolio real (pesos actuales)**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Retorno esperado", f"{act_r:.1%}")
    m2.metric("Volatilidad",      f"{act_v:.1%}")
    m3.metric("Sharpe Ratio",     f"{act_s:.2f}")

with col_hypo:
    st.markdown("**⭐ Portfolio hipotético (sliders)**")
    h1, h2, h3 = st.columns(3)
    h1.metric("Retorno esperado", f"{hyp_r:.1%}", f"{hyp_r - act_r:+.1%}")
    h2.metric("Volatilidad",      f"{hyp_v:.1%}", f"{hyp_v - act_v:+.1%}",
              delta_color="inverse")
    h3.metric("Sharpe Ratio",     f"{hyp_s:.2f}", f"{hyp_s - act_s:+.2f}")

st.divider()


# ── PESOS ÓPTIMOS ──────────────────────────────────────────────────────────────

st.subheader("Distribución óptima según el modelo (Max Sharpe — scipy)")
st.caption(
    "Resultado exacto y reproducible. Basado en retornos históricos del período seleccionado. "
    "Referencia orientativa — los retornos pasados no garantizan resultados futuros."
)

aw_map = {t: actual_weights[ticker_list.index(t)] for t in available}
hw_map = {t: hypo_weights[ticker_list.index(t)]   for t in available}

opt_df = pd.DataFrame({
    "Ticker":              available,
    "Peso actual (%)":     [round(aw_map[t] * 100, 1) for t in available],
    "Peso hipotético (%)": [round(hw_map[t] * 100, 1) for t in available],
    "Peso óptimo (%)":     (best_w_scipy * 100).round(1),
    "vs Actual (pp)":      [round(w * 100 - aw_map[t] * 100, 1)
                            for t, w in zip(available, best_w_scipy)],
}).sort_values("Peso óptimo (%)", ascending=False)

st.dataframe(opt_df, use_container_width=True, hide_index=True)
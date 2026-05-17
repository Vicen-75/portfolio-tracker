import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.set_page_config(page_title="Análisis con simulación", page_icon="🧪", layout="wide")


# ── FUNCIONES ──────────────────────────────────────────────────────────────────

def load_positions():
    try:
        return pd.read_csv("data/positions.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["ticker","shares","avg_price","date_added","category"])

@st.cache_data(ttl=300)
def get_prices(tickers):
    try:
        raw = yf.download(list(tickers), period="2d", auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            return {tickers[0]: float(raw.dropna().iloc[-1])}
        return {t: float(raw[t].dropna().iloc[-1]) for t in tickers if t in raw.columns}
    except:
        return {}

@st.cache_data(ttl=3600)
def get_price_history(tickers, period):
    try:
        raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=tickers[0])
        return raw.dropna()
    except:
        return pd.DataFrame()

def calc_portfolio(df, prices):
    r = df.copy()
    r["current_price"] = r["ticker"].map(prices)
    r["market_value"]  = r["shares"] * r["current_price"]
    r["cost_basis"]    = r["shares"] * r["avg_price"]
    r["pl_usd"]        = r["market_value"] - r["cost_basis"]
    r["pl_pct"]        = r["pl_usd"] / r["cost_basis"] * 100
    total = r["market_value"].sum()
    r["weight"] = r["market_value"] / total * 100
    return r

def apply_trades(df, prices, trades):
    sim  = df.copy()
    cash = 0.0
    for t in trades:
        ticker = t["ticker"]
        price  = prices.get(ticker)
        if not price:
            continue
        shares = t["amount"] / price if t["mode"] == "dollars" else t["amount"]
        if t["action"] == "buy":
            cash -= shares * price
            if ticker in sim["ticker"].values:
                idx     = sim[sim["ticker"] == ticker].index[0]
                old_sh  = sim.loc[idx, "shares"]
                old_avg = sim.loc[idx, "avg_price"]
                new_sh  = old_sh + shares
                sim.loc[idx, "shares"]    = new_sh
                sim.loc[idx, "avg_price"] = (old_sh * old_avg + shares * price) / new_sh
            else:
                sim = pd.concat([sim, pd.DataFrame([{
                    "ticker": ticker, "shares": shares, "avg_price": price,
                    "date_added": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "category": t.get("category", "Otro"),
                }])], ignore_index=True)
        elif t["action"] == "sell":
            if ticker in sim["ticker"].values:
                idx       = sim[sim["ticker"] == ticker].index[0]
                old_sh    = sim.loc[idx, "shares"]
                sell      = min(shares, old_sh)
                cash     += sell * price
                remaining = old_sh - sell
                if remaining < 0.001:
                    sim = sim[sim["ticker"] != ticker].reset_index(drop=True)
                else:
                    sim.loc[idx, "shares"] = remaining
    return sim, cash

def calc_sharpe(tickers, weights_dict, mean_ret, cov):
    avail = [t for t in tickers if t in mean_ret.index and weights_dict.get(t, 0) > 0]
    if not avail:
        return None, None, None
    w = np.array([weights_dict[t] for t in avail])
    w = w / w.sum()
    r = float(np.dot(w, mean_ret[avail]))
    v = float(np.sqrt(w @ cov.loc[avail, avail].values @ w))
    return (round(r / v, 2) if v > 0 else None), round(r, 4), round(v, 4)


# ── CARGA INICIAL ──────────────────────────────────────────────────────────────

positions = load_positions()
if positions.empty:
    st.title("🧪 Análisis con simulación")
    st.warning("No hay posiciones. Ve al Dashboard y añade posiciones primero.")
    st.stop()

if "trades_sim" not in st.session_state:
    st.session_state.trades_sim = []

existing_tickers = sorted(positions["ticker"].tolist())


# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuración")
    period = st.selectbox("Período histórico", ["3mo","6mo","1y","2y"], index=2)
    n_sim  = st.slider("Portfolios simulados", 500, 5000, 2000, step=500)

    st.divider()
    if st.button("🔄 Actualizar datos de mercado"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Fuerza una nueva descarga desde Yahoo Finance.")

    st.divider()
    st.subheader("➕ Añadir operación")

    with st.form("form_sim", clear_on_submit=True):
        action   = st.selectbox("Acción", ["Comprar","Vender"])
        t_select = st.selectbox("Ticker en cartera", ["— nuevo —"] + existing_tickers)
        t_new    = st.text_input("O nuevo ticker", placeholder="AAPL").upper().strip()
        mode     = st.selectbox("Cantidad en", ["Dólares ($)","Acciones"])
        amount   = st.number_input("Cantidad", min_value=0.01, step=1.0)
        category = st.selectbox("Categoría (solo nuevos tickers)",
                       ["ETF","Tech","Salud","Financiero","Consumo","Materiales","Industrial","Otro"])

        if st.form_submit_button("➕ Añadir"):
            ticker = t_new if t_new else (t_select if t_select != "— nuevo —" else None)
            if ticker and amount > 0:
                label = (
                    f"{'🟢' if action=='Comprar' else '🔴'} {action} {ticker} — "
                    f"{'$'+str(int(amount)) if 'Dólares' in mode else str(amount)+' acc.'}"
                )
                st.session_state.trades_sim.append({
                    "action":   "buy" if action == "Comprar" else "sell",
                    "ticker":   ticker,
                    "mode":     "dollars" if "Dólares" in mode else "shares",
                    "amount":   amount,
                    "category": category,
                    "label":    label,
                })
                st.success(f"Añadido: {label}")
            else:
                st.error("Elige un ticker y una cantidad > 0.")

    if st.session_state.trades_sim:
        st.divider()
        st.subheader(f"Cola — {len(st.session_state.trades_sim)} op.")
        for i, t in enumerate(st.session_state.trades_sim):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{i+1}.** {t['label']}")
            if c2.button("✕", key=f"rm_sim_{i}"):
                st.session_state.trades_sim.pop(i)
                st.rerun()
        if st.button("🗑️ Limpiar todo"):
            st.session_state.trades_sim = []
            st.rerun()


# ── PREPARACIÓN DE DATOS ───────────────────────────────────────────────────────

has_trades  = len(st.session_state.trades_sim) > 0
all_tickers = tuple(set(
    positions["ticker"].tolist() +
    [t["ticker"] for t in st.session_state.trades_sim]
))

prices  = get_prices(all_tickers)
current = calc_portfolio(positions, prices)

if has_trades:
    sim_df, cash_net = apply_trades(positions, prices, st.session_state.trades_sim)
    simulated        = calc_portfolio(sim_df, prices)

# Una sola descarga histórica para toda la página
history = get_price_history(all_tickers, period)
if history.empty:
    st.error("No se pudieron descargar datos históricos. Inténtalo de nuevo.")
    st.stop()

log_ret  = np.log(history / history.shift(1)).dropna()
mean_ret = log_ret.mean() * 252
cov      = log_ret.cov()   * 252

cur_w_dict               = dict(zip(current["ticker"], current["weight"] / 100))
cur_sharpe, cur_r, cur_v = calc_sharpe(current["ticker"].tolist(), cur_w_dict, mean_ret, cov)

if has_trades:
    sim_w_dict                   = dict(zip(simulated["ticker"], simulated["weight"] / 100))
    sim_sharpe, sim_r, sim_v     = calc_sharpe(simulated["ticker"].tolist(), sim_w_dict, mean_ret, cov)

common = [t for t in current["ticker"].tolist() if t in mean_ret.index]


# ── MONTE CARLO ────────────────────────────────────────────────────────────────

all_w_mc, port_r_mc, port_v_mc, port_s_mc = [], [], [], []
for _ in range(n_sim):
    w = np.random.dirichlet(np.ones(len(common)))
    r = np.dot(w, mean_ret[common])
    v = np.sqrt(w @ cov.loc[common, common].values @ w)
    all_w_mc.append(w); port_r_mc.append(r); port_v_mc.append(v); port_s_mc.append(r / v)

port_r_mc = np.array(port_r_mc)
port_v_mc = np.array(port_v_mc)
port_s_mc = np.array(port_s_mc)
minv_idx  = int(np.argmin(port_v_mc))


# ── SCIPY: PESOS ÓPTIMOS EXACTOS ──────────────────────────────────────────────

def neg_sharpe_opt(w):
    r = np.dot(w, mean_ret[common])
    v = np.sqrt(w @ cov.loc[common, common].values @ w)
    return -(r / v)

n_ast    = len(common)
opt_res  = minimize(
    neg_sharpe_opt,
    x0          = np.ones(n_ast) / n_ast,
    method      = "SLSQP",
    bounds      = [(0, 1)] * n_ast,
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
)
best_w_scipy = opt_res.x if opt_res.success else np.ones(n_ast) / n_ast
opt_r_scipy  = np.dot(best_w_scipy, mean_ret[common])
opt_v_scipy  = np.sqrt(best_w_scipy @ cov.loc[common, common].values @ best_w_scipy)
opt_s_scipy  = opt_r_scipy / opt_v_scipy


# ── CABECERA ───────────────────────────────────────────────────────────────────

st.title("🧪 Análisis con simulación")
if has_trades:
    st.caption(
        f"{len(st.session_state.trades_sim)} operación(es) en cola · "
        "Datos históricos consistentes en todos los cálculos."
    )
else:
    st.caption("Añade operaciones en el panel lateral para activar el análisis comparativo.")


# ── 1. DISTRIBUCIÓN ÓPTIMA ────────────────────────────────────────────────────

st.subheader("Distribución óptima según el modelo (Max Sharpe — scipy)")
st.caption(
    "Resultado exacto y reproducible. "
    "Referencia orientativa — los retornos pasados no garantizan resultados futuros."
)

opt_rows = []
for ticker, w_opt in zip(common, best_w_scipy):
    cur_w_val = cur_w_dict.get(ticker, 0) * 100
    row = {
        "Ticker":          ticker,
        "Peso actual (%)": round(cur_w_val, 1),
        "Peso óptimo (%)": round(w_opt * 100, 1),
        "vs Actual (pp)":  round(w_opt * 100 - cur_w_val, 1),
    }
    if has_trades:
        row["Peso hipotético (%)"] = round(sim_w_dict.get(ticker, 0) * 100, 1)
    opt_rows.append(row)

opt_df = pd.DataFrame(opt_rows).sort_values("Peso óptimo (%)", ascending=False)
st.dataframe(opt_df, use_container_width=True, hide_index=True)

st.divider()


# ── 2. FRONTERA EFICIENTE ──────────────────────────────────────────────────────

st.subheader("Frontera eficiente — Simulación Monte Carlo")
st.caption(
    "**◆ azul** = portfolio real.  "
    + ("**★ verde** = portfolio hipotético.  " if has_trades else "")
    + "**★ dorado** = Max Sharpe exacto (scipy)."
)

fig1, ax1 = plt.subplots(figsize=(10, 5))
sc = ax1.scatter(port_v_mc, port_r_mc, c=port_s_mc, cmap="viridis", alpha=0.4, s=10)
plt.colorbar(sc, ax=ax1, label="Sharpe Ratio")

ax1.scatter(opt_v_scipy, opt_r_scipy,
            color="gold", s=180, zorder=6,
            label=f"Max Sharpe — scipy (Sharpe {opt_s_scipy:.2f})")
ax1.scatter(port_v_mc[minv_idx], port_r_mc[minv_idx],
            color="cyan", s=150, zorder=5, label="Min Volatilidad")

if cur_v is not None and cur_r is not None:
    ax1.scatter(cur_v, cur_r, color="#00BFFF", s=220, marker="D", zorder=6,
                label=f"Real actual (Sharpe {cur_sharpe:.2f})")

if has_trades and sim_v is not None and sim_r is not None:
    ax1.scatter(sim_v, sim_r, color="#00FF88", s=220, marker="*", zorder=6,
                label=f"Hipotético (Sharpe {sim_sharpe:.2f})")

ax1.set_xlabel("Volatilidad anualizada (Riesgo)")
ax1.set_ylabel("Retorno anualizado")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
ax1.set_title("Frontera eficiente")
ax1.legend(fontsize=8)
sns.despine()
plt.tight_layout()
st.pyplot(fig1)

st.divider()

if not has_trades:
    st.info(
        "Añade operaciones en el panel lateral para activar la comparativa, "
        "el detalle de posiciones y el análisis de cambio de pesos."
    )
    st.stop()


# ── 3. COMPARATIVA ────────────────────────────────────────────────────────────

st.subheader("Comparativa: actual vs hipotético")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("#### 🔵 Portfolio actual")
    m1, m2, m3 = st.columns(3)
    m1.metric("Retorno esperado", f"{cur_r:.1%}" if cur_r else "—")
    m2.metric("Volatilidad",      f"{cur_v:.1%}" if cur_v else "—")
    m3.metric("Sharpe Ratio",     f"{cur_sharpe:.2f}" if cur_sharpe else "—")

with col_b:
    st.markdown("#### 🟢 Portfolio hipotético")
    s1, s2, s3 = st.columns(3)
    s1.metric("Retorno esperado", f"{sim_r:.1%}" if sim_r else "—",
              f"{sim_r - cur_r:+.1%}" if (sim_r and cur_r) else None)
    s2.metric("Volatilidad",      f"{sim_v:.1%}" if sim_v else "—",
              f"{sim_v - cur_v:+.1%}" if (sim_v and cur_v) else None,
              delta_color="inverse")
    s3.metric("Sharpe Ratio",     f"{sim_sharpe:.2f}" if sim_sharpe else "—",
              f"{sim_sharpe - cur_sharpe:+.2f}" if (sim_sharpe and cur_sharpe) else None)

if cash_net > 0.01:
    st.info(f"💰 Efectivo liberado por ventas: **${cash_net:,.2f}**")
elif cash_net < -0.01:
    st.info(f"💸 Efectivo invertido: **${abs(cash_net):,.2f}**")

st.divider()


# ── 4. DETALLE DE POSICIONES ───────────────────────────────────────────────────

st.subheader("Detalle de posiciones")

all_t = sorted(set(current["ticker"]) | set(simulated["ticker"]))
rows  = []
for t in all_t:
    c_row  = current[current["ticker"]     == t]
    s_row  = simulated[simulated["ticker"] == t]
    cur_sh = c_row["shares"].values[0]       if not c_row.empty else 0.0
    sim_sh = s_row["shares"].values[0]       if not s_row.empty else 0.0
    cur_w  = c_row["weight"].values[0]       if not c_row.empty else 0.0
    sim_w  = s_row["weight"].values[0]       if not s_row.empty else 0.0
    cur_mv = c_row["market_value"].values[0] if not c_row.empty else 0.0
    sim_mv = s_row["market_value"].values[0] if not s_row.empty else 0.0
    rows.append({
        "Ticker":             t,
        "Acc. actuales":      round(cur_sh, 3),
        "Acc. hipotéticas":   round(sim_sh, 3),
        "Δ Acciones":         round(sim_sh - cur_sh, 3),
        "Peso actual %":      round(cur_w, 1),
        "Peso hipotético %":  round(sim_w, 1),
        "Δ Peso (pp)":        round(sim_w - cur_w, 1),
        "Valor actual $":     round(cur_mv, 0),
        "Valor hipotético $": round(sim_mv, 0),
    })

comp_df = pd.DataFrame(rows)

def highlight_row(row):
    d = row["Δ Acciones"]
    if d > 0:   return ["background-color: #c6efce; color: #276221"] * len(row)
    elif d < 0: return ["background-color: #ffc7ce; color: #9c0006"] * len(row)
    return [""] * len(row)

st.dataframe(
    comp_df.style
        .apply(highlight_row, axis=1)
        .format({
            "Peso actual %":      "{:.1f}%",
            "Peso hipotético %":  "{:.1f}%",
            "Δ Peso (pp)":        "{:+.1f}",
            "Valor actual $":     "${:.0f}",
            "Valor hipotético $": "${:.0f}",
            "Δ Acciones":         "{:+.3f}",
        }),
    use_container_width=True, hide_index=True
)

st.divider()


# ── 5. CAMBIO DE PESOS ─────────────────────────────────────────────────────────

st.subheader("Cambio de pesos")
changed = comp_df[comp_df["Δ Peso (pp)"] != 0].sort_values("Δ Peso (pp)")

if not changed.empty:
    fig2, ax2 = plt.subplots(figsize=(9, max(3, len(changed) * 0.5)))
    colors = ["#3B6D11" if v > 0 else "#A32D2D" for v in changed["Δ Peso (pp)"]]
    ax2.barh(changed["Ticker"], changed["Δ Peso (pp)"], color=colors)
    ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Cambio en peso (puntos porcentuales)")
    ax2.set_title("Impacto de las operaciones en la asignación del portfolio")
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.info("Las operaciones no producen cambios significativos en la asignación.")
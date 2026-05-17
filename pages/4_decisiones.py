import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diario de decisiones", page_icon="📓", layout="wide")


# ── FUNCIONES ──────────────────────────────────────────────────────────────────

def load_decisions():
    try:
        df = pd.read_csv("data/decisions.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date", ascending=False).reset_index(drop=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=[
            "id","date","operations","period",
            "predicted_return","predicted_vol","predicted_sharpe",
            "cash_invested","notes",
            "actual_return","actual_vol","actual_sharpe","review_date"
        ])

def save_decisions(df):
    df.to_csv("data/decisions.csv", index=False)

def delete_decision(decision_id):
    df = load_decisions()
    df = df[df["id"] != decision_id].reset_index(drop=True)
    save_decisions(df)

def load_positions():
    try:
        return pd.read_csv("data/positions.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["ticker","shares","avg_price","date_added","category"])

@st.cache_data(ttl=300)
def get_current_prices(tickers):
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

def compute_current_sharpe(period="1y"):
    """Calcula el Sharpe ratio actual del portfolio real."""
    positions = load_positions()
    if positions.empty:
        return None, None, None

    ticker_list = positions["ticker"].tolist()
    tickers     = tuple(ticker_list)
    prices      = get_current_prices(tickers)

    positions = positions.copy()
    positions["current_price"] = positions["ticker"].map(prices)
    positions["market_value"]  = positions["shares"] * positions["current_price"]
    total_mv = positions["market_value"].sum()
    if total_mv == 0:
        return None, None, None
    positions["weight"] = positions["market_value"] / total_mv

    weights_dict = dict(zip(positions["ticker"], positions["weight"]))

    history = get_price_history(tickers, period)
    if history.empty:
        return None, None, None

    log_ret  = np.log(history / history.shift(1)).dropna()
    mean_ret = log_ret.mean() * 252
    cov      = log_ret.cov()   * 252

    # Alineación explícita por nombre de ticker
    available = [t for t in ticker_list if t in mean_ret.index]
    mr = mean_ret[available]
    cm = cov.loc[available, available]
    w  = np.array([weights_dict.get(t, 0) for t in available])
    w  = w / w.sum()

    r = float(np.dot(w, mr))
    v = float(np.sqrt(w @ cm.values @ w))
    s = round(r / v, 2) if v > 0 else None
    return s, round(r, 4), round(v, 4)


# ── CABECERA ───────────────────────────────────────────────────────────────────

st.title("📓 Diario de decisiones")
st.caption(
    "Historial de todas tus decisiones de inversión. "
    "Compara tus predicciones con la realidad para aprender y mejorar."
)

decisions = load_decisions()

if decisions.empty:
    st.info(
        "Aún no hay decisiones guardadas. "
        "Ve a **Análisis con simulación**, añade operaciones en la cola "
        "y pulsa **💾 Guardar esta decisión** al final de la página."
    )
    st.stop()


# ── CONTROLES ──────────────────────────────────────────────────────────────────

col_r, col_s = st.columns([3, 1])
with col_r:
    st.caption(f"{len(decisions)} decisión(es) guardada(s).")
with col_s:
    if st.button("🔄 Actualizar datos de mercado"):
        st.cache_data.clear()
        st.rerun()

st.divider()


# ── LISTA DE DECISIONES ────────────────────────────────────────────────────────

for idx, row in decisions.iterrows():
    has_actual = not pd.isna(row.get("actual_sharpe"))
    status     = "✅ Revisada" if has_actual else "⏳ Pendiente de revisión"
    date_str   = row["date"].strftime("%d %b %Y")

    with st.expander(
        f"📅 {date_str}  ·  {row['operations']}  ·  Sharpe predicho {row['predicted_sharpe']:.2f}  ·  {status}",
        expanded=(idx == 0)
    ):
        # ── Predicción vs Real ─────────────────────────────────────────────────
        col_pred, col_real = st.columns(2)

        with col_pred:
            st.markdown("**📊 Predicción en el momento de la decisión**")
            p1, p2, p3 = st.columns(3)
            p1.metric("Retorno esperado", f"{row['predicted_return']:.1%}")
            p2.metric("Volatilidad",      f"{row['predicted_vol']:.1%}")
            p3.metric("Sharpe",           f"{row['predicted_sharpe']:.2f}")
            st.caption(f"Período usado: **{row['period']}**")
            st.caption(f"Capital invertido: **${row['cash_invested']:,.0f}**")
            st.caption(f"Operaciones: {row['operations']}")
            if row.get("notes") and str(row["notes"]) not in ["nan","None",""]:
                st.markdown(f"> *{row['notes']}*")

        with col_real:
            st.markdown("**📈 Resultado real (portfolio actual)**")
            if has_actual:
                delta_s = row["actual_sharpe"] - row["predicted_sharpe"]
                delta_r = row["actual_return"]  - row["predicted_return"]
                delta_v = row["actual_vol"]     - row["predicted_vol"]
                r1, r2, r3 = st.columns(3)
                r1.metric("Retorno real",    f"{row['actual_return']:.1%}",
                           f"{delta_r:+.1%}")
                r2.metric("Volatilidad real", f"{row['actual_vol']:.1%}",
                           f"{delta_v:+.1%}", delta_color="inverse")
                r3.metric("Sharpe real",      f"{row['actual_sharpe']:.2f}",
                           f"{delta_s:+.2f}")
                st.caption(f"Revisado el: **{row['review_date']}**")

                # Mini evaluación
                if delta_s >= 0.1:
                    st.success("📈 El Sharpe mejoró más de lo predicho. Buena decisión.")
                elif delta_s >= -0.1:
                    st.info("➡️ El Sharpe se mantuvo cerca de la predicción.")
                else:
                    st.warning("📉 El Sharpe quedó por debajo de la predicción. Momento de analizar por qué.")
            else:
                st.info("Datos reales aún no actualizados.")
                if st.button("📊 Actualizar con datos actuales del portfolio",
                             key=f"upd_{row['id']}"):
                    with st.spinner("Calculando métricas actuales..."):
                        sharpe, ret, vol = compute_current_sharpe(row["period"])
                    if sharpe:
                        all_df = load_decisions()
                        mask   = all_df["id"] == row["id"]
                        all_df.loc[mask, "actual_sharpe"]  = sharpe
                        all_df.loc[mask, "actual_return"]  = ret
                        all_df.loc[mask, "actual_vol"]     = vol
                        all_df.loc[mask, "review_date"]    = pd.Timestamp.now().strftime("%Y-%m-%d")
                        save_decisions(all_df)
                        st.success("✅ Datos reales guardados.")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("No se pudieron calcular las métricas. Intenta de nuevo.")

        # ── Gráfico comparativo (si tiene datos reales) ────────────────────────
        if has_actual:
            st.markdown("---")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            metrics    = ["Retorno", "Volatilidad", "Sharpe"]
            pred_vals  = [row["predicted_return"] * 100,
                          row["predicted_vol"]     * 100,
                          row["predicted_sharpe"]]
            actual_vals = [row["actual_return"] * 100,
                           row["actual_vol"]    * 100,
                           row["actual_sharpe"]]
            x = np.arange(len(metrics))
            w = 0.35
            ax.bar(x - w/2, pred_vals,  w, label="Predicho", color="#185FA5", alpha=0.85)
            ax.bar(x + w/2, actual_vals, w, label="Real",     color="#3B6D11", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.set_title("Predicción vs realidad")
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── Botón eliminar ─────────────────────────────────────────────────────
        st.markdown("---")
        col_del, col_space = st.columns([1, 4])
        with col_del:
            if st.button("🗑️ Eliminar esta decisión", key=f"del_{row['id']}"):
                delete_decision(row["id"])
                st.success("Decisión eliminada.")
                st.rerun()

st.divider()


# ── RESUMEN GLOBAL ─────────────────────────────────────────────────────────────

reviewed = decisions.dropna(subset=["actual_sharpe"])

if len(reviewed) >= 1:
    st.subheader("Resumen global — Predicción vs realidad")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    labels = [row["date"].strftime("%d %b %y") for _, row in reviewed.iterrows()]
    x      = np.arange(len(reviewed))
    w      = 0.35

    ax2.bar(x - w/2, reviewed["predicted_sharpe"], w,
            label="Sharpe predicho", color="#185FA5", alpha=0.85)
    ax2.bar(x + w/2, reviewed["actual_sharpe"],    w,
            label="Sharpe real",     color="#3B6D11", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Todas las decisiones revisadas: predicción vs resultado real")
    ax2.legend()
    ax2.axhline(reviewed["predicted_sharpe"].mean(),
                color="#185FA5", linestyle="--", linewidth=0.8, alpha=0.6,
                label="Media predicho")
    ax2.axhline(reviewed["actual_sharpe"].mean(),
                color="#3B6D11", linestyle="--", linewidth=0.8, alpha=0.6,
                label="Media real")
    plt.tight_layout()
    st.pyplot(fig2)

    # Tabla resumen
    summary = reviewed[["date","operations","predicted_sharpe","actual_sharpe","review_date"]].copy()
    summary["date"]        = summary["date"].dt.strftime("%d %b %Y")
    summary["Δ Sharpe"]    = (summary["actual_sharpe"] - summary["predicted_sharpe"]).round(2)
    summary.columns        = ["Fecha","Operaciones","Sharpe predicho","Sharpe real","Revisado","Δ Sharpe"]

    def color_delta(val):
        if isinstance(val, float) and val >= 0: return "color: green"
        elif isinstance(val, float) and val < 0: return "color: red"
        return ""

    st.dataframe(
        summary.style.map(color_delta, subset=["Δ Sharpe"]),
        use_container_width=True, hide_index=True
    )
else:
    st.info(
        "El resumen global aparecerá cuando actualices los datos reales "
        "de al menos una decisión con el botón '📊 Actualizar'."
    )
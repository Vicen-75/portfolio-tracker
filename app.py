import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from datetime import date

st.set_page_config(page_title="Portfolio Tracker", page_icon="📈", layout="wide")


# ── FUNCIONES ──────────────────────────────────────────────────────────────────

def load_positions():
    try:
        return pd.read_csv("data/positions.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(
            columns=["ticker","shares","avg_price","date_added","category"])

def save_positions(df):
    df.to_csv("data/positions.csv", index=False)

def load_snapshots():
    try:
        df = pd.read_csv("data/snapshots.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(
            columns=["date","total_value","total_invested","total_pl","total_pl_pct","notes"])

def save_snapshot(total_value, total_invested, total_pl, total_pl_pct, notes=""):
    snapshots = load_snapshots()
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    if not snapshots.empty and today in snapshots["date"].dt.strftime("%Y-%m-%d").values:
        return False
    new_row = pd.DataFrame([{
        "date": today, "total_value": round(total_value, 2),
        "total_invested": round(total_invested, 2), "total_pl": round(total_pl, 2),
        "total_pl_pct": round(total_pl_pct, 2), "notes": notes,
    }])
    pd.concat([snapshots, new_row], ignore_index=True).to_csv("data/snapshots.csv", index=False)
    return True

@st.cache_data(ttl=300)
def get_current_prices(tickers):
    try:
        raw = yf.download(list(tickers), period="2d", auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            return {tickers[0]: float(raw.dropna().iloc[-1])}
        return {t: float(raw[t].dropna().iloc[-1]) for t in tickers if t in raw.columns}
    except Exception as e:
        st.error(f"Error al obtener precios: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_price_history(tickers, period="1y"):
    try:
        raw = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=tickers[0])
        return raw.dropna()
    except:
        return pd.DataFrame()


# ── CARGA Y CÁLCULO ────────────────────────────────────────────────────────────

positions = load_positions()
if positions.empty:
    st.title("📈 Portfolio Tracker")
    st.warning("No hay posiciones todavía. Añade la primera en el panel lateral.")
    st.stop()

tickers = tuple(positions["ticker"].tolist())
prices  = get_current_prices(tickers)

positions["current_price"] = positions["ticker"].map(prices)
positions["market_value"]  = positions["shares"] * positions["current_price"]
positions["cost_basis"]    = positions["shares"] * positions["avg_price"]
positions["pl_usd"]        = positions["market_value"] - positions["cost_basis"]
positions["pl_pct"]        = positions["pl_usd"] / positions["cost_basis"] * 100
positions["weight"]        = positions["market_value"] / positions["market_value"].sum() * 100

total_value    = positions["market_value"].sum()
total_invested = positions["cost_basis"].sum()
total_pl       = positions["pl_usd"].sum()
total_pl_pct   = total_pl / total_invested * 100
n_winners      = int((positions["pl_usd"] > 0).sum())


# ── HEADER ─────────────────────────────────────────────────────────────────────

st.title("📈 Portfolio Tracker")
st.caption(
    f"Actualizado: {pd.Timestamp.now().strftime('%d %b %Y, %H:%M')}  "
    "· Precios con ~15 min de delay (Yahoo Finance)"
)


# ── MÉTRICAS ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Valor total",           f"${total_value:,.0f}")
c2.metric("Capital invertido",     f"${total_invested:,.0f}")
c3.metric("Ganancia no realizada", f"${total_pl:,.0f}", f"{total_pl_pct:+.1f}%")
c4.metric("Posiciones ganadoras",  f"{n_winners} / {len(positions)}")

st.divider()


# ── TABLA DE POSICIONES ────────────────────────────────────────────────────────

st.subheader("Posiciones")

table = positions[[
    "ticker","category","shares","current_price",
    "market_value","cost_basis","pl_usd","pl_pct","weight"
]].sort_values("pl_usd", ascending=False).copy()
table.columns = [
    "Ticker","Tipo","Acciones","Precio actual",
    "Valor","Costo base","P&L $","P&L %","Peso %"
]

def color_pl(val):
    if isinstance(val, float) and val > 0: return "color: green"
    elif isinstance(val, float) and val < 0: return "color: red"
    return ""

st.dataframe(
    table.style
        .format({
            "Acciones":      "{:.3f}",
            "Precio actual": "${:.2f}",
            "Valor":         "${:.0f}",
            "Costo base":    "${:.0f}",
            "P&L $":         "${:+.0f}",
            "P&L %":         "{:+.1f}%",
            "Peso %":        "{:.1f}%",
        })
        .map(color_pl, subset=["P&L $","P&L %"]),
    use_container_width=True,
    hide_index=True,
    height=560          # muestra ~15 filas sin scroll
)

st.divider()


# ── GRÁFICOS ASIGNACIÓN Y P&L ─────────────────────────────────────────────────

COLORS = ["#185FA5","#534AB7","#D4537E","#BA7517","#888780","#1D9E75","#5F5E5A"]
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Asignación por categoría")
    cat_data = positions.groupby("category")["market_value"].sum()
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.pie(cat_data.values, labels=cat_data.index, autopct="%1.0f%%",
            startangle=90, colors=COLORS[:len(cat_data)])
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

with col_right:
    st.subheader("P&L por posición (USD)")
    pl_data    = positions.sort_values("pl_usd")
    bar_colors = ["#3B6D11" if v >= 0 else "#A32D2D" for v in pl_data["pl_usd"]]
    fig2, ax2  = plt.subplots(figsize=(5, 4))
    ax2.barh(pl_data["ticker"], pl_data["pl_usd"], color=bar_colors)
    ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("P&L (USD)")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()


# ── RETORNOS ACUMULADOS (PLOTLY — INTERACTIVO) ────────────────────────────────

st.subheader("Retornos acumulados (1 año)")

history = get_price_history(tickers)

if not history.empty:
    cum = (history / history.iloc[0] - 1) * 100

    # Mapa ticker → categoría
    ticker_cat = dict(zip(positions["ticker"], positions["category"]))

    # Colores por categoría
    CAT_COLORS = {
        "ETF": "#185FA5", "Tech": "#534AB7", "Salud": "#D4537E",
        "Materiales": "#BA7517", "Industrial": "#888780",
        "Consumo": "#1D9E75", "Financiero": "#5F5E5A", "Otro": "#888888"
    }

    # Controles
    col_view, col_filter = st.columns([1, 3])
    with col_view:
        view = st.radio(
            "Vista",
            ["Individual", "Por categoría"],
            horizontal=True,
            label_visibility="collapsed"
        )

    fig_cum = go.Figure()

    if view == "Individual":
        with col_filter:
            selected = st.multiselect(
                "Filtrar tickers",
                options=sorted(cum.columns.tolist()),
                default=sorted(cum.columns.tolist()),
                label_visibility="collapsed"
            )

        for ticker in selected:
            if ticker in cum.columns:
                cat   = ticker_cat.get(ticker, "Otro")
                color = CAT_COLORS.get(cat, "#888888")
                fig_cum.add_trace(go.Scatter(
                    x    = cum.index,
                    y    = cum[ticker].round(2),
                    name = ticker,
                    mode = "lines",
                    line = dict(color=color, width=1.8),
                    hovertemplate = (
                        f"<b>{ticker}</b> ({cat})<br>"
                        "%{x|%d %b %Y}<br>"
                        "<b>%{y:.1f}%</b><extra></extra>"
                    )
                ))
    else:
        # Vista por categoría: retorno medio ponderado por capitalización
        cat_groups = {}
        for col in cum.columns:
            cat = ticker_cat.get(col, "Otro")
            if cat not in cat_groups:
                cat_groups[cat] = []
            cat_groups[cat].append(col)

        for cat, cat_tickers in sorted(cat_groups.items()):
            tickers_in = [t for t in cat_tickers if t in cum.columns]
            if not tickers_in:
                continue
            # Media simple de los retornos de la categoría
            cat_avg = cum[tickers_in].mean(axis=1).round(2)
            label   = f"{cat} ({', '.join(tickers_in)})"
            fig_cum.add_trace(go.Scatter(
                x    = cat_avg.index,
                y    = cat_avg,
                name = cat,
                mode = "lines",
                line = dict(color=CAT_COLORS.get(cat, "#888888"), width=2.5),
                hovertemplate = (
                    f"<b>{cat}</b><br>"
                    f"<span style='font-size:11px'>{', '.join(tickers_in)}</span><br>"
                    "%{x|%d %b %Y}<br>"
                    "<b>%{y:.1f}%</b><extra></extra>"
                )
            ))

    # Línea en 0
    fig_cum.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)

    # Layout con selector de rango temporal
    fig_cum.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(count=3,  label="3M",  step="month", stepmode="backward"),
                    dict(count=6,  label="6M",  step="month", stepmode="backward"),
                    dict(count=1,  label="1A",  step="year",  stepmode="backward"),
                    dict(step="all", label="Todo"),
                ],
                bgcolor="#f0f0f0",
            ),
            type="date",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(title="Retorno (%)"),
        hovermode="x unified",
        height=460,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            font=dict(size=11)
        ),
        margin=dict(l=50, r=20, t=80, b=50),
    )

    st.plotly_chart(fig_cum, use_container_width=True)

    # Tabla de retorno total
    ret_table = cum.iloc[-1].reset_index()
    ret_table.columns = ["Ticker","Retorno (%)"]
    ret_table["Retorno (%)"] = ret_table["Retorno (%)"].round(2)
    st.dataframe(
        ret_table.sort_values("Retorno (%)", ascending=False),
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("No se pudieron cargar datos históricos.")

st.divider()


# ── EVOLUCIÓN DEL PATRIMONIO ───────────────────────────────────────────────────

st.subheader("📈 Evolución del patrimonio")
snapshots = load_snapshots()

if len(snapshots) < 2:
    st.info("Guarda tu primer snapshot hoy. El gráfico aparecerá con al menos dos snapshots.")
else:
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(snapshots["date"], snapshots["total_value"],
             color="#185FA5", linewidth=2, marker="o", label="Valor del portfolio")
    ax3.plot(snapshots["date"], snapshots["total_invested"],
             color="#888780", linewidth=1.5, linestyle="--", marker="o", label="Capital invertido")
    ax3.fill_between(snapshots["date"], snapshots["total_invested"], snapshots["total_value"],
                     where=(snapshots["total_value"] >= snapshots["total_invested"]),
                     alpha=0.15, color="#3B6D11", label="Rentabilidad del mercado")
    ax3.fill_between(snapshots["date"], snapshots["total_invested"], snapshots["total_value"],
                     where=(snapshots["total_value"] < snapshots["total_invested"]),
                     alpha=0.15, color="#A32D2D")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax3.set_title("Crecimiento del patrimonio mes a mes")
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    snap_table = snapshots[["date","total_value","total_invested","total_pl","total_pl_pct","notes"]].copy()
    snap_table.columns = ["Fecha","Valor total","Capital invertido","P&L $","P&L %","Notas"]
    st.dataframe(
        snap_table.style.format({
            "Valor total":       "${:,.0f}",
            "Capital invertido": "${:,.0f}",
            "P&L $":             "${:+,.0f}",
            "P&L %":             "{:+.1f}%",
        }),
        use_container_width=True,
        hide_index=True
    )


# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    if st.button("🔄 Actualizar datos de mercado"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Fuerza una nueva descarga desde Yahoo Finance.")

    st.divider()
    st.header("➕ Añadir posición")
    with st.form("nueva_posicion"):
        ticker_new    = st.text_input("Ticker", placeholder="AAPL").upper().strip()
        category_new  = st.selectbox("Categoría",
            ["ETF","Tech","Salud","Financiero","Consumo","Materiales","Industrial","Otro"])
        shares_new    = st.number_input("Acciones", min_value=0.001, step=0.001, format="%.3f")
        avg_price_new = st.number_input("Precio medio ($)", min_value=0.01, step=0.01)
        date_new      = st.date_input("Fecha de compra", value=date.today())
        submitted     = st.form_submit_button("Añadir")
        if submitted and ticker_new and shares_new > 0 and avg_price_new > 0:
            new_row = pd.DataFrame([{
                "ticker": ticker_new, "shares": shares_new,
                "avg_price": avg_price_new,
                "date_added": date_new.strftime("%Y-%m-%d"),
                "category": category_new,
            }])
            updated = pd.concat([load_positions(), new_row], ignore_index=True)
            save_positions(updated)
            st.success(f"✅ {ticker_new} añadido")
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.header("📸 Snapshot mensual")
    snapshots_sb = load_snapshots()
    if not snapshots_sb.empty:
        last = snapshots_sb.iloc[-1]
        st.caption(f"Último: {last['date'].strftime('%d %b %Y')}")
        st.caption(f"Valor: ${last['total_value']:,.0f}  ·  P&L: {last['total_pl_pct']:+.1f}%")
    else:
        st.caption("Aún no hay snapshots guardados.")
    note_input = st.text_input("Nota del mes", placeholder="Compré VOO este mes...")
    if st.button("💾 Guardar snapshot de hoy"):
        saved = save_snapshot(total_value, total_invested, total_pl, total_pl_pct, note_input)
        if saved:
            st.success("✅ Snapshot guardado")
            st.rerun()
        else:
            st.warning("Ya guardaste un snapshot hoy.")
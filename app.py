import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

st.set_page_config(page_title="Portfolio Tracker", page_icon="📈", layout="wide")


# ── CLASIFICACIÓN DE TICKERS ───────────────────────────────────────────────────
# Fuente única de clasificación. Añade aquí cualquier ticker nuevo.

CLASSIFICATION = {
    "ASML":  {"asset_class": "Equity", "sector": "Technology",           "industry": "Semiconductor Equipment"},
    "QQQ":   {"asset_class": "ETF",    "sector": "Technology",           "industry": "Nasdaq 100"},
    "VOO":   {"asset_class": "ETF",    "sector": "Broad Market",         "industry": "S&P 500"},
    "TSM":   {"asset_class": "Equity", "sector": "Technology",           "industry": "Semiconductors"},
    "JNJ":   {"asset_class": "Equity", "sector": "Healthcare",           "industry": "Pharmaceuticals"},
    "NVDA":  {"asset_class": "Equity", "sector": "Technology",           "industry": "Semiconductors"},
    "WPM":   {"asset_class": "Equity", "sector": "Basic Materials",      "industry": "Precious Metals"},
    "VXUS":  {"asset_class": "ETF",    "sector": "International Equity", "industry": "Global ex-US"},
    "SBGSY": {"asset_class": "Equity", "sector": "Industrial",           "industry": "Electrical Equipment"},
    "VEA":   {"asset_class": "ETF",    "sector": "International Equity", "industry": "Developed Markets"},
    "BND":   {"asset_class": "ETF",    "sector": "Fixed Income",         "industry": "Gov & Corp Bonds"},
    "VWO":   {"asset_class": "ETF",    "sector": "International Equity", "industry": "Emerging Markets"},
    "PG":    {"asset_class": "Equity", "sector": "Consumer",             "industry": "Consumer Staples"},
    "JPM":   {"asset_class": "Equity", "sector": "Financial",            "industry": "Banks"},
    "NVO":   {"asset_class": "Equity", "sector": "Healthcare",           "industry": "Pharmaceuticals"},
    "LLY":   {"asset_class": "Equity", "sector": "Healthcare",           "industry": "Pharmaceuticals"},
    "XLE":   {"asset_class": "ETF",    "sector": "Energy",               "industry": "Energy Sector"},
}

# Paleta de colores por sector
SECTOR_COLORS = {
    "Technology":           "#534AB7",
    "Healthcare":           "#D4537E",
    "Financial":            "#5F5E5A",
    "Basic Materials":      "#BA7517",
    "Industrial":           "#888780",
    "Consumer":             "#1D9E75",
    "Energy":               "#E67E22",
    "International Equity": "#2196F3",
    "Fixed Income":         "#78909C",
    "Broad Market":         "#43A047",
    "Other":                "#BDBDBD",
}


# ── FUNCIONES ──────────────────────────────────────────────────────────────────

def load_positions():
    try:
        df = pd.read_csv("data/positions.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=[
            "ticker","shares","avg_price","date_added","category",
            "asset_class","sector","industry"
        ])
    # Aplicar clasificación desde el dict para tickers conocidos
    for col, key in [("asset_class","asset_class"),("sector","sector"),("industry","industry")]:
        if col not in df.columns:
            df[col] = df["ticker"].map(lambda t: CLASSIFICATION.get(t, {}).get(key, "Other"))
        else:
            df[col] = df.apply(
                lambda row: CLASSIFICATION[row["ticker"]][key]
                if row["ticker"] in CLASSIFICATION
                else (row[col] if pd.notna(row[col]) else "Other"),
                axis=1
            )
    return df

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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Valor total",           f"${total_value:,.0f}")
c2.metric("Capital invertido",     f"${total_invested:,.0f}")
c3.metric("Ganancia no realizada", f"${total_pl:,.0f}", f"{total_pl_pct:+.1f}%")
c4.metric("Posiciones ganadoras",  f"{n_winners} / {len(positions)}")

st.divider()


# ── RETORNOS ACUMULADOS (PLOTLY) ───────────────────────────────────────────────
# Posición: justo después de las métricas, antes de la tabla

st.subheader("Retornos acumulados (1 año)")

history = get_price_history(tickers)

if not history.empty:
    cum = (history / history.iloc[0] - 1) * 100

    # ── Filtros por clasificación ──────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    with f1:
        view = st.radio("Vista", ["Individual","Por categoría"],
                        horizontal=True, label_visibility="collapsed")
    with f2:
        sel_class = st.multiselect(
            "Asset Class",
            sorted(positions["asset_class"].unique()),
            default=sorted(positions["asset_class"].unique()),
            label_visibility="collapsed",
            placeholder="Asset Class…"
        )
    with f3:
        sel_sector = st.multiselect(
            "Sector",
            sorted(positions["sector"].unique()),
            default=sorted(positions["sector"].unique()),
            label_visibility="collapsed",
            placeholder="Sector…"
        )
    with f4:
        sel_industry = st.multiselect(
            "Industry",
            sorted(positions["industry"].unique()),
            default=sorted(positions["industry"].unique()),
            label_visibility="collapsed",
            placeholder="Industry…"
        )

    # Tickers filtrados
    mask = (
        positions["asset_class"].isin(sel_class) &
        positions["sector"].isin(sel_sector) &
        positions["industry"].isin(sel_industry)
    )
    filtered_tickers = positions[mask]["ticker"].tolist()

    fig_cum = go.Figure()

    if view == "Individual":
        for _, pos_row in positions[mask].iterrows():
            t     = pos_row["ticker"]
            color = SECTOR_COLORS.get(pos_row["sector"], "#888")
            if t in cum.columns:
                fig_cum.add_trace(go.Scatter(
                    x    = cum.index,
                    y    = cum[t].round(2),
                    name = t,
                    mode = "lines",
                    line = dict(color=color, width=1.5),
                    hovertemplate=(
                        f"<b>{t}</b> · {pos_row['sector']}<br>"
                        "%{x|%d %b %Y}<br><b>%{y:.1f}%</b><extra></extra>"
                    )
                ))
    else:
        # Retorno medio por sector (de los tickers filtrados)
        for sector in sorted(positions[mask]["sector"].unique()):
            tickers_in = positions[mask & (positions["sector"] == sector)]["ticker"].tolist()
            tickers_in = [t for t in tickers_in if t in cum.columns]
            if tickers_in:
                avg = cum[tickers_in].mean(axis=1).round(2)
                fig_cum.add_trace(go.Scatter(
                    x    = avg.index, y = avg,
                    name = sector, mode = "lines",
                    line = dict(color=SECTOR_COLORS.get(sector,"#888"), width=2.5),
                    hovertemplate=(
                        f"<b>{sector}</b><br>"
                        f"<span style='font-size:11px'>{', '.join(tickers_in)}</span><br>"
                        "%{x|%d %b %Y}<br><b>%{y:.1f}%</b><extra></extra>"
                    )
                ))

    # ── Línea del portfolio total (siempre visible) ────────────────────────────
    total_mv   = positions["market_value"].sum()
    port_cum   = pd.Series(0.0, index=history.index)
    for _, row in positions.iterrows():
        t = row["ticker"]
        if t in cum.columns:
            port_cum += (row["market_value"] / total_mv) * cum[t]

    fig_cum.add_trace(go.Scatter(
        x    = port_cum.index,
        y    = port_cum.round(2),
        name = "📊 Portfolio total",
        mode = "lines",
        line = dict(color="#FF6B35", width=4, dash="solid"),
        hovertemplate=(
            "<b>Portfolio total</b><br>"
            "%{x|%d %b %Y}<br><b>%{y:.1f}%</b><extra></extra>"
        )
    ))

    fig_cum.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.35)
    fig_cum.update_layout(
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1A", step="year",  stepmode="backward"),
                dict(step="all", label="Todo"),
            ]),
            type="date", rangeslider=dict(visible=False),
        ),
        yaxis=dict(title="Retorno (%)"),
        hovermode="x unified",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=50, r=20, t=80, b=50),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Tabla de retorno total al final del período
    ret_table = cum.iloc[-1].reset_index()
    ret_table.columns = ["Ticker","Retorno (%)"]
    ret_table["Retorno (%)"] = ret_table["Retorno (%)"].round(2)
    st.dataframe(
        ret_table.sort_values("Retorno (%)", ascending=False),
        use_container_width=True, hide_index=True
    )
else:
    st.warning("No se pudieron cargar datos históricos.")

st.divider()


# ── TABLA DE POSICIONES ────────────────────────────────────────────────────────

st.subheader("Posiciones")

table = positions[[
    "ticker","asset_class","sector","industry",
    "shares","current_price","market_value","cost_basis","pl_usd","pl_pct","weight"
]].sort_values("pl_usd", ascending=False).copy()

table.columns = [
    "Ticker","Class","Sector","Industry",
    "Acc.","Precio","Valor","Costo","P&L $","P&L %","Peso %"
]

def color_pl(val):
    if isinstance(val, float) and val > 0: return "color: green"
    elif isinstance(val, float) and val < 0: return "color: red"
    return ""

st.dataframe(
    table.style
        .format({
            "Acc.":    "{:.3f}",
            "Precio":  "${:.2f}",
            "Valor":   "${:.0f}",
            "Costo":   "${:.0f}",
            "P&L $":   "${:+.0f}",
            "P&L %":   "{:+.1f}%",
            "Peso %":  "{:.1f}%",
        })
        .map(color_pl, subset=["P&L $","P&L %"]),
    column_config={
        "Ticker":   st.column_config.TextColumn("Ticker",   width="small"),
        "Class":    st.column_config.TextColumn("Class",    width="small"),
        "Sector":   st.column_config.TextColumn("Sector",   width="medium"),
        "Industry": st.column_config.TextColumn("Industry", width="medium"),
        "Acc.":     st.column_config.TextColumn("Acc.",     width="small"),
        "Precio":   st.column_config.TextColumn("Precio",   width="small"),
        "Valor":    st.column_config.TextColumn("Valor",    width="small"),
        "Costo":    st.column_config.TextColumn("Costo",    width="small"),
        "P&L $":    st.column_config.TextColumn("P&L $",    width="small"),
        "P&L %":    st.column_config.TextColumn("P&L %",    width="small"),
        "Peso %":   st.column_config.TextColumn("Peso %",   width="small"),
    },
    use_container_width=True,
    hide_index=True,
    height=560
)

st.divider()


# ── ASIGNACIÓN — 3 DONUTS (PLOTLY) ────────────────────────────────────────────

st.subheader("Asignación del portfolio")

fig_alloc = make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],
    subplot_titles=["Asset Class", "Sector", "Industry"]
)

# Paleta de colores auto-asignada por categoría
def pie_colors(labels, color_map):
    return [color_map.get(l, "#BDBDBD") for l in labels]

ASSET_COLORS = {"ETF": "#185FA5", "Equity": "#534AB7"}
INDUSTRY_COLORS = {
    "Semiconductor Equipment": "#7C4DFF",
    "Nasdaq 100":              "#3D5AFE",
    "S&P 500":                 "#43A047",
    "Semiconductors":          "#651FFF",
    "Pharmaceuticals":         "#E91E63",
    "Precious Metals":         "#FFB300",
    "Global ex-US":            "#1565C0",
    "Electrical Equipment":    "#546E7A",
    "Developed Markets":       "#1976D2",
    "Gov & Corp Bonds":        "#78909C",
    "Emerging Markets":        "#0288D1",
    "Consumer Staples":        "#00897B",
    "Banks":                   "#455A64",
    "Energy Sector":           "#F57C00",
    "Other":                   "#BDBDBD",
}

for col_idx, (group_col, color_map, hover_extra) in enumerate([
    ("asset_class", ASSET_COLORS,    ""),
    ("sector",      SECTOR_COLORS,   ""),
    ("industry",    INDUSTRY_COLORS, ""),
], start=1):
    grp  = positions.groupby(group_col)["market_value"].sum().reset_index()
    grp.columns = ["label","value"]
    fig_alloc.add_trace(
        go.Pie(
            labels   = grp["label"],
            values   = grp["value"].round(0),
            name     = group_col,
            hole     = 0.42,
            marker   = dict(colors=pie_colors(grp["label"].tolist(), color_map)),
            textinfo = "percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "$%{value:,.0f}<br>"
                "%{percent}<extra></extra>"
            ),
        ),
        row=1, col=col_idx
    )

fig_alloc.update_layout(
    height      = 380,
    showlegend  = True,
    legend      = dict(orientation="v", font=dict(size=10)),
    margin      = dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig_alloc, use_container_width=True)

st.divider()


# ── P&L POR POSICIÓN (PLOTLY) ─────────────────────────────────────────────────

st.subheader("P&L por posición")

pl_sorted = positions.sort_values("pl_usd").copy()
pl_colors = ["#3B6D11" if v >= 0 else "#A32D2D" for v in pl_sorted["pl_usd"]]

fig_pl = go.Figure(go.Bar(
    x            = pl_sorted["pl_usd"].round(2),
    y            = pl_sorted["ticker"],
    orientation  = "h",
    marker       = dict(color=pl_colors),
    customdata   = pl_sorted[["pl_pct","market_value","cost_basis","sector","industry"]].values,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "P&L:         $%{x:+,.0f}<br>"
        "P&L %:       %{customdata[0]:+.1f}%<br>"
        "Valor actual: $%{customdata[1]:,.0f}<br>"
        "Costo base:   $%{customdata[2]:,.0f}<br>"
        "Sector:       %{customdata[3]}<br>"
        "Industry:     %{customdata[4]}<extra></extra>"
    )
))
fig_pl.add_vline(x=0, line_color="gray", line_dash="dash", opacity=0.4)
fig_pl.update_layout(
    height   = max(350, len(pl_sorted) * 30),
    xaxis    = dict(title="P&L (USD)"),
    yaxis    = dict(title=""),
    showlegend = False,
    margin   = dict(l=80, r=20, t=20, b=50),
)
st.plotly_chart(fig_pl, use_container_width=True)

st.divider()


# ── EVOLUCIÓN DEL PATRIMONIO ───────────────────────────────────────────────────

st.subheader("📈 Evolución del patrimonio")
snapshots = load_snapshots()

if len(snapshots) < 2:
    st.info("Guarda tu primer snapshot. El gráfico aparecerá con al menos dos snapshots.")
else:
    fig_snap, ax_snap = plt.subplots(figsize=(10, 4))
    ax_snap.plot(snapshots["date"], snapshots["total_value"],
                 color="#185FA5", linewidth=2, marker="o", label="Valor del portfolio")
    ax_snap.plot(snapshots["date"], snapshots["total_invested"],
                 color="#888780", linewidth=1.5, linestyle="--", marker="o", label="Capital invertido")
    ax_snap.fill_between(snapshots["date"], snapshots["total_invested"], snapshots["total_value"],
                         where=(snapshots["total_value"] >= snapshots["total_invested"]),
                         alpha=0.15, color="#3B6D11", label="Rentabilidad del mercado")
    ax_snap.fill_between(snapshots["date"], snapshots["total_invested"], snapshots["total_value"],
                         where=(snapshots["total_value"] < snapshots["total_invested"]),
                         alpha=0.15, color="#A32D2D")
    ax_snap.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_snap.set_title("Crecimiento del patrimonio mes a mes")
    ax_snap.legend()
    plt.tight_layout()
    st.pyplot(fig_snap)
    plt.close(fig_snap)

    snap_table = snapshots[["date","total_value","total_invested","total_pl","total_pl_pct","notes"]].copy()
    snap_table.columns = ["Fecha","Valor total","Capital invertido","P&L $","P&L %","Notas"]
    st.dataframe(
        snap_table.style.format({
            "Valor total":       "${:,.0f}",
            "Capital invertido": "${:,.0f}",
            "P&L $":             "${:+,.0f}",
            "P&L %":             "{:+.1f}%",
        }),
        use_container_width=True, hide_index=True
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
        ticker_new = st.text_input("Ticker", placeholder="AAPL").upper().strip()

        # Opciones de clasificación derivadas del dict existente + "Other"
        ac_opts  = sorted({v["asset_class"] for v in CLASSIFICATION.values()})
        sec_opts = sorted({v["sector"]      for v in CLASSIFICATION.values()}) + ["Other"]
        ind_opts = sorted({v["industry"]    for v in CLASSIFICATION.values()}) + ["Other"]

        # Si el ticker ya está en el dict, informar al usuario
        if ticker_new in CLASSIFICATION:
            known = CLASSIFICATION[ticker_new]
            st.info(
                f"Clasificación automática: "
                f"**{known['asset_class']}** · {known['sector']} · {known['industry']}"
            )

        asset_class_new = st.selectbox("Asset Class", ac_opts)
        sector_new      = st.selectbox("Sector",      sec_opts)
        industry_new    = st.selectbox("Industry",    ind_opts)
        shares_new      = st.number_input("Acciones",        min_value=0.001, step=0.001, format="%.3f")
        avg_price_new   = st.number_input("Precio medio ($)", min_value=0.01,  step=0.01)
        date_new        = st.date_input("Fecha de compra", value=date.today())
        submitted       = st.form_submit_button("Añadir")

        if submitted and ticker_new and shares_new > 0 and avg_price_new > 0:
            new_row = pd.DataFrame([{
                "ticker":      ticker_new,
                "shares":      shares_new,
                "avg_price":   avg_price_new,
                "date_added":  date_new.strftime("%Y-%m-%d"),
                "category":    sector_new,       # compatibilidad con CSV anterior
                "asset_class": asset_class_new,
                "sector":      sector_new,
                "industry":    industry_new,
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
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date



# ── CLASIFICACIÓN ──────────────────────────────────────────────────────────────

CLASSIFICATION = {
    "ASML":  {"asset_class": "Equity", "sector": "Technology",           "industry": "Semiconductor Eq."},
    "QQQ":   {"asset_class": "ETF",    "sector": "Technology",           "industry": "Nasdaq 100"},
    "VOO":   {"asset_class": "ETF",    "sector": "Broad Market",         "industry": "S&P 500"},
    "TSM":   {"asset_class": "Equity", "sector": "Technology",           "industry": "Semiconductors"},
    "JNJ":   {"asset_class": "Equity", "sector": "Healthcare",           "industry": "Pharma."},
    "NVDA":  {"asset_class": "Equity", "sector": "Technology",           "industry": "Semiconductors"},
    "WPM":   {"asset_class": "Equity", "sector": "Basic Materials",      "industry": "Precious Metals"},
    "VXUS":  {"asset_class": "ETF",    "sector": "International Equity", "industry": "Global ex-US"},
    "SBGSY": {"asset_class": "Equity", "sector": "Industrial",           "industry": "Electrical Eq."},
    "VEA":   {"asset_class": "ETF",    "sector": "International Equity", "industry": "Developed Markets"},
    "BND":   {"asset_class": "ETF",    "sector": "Fixed Income",         "industry": "Gov & Corp Bonds"},
    "VWO":   {"asset_class": "ETF",    "sector": "International Equity", "industry": "Emerging Markets"},
    "PG":    {"asset_class": "Equity", "sector": "Consumer",             "industry": "Staples"},
    "JPM":   {"asset_class": "Equity", "sector": "Financial",            "industry": "Banks"},
    "NVO":   {"asset_class": "Equity", "sector": "Healthcare",           "industry": "Pharma."},
    "LLY":   {"asset_class": "Equity", "sector": "Healthcare",           "industry": "Pharma."},
    "XLE":   {"asset_class": "ETF",    "sector": "Energy",               "industry": "Energy Sector"},
}

SECTOR_COLORS = {
    "Technology":           "#9092D5",
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
ASSET_COLORS = {"ETF": "#DCDCDC", "Equity": "#F3E189"}
INDUSTRY_COLORS = {
    "Semiconductor Eq.": "#7C4DFF",
    "Nasdaq 100":         "#3D5AFE",
    "S&P 500":            "#43A047",
    "Semiconductors":     "#651FFF",
    "Pharma.":            "#E91E63",
    "Precious Metals":    "#FFB300",
    "Global ex-US":       "#1565C0",
    "Electrical Eq.":     "#546E7A",
    "Developed Markets":  "#1976D2",
    "Gov & Corp Bonds":   "#78909C",
    "Emerging Markets":   "#0288D1",
    "Staples":            "#00897B",
    "Banks":              "#455A64",
    "Energy Sector":      "#F57C00",
    "Other":              "#BDBDBD",
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


# ── 1. HEADER Y MÉTRICAS ───────────────────────────────────────────────────────

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


# ── 2. ASIGNACIÓN DEL PORTFOLIO ───────────────────────────────────────────────

st.subheader("Asignación del portfolio")

col_donut, col_tree = st.columns([1, 2])

# ── Donut: Asset Class ────────────────────────────────────────────────────────
with col_donut:
    ac_grp = positions.groupby("asset_class")["market_value"].sum().reset_index()
    fig_ac = go.Figure(go.Pie(
        labels           = ac_grp["asset_class"],
        values           = ac_grp["market_value"].round(0),
        hole             = 0.42,
        marker           = dict(colors=[ASSET_COLORS.get(l,"#BDBDBD") for l in ac_grp["asset_class"]]),
        textinfo         = "label+percent",
        textposition     = "inside",
        insidetextorientation = "auto",
        texttemplate     = "%{label}<br>%{percent:.0%}",
        showlegend       = False,
        hovertemplate    = "<b>%{label}</b><br>$%{value:,.0f} · %{percent:.1%}<extra></extra>",
    ))
    fig_ac.update_layout(
        title  = dict(text="Asset Class", x=0.5, xanchor="center", font=dict(size=13)),
        height = 300,
        margin = dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_ac, use_container_width=True)

# ── Treemap: Sector → Industry ────────────────────────────────────────────────
with col_tree:
    sec_grp = positions.groupby("sector")["market_value"].sum().reset_index()
    ind_grp = positions.groupby(["sector","industry"])["market_value"].sum().reset_index()

    ids, labels_t, parents, values_t, colors_t = [], [], [], [], []

    for _, row in sec_grp.iterrows():
        ids.append(row["sector"])
        labels_t.append(row["sector"])
        parents.append("")
        values_t.append(row["market_value"])
        colors_t.append(SECTOR_COLORS.get(row["sector"],"#BDBDBD"))

    for _, row in ind_grp.iterrows():
        uid = f"{row['sector']}|{row['industry']}"
        ids.append(uid)
        labels_t.append(row["industry"])
        parents.append(row["sector"])
        values_t.append(row["market_value"])
        colors_t.append(SECTOR_COLORS.get(row["sector"],"#BDBDBD"))

    fig_tree = go.Figure(go.Treemap(
        ids           = ids,
        labels        = labels_t,
        parents       = parents,
        values        = values_t,
        branchvalues  = "total",
        marker        = dict(colors=colors_t),
        texttemplate  = "<b>%{label}</b><br>%{percentRoot:.0%}",
        hovertemplate = (
            "<b>%{label}</b><br>"
            "$%{value:,.0f}<br>"
            "%{percentRoot:.1%} del total<extra></extra>"
        ),
        tiling        = dict(packing="squarify"),
    ))
    fig_tree.update_layout(
        title  = dict(text="Sector & Industry", x=0.5, xanchor="center", font=dict(size=13)),
        height = 300,
        margin = dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_tree, use_container_width=True)

# ── Sunburst oculto — despliega para comparar ─────────────────────────────────
with st.expander("🌞 Ver como Sunburst (Asset Class → Sector → Industry)", expanded=False):

    sun_grp = positions.groupby(
        ["asset_class","sector","industry"]
    )["market_value"].sum().reset_index()

    ids_s, labels_s, parents_s, values_s, colors_s = [], [], [], [], []

    # Nivel 1: Asset Class
    for ac in sun_grp["asset_class"].unique():
        ids_s.append(ac)
        labels_s.append(ac)
        parents_s.append("")
        values_s.append(sun_grp[sun_grp["asset_class"]==ac]["market_value"].sum())
        colors_s.append(ASSET_COLORS.get(ac,"#BDBDBD"))

    # Nivel 2: Sector dentro de Asset Class
    for (ac, sec), grp in sun_grp.groupby(["asset_class","sector"]):
        uid = f"{ac}|{sec}"
        ids_s.append(uid)
        labels_s.append(sec)
        parents_s.append(ac)
        values_s.append(grp["market_value"].sum())
        colors_s.append(SECTOR_COLORS.get(sec,"#BDBDBD"))

    # Nivel 3: Industry dentro de Sector × Asset Class
    for _, row in sun_grp.iterrows():
        uid = f"{row['asset_class']}|{row['sector']}|{row['industry']}"
        ids_s.append(uid)
        labels_s.append(row["industry"])
        parents_s.append(f"{row['asset_class']}|{row['sector']}")
        values_s.append(row["market_value"])
        colors_s.append(SECTOR_COLORS.get(row["sector"],"#BDBDBD"))

    fig_sun = go.Figure(go.Sunburst(
        ids          = ids_s,
        labels       = labels_s,
        parents      = parents_s,
        values       = values_s,
        branchvalues = "total",
        marker       = dict(colors=colors_s),
        hovertemplate= (
            "<b>%{label}</b><br>"
            "$%{value:,.0f}<br>"
            "%{percentRoot:.1%} del total"
            "<extra></extra>"
        ),
        texttemplate         = "%{label}<br>%{percentRoot:.0%}",
        insidetextorientation= "radial",
    ))
    fig_sun.update_layout(
        height = 500,
        margin = dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_sun, use_container_width=True)

st.divider()


# ── 3. TABLA DE POSICIONES ────────────────────────────────────────────────────

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
    if isinstance(val, float) and val > 0:  return "color: green"
    elif isinstance(val, float) and val < 0: return "color: red"
    return ""

st.dataframe(
    table.style
        .format({
            "Acc.":   "{:.3f}",
            "Precio": "${:.2f}",
            "Valor":  "${:.0f}",
            "Costo":  "${:.0f}",
            "P&L $":  "${:+.0f}",
            "P&L %":  "{:+.1f}%",
            "Peso %": "{:.1f}%",
        })
        .map(color_pl, subset=["P&L $","P&L %"]),
    column_config={
        "Ticker":   st.column_config.TextColumn("Ticker"),
        "Class":    st.column_config.TextColumn("Class"),
        "Sector":   st.column_config.TextColumn("Sector"),
        "Industry": st.column_config.TextColumn("Industry"),
        "Acc.":     st.column_config.NumberColumn("Acc.",    format="%.3f"),
        "Precio":   st.column_config.NumberColumn("Precio",  format="$%.2f"),
        "Valor":    st.column_config.NumberColumn("Valor",   format="$%.0f"),
        "Costo":    st.column_config.NumberColumn("Costo",   format="$%.0f"),
        "P&L $":    st.column_config.NumberColumn("P&L $",   format="$%.0f"),
        "P&L %":    st.column_config.NumberColumn("P&L %",   format="%.1f"),
        "Peso %":   st.column_config.NumberColumn("Peso %",  format="%.1f"),
    },
    use_container_width=True,
    hide_index=True,
    height=560
)

st.divider()


# ── 4. RETORNOS ACUMULADOS (PLOTLY) ───────────────────────────────────────────

st.subheader("Retornos acumulados (1 año)")

history = get_price_history(tickers)

if not history.empty:
    cum = (history / history.iloc[0] - 1) * 100

    # ── Controles: vista + filtros en expander ──────────────────────────────────
    ctrl_left, ctrl_right = st.columns([2, 4])
    with ctrl_left:
        view = st.radio(
            "Vista",
            ["Individual", "Por categoría"],
            horizontal=True,
            label_visibility="collapsed"
        )

    # Filtros dentro de expander — no ocupan espacio fijo
    with st.expander("🔍 Filtrar por clasificación", expanded=False):
        fa, fb, fc = st.columns(3)
        with fa:
            sel_class = st.multiselect(
                "Asset Class",
                sorted(positions["asset_class"].unique()),
                default=sorted(positions["asset_class"].unique()),
            )
        with fb:
            sel_sector = st.multiselect(
                "Sector",
                sorted(positions["sector"].unique()),
                default=sorted(positions["sector"].unique()),
            )
        with fc:
            sel_industry = st.multiselect(
                "Industry",
                sorted(positions["industry"].unique()),
                default=sorted(positions["industry"].unique()),
            )
    # Si el expander nunca se abre, usar todos por defecto
    if "sel_class"    not in dir(): sel_class    = sorted(positions["asset_class"].unique())
    if "sel_sector"   not in dir(): sel_sector   = sorted(positions["sector"].unique())
    if "sel_industry" not in dir(): sel_industry = sorted(positions["industry"].unique())

    mask = (
        positions["asset_class"].isin(sel_class) &
        positions["sector"].isin(sel_sector) &
        positions["industry"].isin(sel_industry)
    )
    filtered_pos = positions[mask]

    fig_cum = go.Figure()

    if view == "Individual":
        for _, pos_row in filtered_pos.iterrows():
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
        for sector in sorted(filtered_pos["sector"].unique()):
            tickers_in = filtered_pos[filtered_pos["sector"] == sector]["ticker"].tolist()
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

    # Línea del portfolio total (siempre visible, naranja gruesa)
    total_mv  = positions["market_value"].sum()
    port_cum  = pd.Series(0.0, index=history.index)
    for _, row in positions.iterrows():
        t = row["ticker"]
        if t in cum.columns:
            port_cum += (row["market_value"] / total_mv) * cum[t]

    fig_cum.add_trace(go.Scatter(
        x    = port_cum.index,
        y    = port_cum.round(2),
        name = "📊 Portfolio",
        mode = "lines",
        line = dict(color="#FF6B35", width=4),
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

            showspikes    = True,
            spikemode     = "across",
            spikesnap     = "cursor",
            spikecolor    = "gray",
            spikedash     = "dot",
            spikethickness= 1,
            
        ),
        yaxis=dict(title="Retorno (%)"),
        hovermode="closest",
        height=460,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",   # fondo transparente = más discreto
        ),
        margin=dict(l=50, r=20, t=70, b=50),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Tabla de retornos totales — compacta, dentro de expander
    with st.expander("📋 Retorno total por ticker al cierre del período", expanded=False):
        ret_raw = cum.iloc[-1].reset_index()
        ret_raw.columns = ["Ticker","Retorno (%)"]
        ret_raw["Retorno (%)"] = ret_raw["Retorno (%)"].round(2)
        ret_raw = ret_raw.merge(
            positions[["ticker","asset_class","sector","industry"]],
            left_on="Ticker", right_on="ticker", how="left"
        ).drop(columns=["ticker"])
        ret_raw = ret_raw.sort_values("Retorno (%)", ascending=False)
        st.dataframe(
            ret_raw.style.map(
                lambda v: "color: green" if isinstance(v, float) and v > 0
                          else ("color: red" if isinstance(v, float) and v < 0 else ""),
                subset=["Retorno (%)"]
            ).format({"Retorno (%)": "{:+.2f}%"}),
            use_container_width=True,
            hide_index=True
        )

else:
    st.warning("No se pudieron cargar datos históricos.")

st.divider()


# ── 5. P&L POR POSICIÓN (PLOTLY) ─────────────────────────────────────────────

st.subheader("P&L por posición")

pl_sorted  = positions.sort_values("pl_usd").copy()
pl_colors  = ["#3B6D11" if v >= 0 else "#A32D2D" for v in pl_sorted["pl_usd"]]

fig_pl = go.Figure(go.Bar(
    x           = pl_sorted["pl_usd"].round(2),
    y           = pl_sorted["ticker"],
    orientation = "h",
    marker      = dict(color=pl_colors),
    customdata  = pl_sorted[["pl_pct","market_value","cost_basis","sector","industry"]].values,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "P&L:          $%{x:+,.0f}<br>"
        "P&L %:        %{customdata[0]:+.1f}%<br>"
        "Valor actual: $%{customdata[1]:,.0f}<br>"
        "Costo base:   $%{customdata[2]:,.0f}<br>"
        "Sector:       %{customdata[3]}<br>"
        "Industry:     %{customdata[4]}<extra></extra>"
    )
))
fig_pl.add_vline(x=0, line_color="gray", line_dash="dash", opacity=0.4)
fig_pl.update_layout(
    height     = max(350, len(pl_sorted) * 30),
    xaxis      = dict(title="P&L (USD)"),
    yaxis      = dict(title=""),
    showlegend = False,
    margin     = dict(l=80, r=20, t=20, b=50),
)
st.plotly_chart(fig_pl, use_container_width=True)

st.divider()


# ── 6. EVOLUCIÓN DEL PATRIMONIO ───────────────────────────────────────────────

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

        # Mostrar clasificación automática si el ticker ya existe
        if ticker_new in CLASSIFICATION:
            known = CLASSIFICATION[ticker_new]
            st.success(
                f"✅ Clasificación automática: "
                f"{known['asset_class']} · {known['sector']} · {known['industry']}"
            )

        st.caption("Clasificación — elige de la lista o escribe una nueva:")

        ac_opts  = sorted({v["asset_class"] for v in CLASSIFICATION.values()})
        sec_opts = sorted({v["sector"]      for v in CLASSIFICATION.values()})
        ind_opts = sorted({v["industry"]    for v in CLASSIFICATION.values()})

        col1, col2 = st.columns(2)
        with col1:
            asset_class_sel    = st.selectbox("Asset Class", ac_opts)
            asset_class_custom = st.text_input("Nueva Asset Class", "",
                                               help="Rellena solo si quieres una nueva no listada")
        with col2:
            sector_sel    = st.selectbox("Sector", sec_opts)
            sector_custom = st.text_input("Nuevo Sector", "",
                                          help="Rellena solo si quieres un nuevo sector")

        industry_sel    = st.selectbox("Industry", ind_opts)
        industry_custom = st.text_input("Nueva Industry", "",
                                        help="Rellena solo si quieres una nueva industria")

        shares_new    = st.number_input("Acciones",         min_value=0.001, step=0.001, format="%.3f")
        avg_price_new = st.number_input("Precio medio ($)", min_value=0.01,  step=0.01)
        date_new      = st.date_input("Fecha de compra", value=date.today())
        submitted     = st.form_submit_button("Añadir")

        if submitted and ticker_new and shares_new > 0 and avg_price_new > 0:
            # Si el ticker está en el dict, usar clasificación automática
            # Si no, usar lo que el usuario eligió (custom override el selectbox)
            if ticker_new in CLASSIFICATION:
                ac  = CLASSIFICATION[ticker_new]["asset_class"]
                sec = CLASSIFICATION[ticker_new]["sector"]
                ind = CLASSIFICATION[ticker_new]["industry"]
            else:
                ac  = asset_class_custom.strip() or asset_class_sel
                sec = sector_custom.strip()       or sector_sel
                ind = industry_custom.strip()     or industry_sel

            new_row = pd.DataFrame([{
                "ticker":      ticker_new,
                "shares":      shares_new,
                "avg_price":   avg_price_new,
                "date_added":  date_new.strftime("%Y-%m-%d"),
                "category":    sec,   # compatibilidad CSV anterior
                "asset_class": ac,
                "sector":      sec,
                "industry":    ind,
            }])
            updated = pd.concat([load_positions(), new_row], ignore_index=True)
            save_positions(updated)
            st.success(f"✅ {ticker_new} añadido — {ac} · {sec} · {ind}")
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
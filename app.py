import streamlit as st

st.set_page_config(
    page_title = "Portfolio Tracker",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

pg = st.navigation({
    "📊 Mi Portfolio": [
        st.Page("pages/dashboard.py",
                title   = "Dashboard",
                icon    = "📈",
                default = True),
        st.Page("pages/simulacion_sr.py",
                title = "Simulación SR",
                icon  = "🧪"),
        st.Page("pages/analisis_tickers.py",
                title = "Análisis Tickers",
                icon  = "🔍"),
        st.Page("pages/historial_decisiones.py",
                title = "Historial Decisiones",
                icon  = "📓"),
        st.Page("pages/simulacion_sr2.py",
                title = "Simulación SR 2",
                icon  = "⚗️"),
    ]
})

pg.run()
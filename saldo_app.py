# ======================================
# IMPORT
# ======================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import hashlib
from io import BytesIO
from PIL import Image

# ======================================
# CONFIGURAZIONE PAGINA
# ======================================
logo = Image.open("Image/157214392_891863794931335_5614608524370432599_n.jpg")

st.set_page_config(
    page_title="📊 FinEdu Financial Analysis Tool",
    page_icon=logo,
    layout="wide"
)

st.image(logo, width=260)
st.markdown("<h1 style='text-align:center;'>📊 Risk Situation Room</h1>", unsafe_allow_html=True)

# ======================================
# LOGIN
# ======================================
st.sidebar.title("🔐 Login")

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

try:
    with open("users.json") as f:
        USERS = json.load(f)
except:
    USERS = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    u = st.sidebar.text_input("Username")
    p = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if USERS.get(u) == hash_password(p):
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.sidebar.error("Credenziali errate")
    st.stop()

# ======================================
# APP
# ======================================
st.title("💼 Monthly Financial Reporting & Budget Control")

# ======================================
# 1. GESTIONE SPESE
# ======================================
st.header("1️⃣ Gestione Spese ed Entrate")

MONTHS = ['gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic']
BASE_COLS = ['Tipo','Tipologia','Dettaglio'] + MONTHS

def empty_finance_df():
    return pd.DataFrame({
        'Tipo': ['Entrate'],
        'Tipologia': [''],
        'Dettaglio': [''],
        **{m: [0.0] for m in MONTHS}
    })

# ---- FORMAT EXCEL ----
with st.expander("📥 Template Excel"):
    df_template = pd.DataFrame(columns=BASE_COLS)
    buffer = BytesIO()
    df_template.to_excel(buffer, index=False)
    st.download_button("Scarica format Excel", buffer.getvalue(), "format.xlsx")

# ---- MODALITÀ ----
mode = st.radio(
    "Modalità di inserimento dati",
    ["📤 Carica Excel / CSV", "✍️ Inserimento manuale"],
    horizontal=True
)

df = None

# ---- UPLOAD ----
if mode == "📤 Carica Excel / CSV":
    file = st.file_uploader("Carica file", type=["csv", "xlsx"])
    if file:
        df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)

# ---- INSERIMENTO MANUALE (EXCEL-LIKE) ----
if mode == "✍️ Inserimento manuale" or df is not None:
    if df is None:
        if "finance_df" not in st.session_state:
            st.session_state.finance_df = empty_finance_df()
        df = st.session_state.finance_df

    for col in BASE_COLS:
        if col not in df.columns:
            df[col] = 0 if col in MONTHS else ""

    st.subheader("📋 Inserimento dati (stile Excel)")

    df = st.data_editor(
        df,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "Tipo": st.column_config.SelectboxColumn(
                "Tipo", options=["Entrate", "Uscite"]
            ),
            **{m: st.column_config.NumberColumn(m, format="€ %.2f") for m in MONTHS}
        }
    )

    st.session_state.finance_df = df

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Reset tabella"):
            st.session_state.finance_df = empty_finance_df()
            st.rerun()
    with c2:
        out = BytesIO()
        df[BASE_COLS].to_excel(out, index=False)
        st.download_button("💾 Scarica Excel", out.getvalue(), "dati_finanziari.xlsx")

# ======================================
# CALCOLI
# ======================================
if df is not None:

    for m in MONTHS:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0.0)

    df["Totale"] = df[MONTHS].sum(axis=1)

    entrate = df[df["Tipo"]=="Entrate"]["Totale"].sum()
    uscite = df[df["Tipo"]=="Uscite"]["Totale"].sum()
    saldo = entrate - uscite

    # ======================================
    # KPI ANNUALI
    # ======================================
    st.header("📈 KPI Annuali")

    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Entrate Totali", f"€{entrate:,.0f}")
    c2.metric("💸 Uscite Totali", f"€{uscite:,.0f}")
    c3.metric("📊 Saldo Annuale", f"€{saldo:,.0f}")

    # ---- GAUGE SALDO ----
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=saldo,
        title={'text': "Saldo Annuale"},
        gauge={
            'axis': {'range': [-uscite, entrate]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [-uscite, 0], 'color': "#ffcccc"},
                {'range': [0, entrate], 'color': "#ccffcc"}
            ],
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ======================================
    # KPI MENSILI
    # ======================================
    st.header("📅 KPI Mensili")

    rows = []
    saldo_cumulativo = 0

    for m in MONTHS:
        e = df[df["Tipo"]=="Entrate"][m].sum()
        u = df[df["Tipo"]=="Uscite"][m].sum()
        s = e - u
        saldo_cumulativo += s

        rows.append({
            "📅 Mese": m.capitalize(),
            "💰 Entrate": e,
            "💸 Uscite": u,
            "📈 Saldo": s,
            "📊 Cumulativo": saldo_cumulativo,
            "🚦": "🟢" if s > 1000 else "🟡" if s > 0 else "🔴"
        })

    kpi_mensili = pd.DataFrame(rows)

    st.dataframe(
        kpi_mensili,
        use_container_width=True,
        hide_index=True
    )

    # ---- GRAFICI ----
    st.header("📊 Analisi Grafica")

    fig_cum = px.line(
        kpi_mensili,
        x="📅 Mese",
        y="📊 Cumulativo",
        markers=True,
        title="Saldo Cumulativo"
    )

    fig_flow = px.bar(
        kpi_mensili,
        x="📅 Mese",
        y=["💰 Entrate","💸 Uscite"],
        barmode="group",
        title="Entrate vs Uscite"
    )

    st.plotly_chart(fig_cum, use_container_width=True)
    st.plotly_chart(fig_flow, use_container_width=True)

    # ======================================
    # ANALISI SPESE
    # ======================================
    st.header("🧠 Analisi Spese")

    spese_tipologia = (
        df[df["Tipo"]=="Uscite"]
        .groupby("Tipologia")["Totale"]
        .sum()
        .sort_values(ascending=False)
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📊 Spesa per Tipologia")
        st.plotly_chart(
            px.bar(
                spese_tipologia.reset_index(),
                x="Tipologia",
                y="Totale",
                text_auto=True
            ),
            use_container_width=True
        )

    with c2:
        st.subheader("🥧 Distribuzione Spese")
        st.plotly_chart(
            px.pie(
                spese_tipologia.reset_index(),
                names="Tipologia",
                values="Totale"
            ),
            use_container_width=True
        )
# ======================================
# 2. COSTRUZIONE PORTAFOGLIO
# ======================================
st.divider()
st.header("2. Simulazione Portafoglio")

tickers_dict = {
    "Azioni": ["MSFT","GOOGL","AMZN"],
    "Bond": ["BND","TLT","AGG"],
    "Crypto": ["BTC-USD","ETH-USD"],
    "ETF Azionari": ["VWCE.DE","SWDA.MI","EIMI.L","VUSA.MI"],
    "ETF Bond": ["VAGF.MI","IBGL.MI","EMBE.MI"],
    "Commodities": ["SGLD.MI","CRUD.MI","WEAT.L","COPA.L"]
}

asset_class = st.multiselect("Asset class", tickers_dict.keys())
manual = st.text_input("Ticker manuali (separati da virgola)")

tickers = list(set(
    [t.strip().upper() for t in manual.split(",") if t.strip()] +
    [t for a in asset_class for t in tickers_dict[a]]
))

if tickers:
    weights_raw = [st.slider(f"{t} (%)", 0, 100, 10) for t in tickers]

    if st.button("Costruisci portafoglio"):
        prices, valid = download_data_robust(tickers)

        if prices.empty:
            st.error("Errore download dati")
            st.stop()

        w_map = dict(zip(tickers, weights_raw))
        weights = np.array([w_map[t] for t in valid])
        weights = weights / weights.sum()

        returns = calculate_returns(prices)
        metrics = portfolio_metrics(weights, returns)

        st.session_state.update({
            "returns_df": returns,
            "weights": weights,
            "valid_tickers": valid,
            "metrics": metrics
        })

        st.plotly_chart(plot_cumulative_returns(weights, returns), use_container_width=True)
        st.plotly_chart(plot_drawdown(weights, returns), use_container_width=True)
        st.plotly_chart(plot_correlation_heatmap(metrics["Correlation Matrix"]), use_container_width=True)

# ======================================
# 3. MONTE CARLO
# ======================================
st.divider()
st.header("3. Proiezione Monte Carlo")

if "returns_df" in st.session_state:
    years = st.slider("Anni", 1, 30, 5)
    scen = st.slider("Scenari", 500, 10000, 2000, step=500)
    nu = st.slider("Gradi di libertà (ν)", 2, 30, 5)

    if st.button("Simula"):
        if "saldo_annuale" not in st.session_state:
            st.error("Inserisci prima i dati finanziari")
            st.stop()

        initial = st.session_state.saldo_annuale
        returns = st.session_state.returns_df

        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)

        draws = simulate_t_copula(
            mu, sigma, returns.corr().values,
            years, scen, nu
        )

        port_ret = np.tensordot(draws, st.session_state.weights, axes=([2],[0]))

        values = np.zeros((scen, years+1))
        values[:,0] = initial

        for t in range(1, years+1):
            values[:,t] = (values[:,t-1] + initial) * (1 + port_ret[:,t-1])

        p5, p50, p95 = np.percentile(values[:,1:], [5,50,95], axis=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=range(1,years+1), y=p50, name="Mediana"))
        fig.add_trace(go.Scatter(
            x=list(range(1,years+1))+list(range(years,0,-1)),
            y=list(p95)+list(p5[::-1]),
            fill="toself", opacity=0.2, name="Banda 5–95%"
        ))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Costruisci prima il portafoglio")








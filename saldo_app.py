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

from utils.portfolio_utils import (
    download_data_robust, calculate_returns, portfolio_metrics,
    simulate_t_copula, create_excel_report_investimento,
    plot_cumulative_returns, plot_return_distribution, plot_weights,
    plot_drawdown, plot_correlation_heatmap,
    plot_risk_contribution, plot_contribution, plot_efficient_frontier
)

# ======================================
# CONFIGURAZIONE PAGINA
# ======================================
logo = Image.open("Image/157214392_891863794931335_5614608524370432599_n.jpg")
st.set_page_config(
    page_title="📊 FinEdu Financial Analysis Tool",
    page_icon=logo,
    layout="wide"
)

st.image(logo, width=280)
st.markdown(
    "<h1 style='text-align:center;'>Risk Situation Room</h1>",
    unsafe_allow_html=True
)

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
            st.session_state.username = u
            st.rerun()
        else:
            st.sidebar.error("Credenziali errate")
    st.stop()

# ======================================
# APP
# ======================================
st.title("📊 Monthly Financial Reporting & Portfolio Management")

# ======================================
# 1. GESTIONE SPESE
# ======================================
st.header("1. Gestione Spese ed Entrate")

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
col1, col2 = st.columns(2)
with col1:
    df_template = pd.DataFrame(columns=BASE_COLS)
    buffer = BytesIO()
    df_template.to_excel(buffer, index=False)
    st.download_button("📥 Scarica format Excel", buffer.getvalue(), "format.xlsx")

# ---- SCELTA MODALITÀ ----
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

# ---- INSERIMENTO MANUALE ----
if mode == "✍️ Inserimento manuale":
    if "finance_df" not in st.session_state:
        st.session_state.finance_df = empty_finance_df()

    df = st.data_editor(
        st.session_state.finance_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Tipo": st.column_config.SelectboxColumn(
                "Tipo", options=["Entrate", "Uscite"]
            )
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

# ---- CALCOLI ----
if df is not None:
    for col in BASE_COLS:
        if col not in df.columns:
            df[col] = 0 if col in MONTHS else ""

    for m in MONTHS:
        df[m] = (
            df[m].astype(str)
            .str.replace(r"[€,]", "", regex=True)
            .astype(float)
            .fillna(0.0)
        )

    df["Totale"] = df[MONTHS].sum(axis=1)

    entrate = df[df["Tipo"] == "Entrate"]["Totale"].sum()
    uscite = df[df["Tipo"] == "Uscite"]["Totale"].sum()
    st.session_state["saldo_annuale"] = entrate - uscite

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.bar(df.groupby("Tipo")[MONTHS].sum().T, title="Flussi mensili"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            px.pie(
                df.groupby("Tipologia")["Totale"].sum().reset_index(),
                names="Tipologia",
                values="Totale",
                title="Distribuzione spese"
            ),
            use_container_width=True
        )

    st.metric("Saldo Annuale", f"€{st.session_state['saldo_annuale']:,.2f}")

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

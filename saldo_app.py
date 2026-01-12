# ======================================
# IMPORT
# ======================================
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
import hashlib
from io import BytesIO
from PIL import Image
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

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
st.markdown("<h1 style='text-align:center;'>Risk Situation Room</h1>", unsafe_allow_html=True)

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

# ---- INSERIMENTO MANUALE CON AGGRID ----
if mode == "✍️ Inserimento manuale" or df is not None:
    if df is None:
        if "finance_df" not in st.session_state:
            st.session_state.finance_df = empty_finance_df()
        df = st.session_state.finance_df

    # Assicuriamoci che tutte le colonne ci siano
    for col in BASE_COLS:
        if col not in df.columns:
            df[col] = 0 if col in MONTHS else ""

    # Configurazione AG Grid
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True)
    gb.configure_column("Tipo", cellEditor='agSelectCellEditor', cellEditorParams={'values': ['Entrate','Uscite']})
    grid_options = gb.build()

    st.subheader("📋 Inserimento / Modifica dati")
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=True,
        height=400,
        allow_unsafe_jscode=True
    )

    df = grid_response['data']
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

# ---- PULIZIA DATI E CALCOLI ----
if df is not None:
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
    saldo = entrate - uscite
    st.session_state["saldo_annuale"] = saldo

    # ---- KPI ANNUALI ----
    st.subheader("📈 KPI Annuali")
    c1, c2, c3 = st.columns(3)
    c1.metric("Saldo Annuale", f"€{saldo:,.2f}")
    c2.metric("Entrate Totali", f"€{entrate:,.2f}")
    c3.metric("Uscite Totali", f"€{uscite:,.2f}")
    perc_risparmio = (saldo / entrate * 100) if entrate > 0 else 0
    st.metric("Percentuale Risparmio", f"{perc_risparmio:.2f}%")
    mese_piu_costoso = df[MONTHS].sum().idxmax()
    spesa_massima = df[MONTHS].sum().max()
    st.metric("Mese più Costoso", mese_piu_costoso, f"€{spesa_massima:,.2f}")

    # ---- KPI MENSILI E SALDO CUMULATIVO ----
    st.subheader("📊 KPI Mensili e Saldo Cumulativo")
    saldo_cumulativo = []
    saldo_temp = 0
    mesi_data = []
    entrate_mensili = []
    uscite_mensili = []

    for m in MONTHS:
        entrate_m = df[df["Tipo"]=="Entrate"][m].sum()
        uscite_m = df[df["Tipo"]=="Uscite"][m].sum()
        saldo_m = entrate_m - uscite_m
        saldo_temp += saldo_m
        saldo_cumulativo.append(saldo_temp)
        mesi_data.append(m)
        entrate_mensili.append(entrate_m)
        uscite_mensili.append(uscite_m)
        st.write(f"**{m.capitalize()}**: Entrate €{entrate_m:,.2f}, Uscite €{uscite_m:,.2f}, Saldo €{saldo_m:,.2f}")

    fig_line = px.line(x=mesi_data, y=saldo_cumulativo, title="Saldo Cumulativo Mensile", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

    fig_bar = px.bar(pd.DataFrame({"Mese": mesi_data, "Entrate": entrate_mensili, "Uscite": uscite_mensili}),
                     x="Mese", y=["Entrate","Uscite"], barmode="group", title="Entrate e Uscite Mensili")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ---- OBIETTIVO RISPARMIO ----
    st.subheader("🎯 Obiettivo di Risparmio")
    obiettivo_annuale = st.number_input("Imposta obiettivo risparmio annuale (€)", value=5000)
    progresso = min(max(int((saldo / obiettivo_annuale) * 100),0),100)
    st.progress(progresso)
    st.write(f"Percentuale obiettivo raggiunta: {progresso}%")

    # ---- KPI PER TIPOLOGIE DI SPESA ----
    st.subheader("📊 KPI per Tipologia")
    spesa_per_tipologia = df[df["Tipo"]=="Uscite"].groupby("Tipologia")["Totale"].sum().sort_values(ascending=False)
    st.dataframe(spesa_per_tipologia)

    st.subheader("📊 Percentuale sul totale per Tipologia")
    percentuali = (spesa_per_tipologia / spesa_per_tipologia.sum() * 100).round(2)
    st.dataframe(percentuali)

    st.subheader("💡 Top 3 Tipologie di Spesa")
    top3 = spesa_per_tipologia.head(3)
    for tip, val in top3.items():
        st.write(f"{tip}: €{val:,.2f}")

    st.subheader("📊 Grafico Spesa Totale per Tipologia")
    fig_tipologia = px.bar(spesa_per_tipologia.reset_index(), x="Tipologia", y="Totale", text_auto=True,
                           title="Spesa Totale per Tipologia")
    st.plotly_chart(fig_tipologia, use_container_width=True)

    st.subheader("📊 Spesa Mensile per Tipologia")
    df_mensile = df[df["Tipo"]=="Uscite"].groupby("Tipologia")[MONTHS].sum().T
    fig_mensile = px.bar(df_mensile, x=df_mensile.index, y=df_mensile.columns, barmode="stack", title="Spesa Mensile per Tipologia")
    st.plotly_chart(fig_mensile, use_container_width=True)

    # ---- GRAFICI GENERALI ----
    st.subheader("📊 Distribuzione Tipologie e Flussi")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(df.groupby("Tipo")[MONTHS].sum().T, title="Flussi Mensili"), use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(df.groupby("Tipologia")["Totale"].sum().reset_index(),
                               names="Tipologia", values="Totale", title="Distribuzione Spese"), use_container_width=True)
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



# ======================================
# IMPORT
# ======================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import hashlib
from io import BytesIO
from PIL import Image
from fpdf import FPDF
import plotly.io as pio
import tempfile
import base64
import textwrap
from scipy.stats import t as student_t
from scipy.stats import norm
import numpy as np
import yfinance as yf

from utils.portfolio_utils import (
    download_data_robust,
    portfolio_metrics,
    cumulative_portfolio_returns,
    plot_correlation_heatmap,
    plot_cumulative_returns,
    simulate_investment,
    plot_return_distribution,
    plot_rolling_volatility,
    plot_drawdown,
    plot_risk_contribution,
    plot_weights,
    plot_efficient_frontier,
    plot_contribution,
    simulate_t_copula,
    create_excel_report_investimento,
    calculate_returns
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

st.image(logo, width=260)
st.markdown("<h1 style='text-align:center;'>📊 FinEdu Financial Analysis Tool</h1>", unsafe_allow_html=True)

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

with st.expander("📥 Template Excel"):
    df_template = pd.DataFrame(columns=BASE_COLS)
    buffer = BytesIO()
    df_template.to_excel(buffer, index=False)
    st.download_button("Scarica format Excel", buffer.getvalue(), "format.xlsx")

mode = st.radio(
    "Modalità di inserimento dati",
    ["📤 Carica Excel / CSV", "✍️ Inserimento manuale"],
    horizontal=True
)

df = None

if mode == "📤 Carica Excel / CSV":
    file = st.file_uploader("Carica file", type=["csv", "xlsx"])
    if file:
        df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)

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
            "Tipo": st.column_config.SelectboxColumn("Tipo", options=["Entrate", "Uscite"]),
            **{m: st.column_config.NumberColumn(m, format="€ %.2f") for m in MONTHS}
        }
    )

    st.session_state.finance_df = df

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
    st.session_state.saldo_annuale = saldo
    # ======================================
    # KPI ANNUALI
    # ======================================
    st.header("📈 KPI Annuali")
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Entrate Totali", f"€{entrate:,.0f}")
    c2.metric("💸 Uscite Totali", f"€{uscite:,.0f}")
    c3.metric("📊 Saldo Annuale", f"€{saldo:,.0f}")

    # ======================================
    # KPI MENSILI
    # ======================================
    st.header("📅 KPI Mensili")

    rows, saldo_cum = [], 0
    for m in MONTHS:
        e = df[df["Tipo"]=="Entrate"][m].sum()
        u = df[df["Tipo"]=="Uscite"][m].sum()
        s = e - u
        saldo_cum += s
        rows.append({
            "Mese": m.capitalize(),
            "Entrate": e,
            "Uscite": u,
            "Saldo": s,
            "Cumulativo": saldo_cum
        })

    kpi_mensili = pd.DataFrame(rows)
    st.dataframe(kpi_mensili, use_container_width=True, hide_index=True)

    st.plotly_chart(
        px.line(kpi_mensili, x="Mese", y="Cumulativo", markers=True, title="Saldo cumulativo"),
        use_container_width=True
    )

    # ======================================
    # 🎯 OBIETTIVI ANNUALI + MENSILI
    # ======================================
    st.header("🎯 Obiettivi di Risparmio")

    col1, col2, col3 = st.columns(3)
    with col1:
        emer = st.number_input("🛟 Fondo Emergenza (€)", 0, 100000, 6000)
    with col2:
        vac = st.number_input("✈️ Vacanze (€)", 0, 50000, 3000)
    with col3:
        casa = st.number_input("🏡 Anticipo Casa (€)", 0, 500000, 20000)

    targets = {
        "Fondo Emergenza": emer,
        "Vacanze": vac,
        "Anticipo Casa": casa
    }

    st.subheader("📆 Obiettivi Mensili")

    risparmio_medio = saldo / 12 if saldo > 0 else 0

    for nome, target in targets.items():
        target_m = target / 12
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risparmio_medio,
            title={'text': f"{nome} – target mensile €{target_m:,.0f}"},
            gauge={
                'axis': {'range': [0, target_m]},
                'bar': {'color': "#3498db"},
                'steps': [{'range': [0, target_m], 'color': "#ecf0f1"}]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)


    # ======================================
    # 🎯 OBIETTIVI DI RISPARMIO
    # ======================================
    st.header("🎯 Obiettivi di Risparmio")

    st.markdown("Imposta i tuoi obiettivi finanziari e monitora il progresso")

    goals = {
        "🛟 Fondo Emergenza": st.number_input("Target Fondo Emergenza (€)", 0, 200000, 10000, step=1000),
        "✈️ Vacanze": st.number_input("Target Vacanze (€)", 0, 50000, 3000, step=500),
        "🏠 Anticipo Casa": st.number_input("Target Anticipo Casa (€)", 0, 500000, 30000, step=5000),
    }

    allocazione_annua = saldo if saldo > 0 else 0

    goal_rows = []

    for nome, target in goals.items():
        progress = min(allocazione_annua / target * 100, 100) if target > 0 else 0

        if progress >= 75:
            status = "🟢 In linea"
        elif progress >= 40:
            status = "🟡 Rallentato"
        else:
            status = "🔴 Critico"

        goal_rows.append({
            "🎯 Obiettivo": nome,
            "🎯 Target €": target,
            "💰 Allocato €": allocazione_annua,
            "📈 Progresso %": progress,
            "🚦 Stato": status
        })

    goals_df = pd.DataFrame(goal_rows)

    st.dataframe(
        goals_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "📈 Progresso %": st.column_config.ProgressColumn(
                "Progresso",
                min_value=0,
                max_value=100,
                format="%.0f%%"
            )
        }
    )

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
        st.plotly_chart(px.bar(spese_tipologia.reset_index(), x="Tipologia", y="Totale", text_auto=True),
                         use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(spese_tipologia.reset_index(), names="Tipologia", values="Totale"),
                         use_container_width=True)
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
        st.plotly_chart(plot_rolling_volatility(weights, returns, window=21), use_container_width=True)
        st.plotly_chart(plot_drawdown(weights, returns), use_container_width=True)
        st.plotly_chart(plot_correlation_heatmap(metrics["Correlation Matrix"]), use_container_width=True)
        st.plotly_chart(plot_contribution(weights, returns), use_container_width=True)
        st.plotly_chart(plot_efficient_frontier(returns, n_portfolios=5000, risk_free=0.02), use_container_width=True)
        st.plotly_chart(plot_weights(weights, tickers), use_container_width=True)
        st.plotly_chart(plot_risk_contribution(weights, returns), use_container_width=True)

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
    
            # Valori iniziali e rendimenti
            initial = st.session_state.saldo_annuale
            returns = st.session_state.returns_df
    
            # Media e deviazione standard annualizzate
            mu = returns.mean() * 252
            sigma = returns.std() * np.sqrt(252)
    
            # Simulazione Monte Carlo con t-copula
            draws = simulate_t_copula(mu, sigma, returns.corr().values, years, scen, nu)
    
            # Rendimento del portafoglio
            port_ret = np.tensordot(draws, st.session_state.weights, axes=([2],[0]))
    
            # Matrice valori simulati
            values = np.zeros((scen, years + 1))
            values[:, 0] = initial
    
            for t in range(1, years + 1):
                values[:, t] = (values[:, t - 1] + initial) * (1 + port_ret[:, t - 1])
    
            # Percentili per bande di confidenza
            p5, p50, p95 = np.percentile(values[:, 1:], [5, 50, 95], axis=0)
    
            # Asse X basato sulla lunghezza dei dati
            x_axis = list(range(1, len(p50) + 1))
    
            # Grafico Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_axis, y=p50, name="Mediana", line=dict(color="#1f77b4", width=2)))
            fig.add_trace(go.Scatter(
                x=x_axis + x_axis[::-1],
                y=list(p95) + list(p5[::-1]),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Banda 5–95%"
            ))
    
            fig.update_layout(
                title="Proiezione Monte Carlo del Portafoglio",
                xaxis_title="Anno",
                yaxis_title="Valore (€)",
                template="plotly_white"
            )
    
            st.plotly_chart(fig, use_container_width=True)
    
            # -----------------------------
            # Decomposizione Capitale vs Rendimento
            # -----------------------------
            # --- DECOMPOSIZIONE CAPITALE VS RENDIMENTO ---
            years_x = np.arange(1, years + 1)  # Aggiungi questa riga
            capitale = np.array([initial * t for t in years_x])
            rendimento_mediano = p50 - capitale
    
            fig_stack = go.Figure()
            fig_stack.add_trace(go.Bar(x=years_x, y=capitale, name="Capitale Investito", marker_color="royalblue"))
            fig_stack.add_trace(go.Bar(x=years_x, y=rendimento_mediano, name="Rendimento (mediano)", marker_color="seagreen"))
            fig_stack.update_layout(
                barmode="stack",
                title="Decomposizione Mediana: Capitale + Rendimento",
                xaxis_title="Anno",
                yaxis_title="Valore (€)",
                template="plotly_white"
            )
            st.plotly_chart(fig_stack, use_container_width=True)
    
            # -----------------------------
            # Statistiche Finali
            # -----------------------------
            final_vals = values[:, -1]
            st.subheader("📊 Statistiche scenari finali")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Media", f"€{np.mean(final_vals):,.2f}")
            c2.metric("Mediana", f"€{np.median(final_vals):,.2f}")
            c3.metric("Minimo", f"€{np.min(final_vals):,.2f}")
            c4.metric("Massimo", f"€{np.max(final_vals):,.2f}")
            st.write(f"**Deviazione Standard:** €{np.std(final_vals):,.2f}")
    
            # Salvataggio dati per l'Excel
            st.session_state["df_pct"] = pd.DataFrame({
                    'Anno': years_x,
                    'Capitale': capitale,
                    'Totale_P5': p5,
                    'Totale_P50': p50,
                    'Totale_P95': p95
                })
                
                # Bottone Excel
            ex_bytes = create_excel_report_investimento(
                    saldo_annuale=initial, 
                    metrics=st.session_state.metrics,
                    df_pct=st.session_state.df_pct, 
                    returns_df=returns,
                    weights=st.session_state.weights, 
                    selected_tickers=st.session_state.valid_tickers
                )
            st.download_button("💾 Scarica Report Excel", data=ex_bytes, file_name="Report_Investimento.xlsx")
    
    else:
        st.info("Costruisci prima il portafoglio")






















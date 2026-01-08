import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import hashlib
from io import BytesIO

# Import dalle tue utility
from utils.portfolio_utils import (
    download_data_robust, calculate_returns, portfolio_metrics, 
    simulate_t_copula, create_excel_report_investimento,
    plot_cumulative_returns, plot_return_distribution, plot_weights, 
    plot_drawdown, plot_rolling_volatility, plot_correlation_heatmap,
    plot_risk_contribution, plot_contribution, plot_efficient_frontier
)

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Report Finanziario", layout="wide")

# --- LOGIN ---
st.sidebar.title("🔐 Login")
try:
    with open("users.json") as f:
        users = json.load(f)
except:
    users = {} # Gestione errore se manca il file

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

if not st.session_state.logged_in:
    u_in = st.sidebar.text_input("Username")
    p_in = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if users.get(u_in) == hash_password(p_in):
            st.session_state.logged_in = True
            st.session_state.username = u_in
            st.rerun()
        else:
            st.sidebar.error("Credenziali errate")
    st.stop()

# --- APP CONTENT ---
st.title("📊 Report Finanziario Mensile & Portfolio Manager")

# 1. GESTIONE SPESE (Identico al tuo)
st.header("1. Gestione Spese ed Entrate")
col_format1, col_format2 = st.columns(2)
with col_format1:
    df_f = pd.DataFrame(columns=['Tipo','Tipologia','Dettaglio','gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic'])
    excel_b = BytesIO(); df_f.to_excel(excel_b, index=False)
    st.download_button("📥 Scarica Format Excel", data=excel_b.getvalue(), file_name="format.xlsx")

uploaded_file = st.file_uploader("Carica Excel/CSV", type=["csv", "xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    months = ['gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic']
    for m in months:
        df[m] = pd.to_numeric(df[m].replace('[\€,]', '', regex=True), errors='coerce').fillna(0)
    
    df['Totale'] = df[months].sum(axis=1)
    entrate = df[df['Tipo']=='Entrate']['Totale'].sum()
    uscite = df[df['Tipo']=='Uscite']['Totale'].sum()
    st.session_state["saldo_annuale"] = entrate - uscite

    # Grafici Spese
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(df.groupby('Tipo')[months].sum().T, title="Flussi Mensili"), use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(df.groupby('Tipologia')['Totale'].sum().reset_index(), names='Tipologia', values='Totale', title="Distribuzione"), use_container_width=True)
    st.metric("Saldo Annuale", f"€{st.session_state['saldo_annuale']:,.2f}")

# 2. COSTRUZIONE PORTAFOGLIO
st.divider()
st.header("2. Simulazione Portafoglio")

tickers_dict = {
    "Azioni": ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META"],
    "Bond": ["BND","TLT","AGG"],
    "Crypto": ["BTC-USD","ETH-USD"],
    "ETF azionari": ["VWCE.DE", "SWDA.MI", "EIMI.L", "VUSA.MI"],
    "ETF bond": ["VAGF.MI", "IBGL.MI", "EMBE.MI"],
    "ETC commodities": ["SGLD.MI", "CRUD.MI", "WEAT.L", "COPA.L"]
}

asset_class = st.multiselect("Seleziona Asset Class", list(tickers_dict.keys()))
t_manual = st.text_input("Inserisci altri ticker (separati da virgola)")
all_t = list(set([t.strip().upper() for t in t_manual.split(",") if t.strip()] + [t for a in asset_class for t in tickers_dict[a]]))

if all_t:
    st.subheader("Assegna pesi (%)")
    w_input = [st.slider(f"{t} (%)", 0, 100, 10, key=f"sl_{t}") for t in all_t]

    if st.button("Costruisci Portafoglio"):
        with st.spinner("Download dati..."):
            df_prezzi, valid_t = download_data_robust(all_t)
        
        if not df_prezzi.empty:
            # Allineamento pesi
            w_map = dict(zip(all_t, w_input))
            v_weights = np.array([w_map[t] for t in valid_t])
            final_w = v_weights / v_weights.sum() if v_weights.sum() > 0 else v_weights
            
            returns_df = calculate_returns(df_prezzi)
            metrics = portfolio_metrics(final_w, returns_df)
            
            # Salvataggio Session State
            st.session_state.update({"returns_df": returns_df, "weights": final_w, "valid_tickers": valid_t, "metrics": metrics})
            
            # --- OUTPUT GRAFICI (Tutti quelli richiesti) ---
            st.subheader("Metriche e Analisi")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Rendimento Annuo", f"{metrics['Rendimento atteso annuo']:.2%}")
            col_m2.metric("Volatilità Annua", f"{metrics['Volatilità annua']:.2%}")
            col_m3.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0):.2%}")

            # Visualizzazione di TUTTI i grafici del tuo file originale
            st.plotly_chart(plot_cumulative_returns(final_w, returns_df), use_container_width=True)
            
            c_g1, c_g2 = st.columns(2)
            with c_g1: st.plotly_chart(plot_return_distribution(final_w, returns_df))
            with c_g2: st.plotly_chart(plot_weights(final_w, valid_t))
            
            st.plotly_chart(plot_drawdown(final_w, returns_df), use_container_width=True)
            st.plotly_chart(plot_correlation_heatmap(metrics["Correlation Matrix"]), use_container_width=True)
            
            c_g3, c_g4 = st.columns(2)
            with c_g3: st.plotly_chart(plot_risk_contribution(final_w, returns_df))
            with c_g4: st.plotly_chart(plot_efficient_frontier(returns_df))
            
            st.plotly_chart(plot_contribution(final_w, returns_df), use_container_width=True)

# =====================
# 3. PROIEZIONE MONTE CARLO
# =====================
st.divider()
st.header("3. Proiezione Monte Carlo")

if "returns_df" in st.session_state:
    y_inv = st.slider("Anni di investimento", 1, 30, 5)
    
    if st.button("Simula Investimento"):
        if "saldo_annuale" not in st.session_state:
            st.error("⚠️ Carica prima il file finanziario per usare il saldo come base!")
        else:
            initial = float(st.session_state.saldo_annuale)
            n_scen = 2000
            nu_val = 5
            years_x = list(range(1, y_inv + 1))
            
            # --- Calcolo Simulazione ---
            returns_df = st.session_state.returns_df
            mu_ann = returns_df.mean() * 252
            sigma_ann = returns_df.std() * np.sqrt(252)
            
            draws = simulate_t_copula(mu_ann, sigma_ann, returns_df.corr().values, y_inv, n_scen, nu_val)
            port_ret = np.tensordot(draws, st.session_state.weights, axes=([2], [0]))
            
            values = np.zeros((n_scen, y_inv + 1))
            values[:, 0] = initial
            for t in range(1, y_inv + 1):
                # Investimento del saldo ogni anno + rendimento
                values[:, t] = (values[:, t-1] + initial) * (1.0 + port_ret[:, t-1])
            
            # --- Percentili ---
            p5, p50, p95 = np.percentile(values[:, 1:], [5, 50, 95], axis=0)

            # 1. Grafico Proiezione con Banda
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(x=years_x, y=p50, name="Mediana", line=dict(color='blue')))
            fig_sim.add_trace(go.Scatter(
                x=years_x + years_x[::-1],
                y=list(p95) + list(p5[::-1]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Banda 5°-95°'
            ))
            fig_sim.update_layout(title="Andamento Valore Accumulato", template="plotly_white")
            st.plotly_chart(fig_sim, use_container_width=True)

            # --- DECOMPOSIZIONE CAPITALE VS RENDIMENTO ---
            capitale = np.array([initial * t for t in years_x])
            rendimento_mediano = p50 - capitale

            # 2. Grafico Stacked Bar
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
            

            # --- STATISTICHE FINALI ---
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
                returns_df=returns_df,
                weights=st.session_state.weights, 
                selected_tickers=st.session_state.valid_tickers
            )
            st.download_button("💾 Scarica Report Excel", data=ex_bytes, file_name="Report_Investimento.xlsx")

else:
    st.info("Configura il portafoglio nella sezione precedente per abilitare la simulazione.")ss


import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import plotly.graph_objects as go
import numpy as np   
from utils.portfolio_utils import download_data, calculate_returns, portfolio_metrics, simulate_investment
from utils.portfolio_utils import plot_cumulative_returns, plot_return_distribution, plot_weights, plot_drawdown, plot_rolling_volatility, plot_correlation_heatmap,plot_risk_contribution, plot_contribution, plot_efficient_frontier

st.markdown("""
# 📊 Report Finanziario Mensile
_Breve riepilogo di entrate, uscite e saldo mensile_
""")

# Pulsante per scaricare il format Excel/CSV
st.header("Scarica il format Excel/CSV")
columns = ['Tipo','Tipologia','Dettaglio','gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic']
df_format = pd.DataFrame(columns=columns)

# CSV
csv_buffer = df_format.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Scarica CSV di esempio",
    data=csv_buffer,
    file_name="format_finanziario.csv",
    mime="text/csv"
)

# Excel
excel_buffer = BytesIO()
df_format.to_excel(excel_buffer, index=False)
st.download_button(
    label="Scarica Excel di esempio",
    data=excel_buffer,
    file_name="format_finanziario.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Upload file Excel/CSV
uploaded_file = st.file_uploader("Carica il file Excel/CSV", type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            # Proviamo prima con UTF-8, altrimenti fallback a latin-1
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {e}")
        st.stop()

    
    # Definizione mesi
    months = ['gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic']
    
    # Pulisci valori (€ -> float)
    for month in months:
        df[month] = df[month].replace('[\€,]', '', regex=True).astype(float)
    
    # Totale per riga
    df['Totale'] = df[months].sum(axis=1)
    
    # Entrate vs Uscite
    st.header("Entrate vs Uscite")
    entrate = df[df['Tipo']=='Entrate']['Totale'].sum()
    uscite = df[df['Tipo']=='Uscite']['Totale'].sum()
    st.write(f"Entrate totali: €{entrate:,.2f}")
    st.write(f"Uscite totali: €{uscite:,.2f}")
    
    # Distribuzione mensile
    st.header("Distribuzione Mensile")
    monthly_sum = df.groupby('Tipo')[months].sum().T
    fig1 = px.bar(monthly_sum, x=monthly_sum.index, y=['Entrate','Uscite'], title="Entrate e Uscite per mese")
    st.plotly_chart(fig1)
    
    # Distribuzione per tipologia
    st.header("Distribuzione per Tipologia")
    category_sum = df.groupby('Tipologia')['Totale'].sum().reset_index()
    fig2 = px.pie(category_sum, names='Tipologia', values='Totale', title="Distribuzione Uscite/Entrate per Tipologia")
    st.plotly_chart(fig2)
    
    # Tabella riepilogativa
    st.header("Tabella riepilogativa")
    st.dataframe(df)
    
    # Saldo mensile
    st.header("Saldo Mensile")
    entrate_mensili = df[df['Tipo']=='Entrate'][months].sum()
    uscite_mensili = df[df['Tipo']=='Uscite'][months].sum()
    saldo_mensile = entrate_mensili - uscite_mensili
    st.write(saldo_mensile.to_frame(name='Saldo Mensile (€)'))
    
    # Istogramma saldo mensile
    fig_saldo = go.Figure()
    fig_saldo.add_trace(go.Bar(x=months, y=saldo_mensile, name="Saldo"))
    fig_saldo.update_layout(
        title="Saldo Mensile (Entrate - Uscite)",
        xaxis_title="Mese",
        yaxis_title="Saldo (€)",
        template="plotly_white"
    )
    st.plotly_chart(fig_saldo)
    
    # Saldo annuale
    st.header("Saldo Annuale")
    saldo_annuale = saldo_mensile.sum()
    st.session_state["saldo_annuale"] = saldo_annuale
    st.write(f"Saldo annuale: €{saldo_annuale:,.2f}")


# =====================
# Simulazione Portafoglio con bottone
# =====================
st.header("Simulazione Portafoglio Investimenti")

# Dizionario ticker per asset class
tickers_dict = {
    "Azioni": ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META"],
    "Bond": ["BND","TLT","AGG"],
    "Crypto": ["BTC-USD","ETH-USD","BNB-USD","ADA-USD","SOL-USD"]
}

# Selezione asset class
asset_class = st.multiselect("Seleziona Asset Class", ["Azioni","Bond","Crypto"])

# Input manuale ticker
tickers_input = st.text_input("Inserisci i ticker separati da virgola", "")
manual_tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Ticker da selezione asset class
selected_from_class = []
for asset in asset_class:
    selected_from_class += tickers_dict[asset]

# Unione ticker e rimozione duplicati
selected_tickers = list(set(manual_tickers + selected_from_class))

if selected_tickers:
    st.subheader("Assegna peso ai titoli")
    weights = []
    for ticker in selected_tickers:
        w = st.slider(f"{ticker} (%)", 0, 100, 10)
        weights.append(w / 100)

    # Bottone per costruire portafoglio
    if st.button("Costruisci Portafoglio"):
        weights = np.array(weights)
        if weights.sum() == 0:
            st.error("I pesi non possono essere tutti 0")
        else:
            weights = weights / weights.sum()

            # Download dati storici e calcolo rendimenti
            data = download_data(selected_tickers)
            returns_df = calculate_returns(data)

            # Metriche portafoglio
            metrics = portfolio_metrics(weights, returns_df)
            st.subheader("Metriche Portafoglio")
            for key, value in metrics.items():
                if key == "Correlation Matrix":
                    continue
                if "Ratio" in key or key == "Max Drawdown":
                    st.write(f"{key}: {value:.2f}")
                else:
                    st.write(f"{key}: {value:.2%}")

            # Grafici Portafoglio
            st.subheader("Grafici Portafoglio")
            st.plotly_chart(plot_cumulative_returns(weights, returns_df))
            st.plotly_chart(plot_return_distribution(weights, returns_df))
            st.plotly_chart(plot_weights(weights, selected_tickers))
            st.plotly_chart(plot_drawdown(weights, returns_df))
            st.plotly_chart(plot_rolling_volatility(weights, returns_df))
            st.plotly_chart(plot_correlation_heatmap(metrics["Correlation Matrix"]))
            st.plotly_chart(plot_risk_contribution(weights, returns_df))
            st.plotly_chart(plot_efficient_frontier(returns_df))
            st.plotly_chart(plot_contribution(weights, returns_df))
# =====================
# Simulazione crescita saldo investito
# =====================
st.subheader("Simulazione Investimento")

years = st.slider("Anni di investimento", 1, 30, 5)

if st.button("Simula Investimento"):
    saldo_annuale = st.session_state["saldo_annuale"]
    if 'saldo_annuale' not in st.session_state:
        st.error("⚠️ Carica prima il file finanziario per calcolare il saldo annuale.")
    else:
        # Capitalizzazione composta
        initial = st.session_state.saldo_annuale
        # Uso rendimento atteso dal portafoglio se disponibile, altrimenti default 5%
        r = 0.05
        if "metrics" in locals() and "Rendimento atteso annuo" in metrics:
            r = metrics["Rendimento atteso annuo"]

        valori = [initial * (((1 + r) ** t - 1) / r) for t in range(1, years + 1)]
        investito = [initial * t for t in range(1, years + 1)]
        rendimento = [v - i for v, i in zip(valori, investito)]


        final_value = valori[-1]

        st.write(f"💰 Se investi il saldo annuale (€{initial:,.2f}) per {years} anni, "
                 f"il valore finale stimato sarà: €{final_value:,.2f}")

        # Grafico istogramma stacked
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, years + 1)),
            y=investito,
            name="Capitale Investito",
            marker_color="royalblue"
        ))
        fig.add_trace(go.Bar(
            x=list(range(1, years + 1)),
            y=rendimento,
            name="Rendimento Maturato",
            marker_color="seagreen"
        ))

        fig.update_layout(
            barmode="stack",
            title="Crescita del portafoglio (Capitale Investito + Rendimento)",
            xaxis_title="Anno",
            yaxis_title="Valore (€)",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)









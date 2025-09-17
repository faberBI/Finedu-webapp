import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import plotly.graph_objects as go

from utils.portfolio_utils import download_data, calculate_returns, portfolio_metrics, simulate_investment
from utils.portfolio_utils import plot_cumulative_returns, plot_return_distribution, plot_weights, plot_drawdown, plot_rolling_volatility, plot_correlation_heatmap,plot_risk_contribution

st.title("Report Finanziario Mensile")

# Pulsante per scaricare il format Excel/CSV
st.header("Scarica il format Excel/CSV")
columns = ['Tipo','Tipologia','Dettaglio','gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic']
# Creiamo un DataFrame vuoto con le colonne
df_format = pd.DataFrame(columns=columns)

# Convertiamo in CSV
csv_buffer = df_format.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Scarica CSV di esempio",
    data=csv_buffer,
    file_name="format_finanziario.csv",
    mime="text/csv"
)

# Convertiamo in Excel
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
    # Leggi Excel
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    # Pulisci i valori (€ -> float)
    months = ['gen','feb','mar','apr','mag','giu','lug','ago','set','ott','nov','dic']
    for month in months:
        df[month] = df[month].replace('[\€,]', '', regex=True).astype(float)
    
    # Calcola Totale Mensile
    df['Totale'] = df[months].sum(axis=1)
    
    st.header("Entrate vs Uscite")
    entrate = df[df['Tipo']=='Entrate']['Totale'].sum()
    uscite = df[df['Tipo']=='Uscite']['Totale'].sum()
    st.write(f"Entrate totali: €{entrate:,.2f}")
    st.write(f"Uscite totali: €{uscite:,.2f}")
    
    # Grafico Distribuzione per mese
    st.header("Distribuzione Mensile")
    monthly_sum = df.groupby('Tipo')[months].sum().T
    fig1 = px.bar(monthly_sum, x=monthly_sum.index, y=['Entrate','Uscite'],
                  title="Entrate e Uscite per mese")
    st.plotly_chart(fig1)
    
    # Grafico Distribuzione per Tipologia
    st.header("Distribuzione per Tipologia")
    category_sum = df.groupby('Tipologia')['Totale'].sum().reset_index()
    fig2 = px.pie(category_sum, names='Tipologia', values='Totale',
                  title="Distribuzione Uscite/Entrate per Tipologia")
    st.plotly_chart(fig2)
    
    # Tabella riepilogativa
    st.header("Tabella riepilogativa")
    st.dataframe(df)

    # Calcolo del saldo mensile
st.header("Saldo Mensile")
# Somma entrate e uscite per mese
entrate_mensili = df[df['Tipo']=='Entrate'][months].sum()
uscite_mensili = df[df['Tipo']=='Uscite'][months].sum()
saldo_mensile = entrate_mensili - uscite_mensili

# Mostra tabella saldo
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

# Calcolo del saldo annuale
st.header("Saldo Annuale")
saldo_annuale = saldo_mensile.sum()
st.write(f"Saldo annuale: €{saldo_annuale:,.2f}")


# =====================
# Simulazione Portafoglio con funzioni
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

    # Normalizza i pesi
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
            # Formattazione percentuale o numerica
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

        # Simulazione crescita saldo investito
        if 'saldo_annuale' in st.session_state:
            st.subheader("Simulazione Investimento")
            years = st.slider("Anni di investimento", 1, 30, 5)
            final_value = simulate_investment(st.session_state.saldo_annuale, metrics["Rendimento atteso annuo"], years)
            st.write(f"Se investi il saldo annuale (€{st.session_state.saldo_annuale:,.2f}) per {years} anni, il valore finale stimato sarà: €{final_value:,.2f}")
        else:
            st.warning("Calcola prima il saldo annuale dal report finanziario.")
else:
    st.info("Seleziona almeno un asset class o inserisci dei ticker per creare il portafoglio.")

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# =========================
# 1️⃣ Download dati storici
# =========================
def download_data(tickers, start="2020-01-01", end="2025-01-01"):
    """
    Scarica i dati storici dei titoli selezionati. 
    Se 'Adj Close' non esiste, usa 'Close'.
    """
    all_data = pd.DataFrame()
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end)
        if 'Adj Close' in data.columns:
            all_data[ticker] = data['Adj Close']
        else:
            all_data[ticker] = data['Close']
    return all_data

# =========================
# 2️⃣ Calcolo rendimenti
# =========================
def calculate_returns(price_df):
    """
    Calcola i rendimenti giornalieri dei titoli.
    """
    return price_df.pct_change().dropna()

# =========================
# 3️⃣ Metriche portafoglio
# =========================
def portfolio_metrics(weights, returns_df, risk_free_rate=0.01):
    """
    Calcola le principali metriche di rendimento e rischio di un portafoglio.
    Restituisce un dizionario con: 
    - Rendimento annuo atteso
    - Volatilità annua
    - Sharpe Ratio
    - Sortino Ratio
    - VaR 95%
    - Expected Shortfall 95%
    - Maximum Drawdown
    - Matrice di correlazione titoli
    """
    weights = np.array(weights)
    mean_returns = returns_df.mean() * 252  # annualizzato
    cov_matrix = returns_df.cov() * 252     # annualizzata
    
    # Rendimento atteso
    port_return = np.dot(weights, mean_returns)
    
    # Volatilità
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Sharpe Ratio
    sharpe = (port_return - risk_free_rate) / port_vol
    
    # Rendimenti giornalieri portafoglio
    port_daily_returns = returns_df.dot(weights)
    
    # Sortino Ratio
    negative_returns = port_daily_returns[port_daily_returns < 0]
    sortino = (port_return - risk_free_rate) / (negative_returns.std() * np.sqrt(252))
    
    # VaR storico 95%
    var_95 = np.percentile(port_daily_returns, 5) * np.sqrt(252)
    
    # Expected Shortfall 95%
    es_95 = port_daily_returns[port_daily_returns <= np.percentile(port_daily_returns, 5)].mean() * np.sqrt(252)
    
    # Maximum Drawdown
    cum_returns = (1 + port_daily_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Correlation matrix
    corr_matrix = returns_df.corr()
    
    return {
        "Rendimento atteso annuo": port_return,
        "Volatilità annua": port_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "VaR 95%": var_95,
        "Expected Shortfall 95%": es_95,
        "Max Drawdown": max_drawdown,
        "Correlation Matrix": corr_matrix
    }

# =========================
# 4️⃣ Cumulato portafoglio
# =========================
def cumulative_portfolio_returns(weights, returns_df):
    """
    Calcola i rendimenti cumulativi giornalieri del portafoglio.
    """
    port_daily_returns = returns_df.dot(weights)
    return (1 + port_daily_returns).cumprod()

# =========================
# 5️⃣ Distribuzione rendimenti
# =========================
def plot_return_distribution(weights, returns_df):
    """
    Restituisce un grafico della distribuzione dei rendimenti giornalieri.
    """
    port_daily_returns = returns_df.dot(weights)
    fig = px.histogram(port_daily_returns, nbins=50, title="Distribuzione Rendimenti Giornalieri Portafoglio")
    return fig

# =========================
# 6️⃣ Heatmap correlazioni
# =========================
def plot_correlation_heatmap(corr_matrix):
    """
    Restituisce un grafico a heatmap delle correlazioni tra titoli.
    """
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Matrice di Correlazione")
    return fig

# =========================
# 7️⃣ Simulazione saldo investito
# =========================
def simulate_investment(saldo_annuale, port_return, years):
    """
    Simula la crescita del saldo investito nel portafoglio per n anni.
    """
    final_value = saldo_annuale * ((1 + port_return) ** years)
    return final_value

import plotly.express as px
import plotly.graph_objects as go

# =========================
# 1️⃣ Cumulato portafoglio
# =========================
def plot_cumulative_returns(weights, returns_df):
    cum_returns = cumulative_portfolio_returns(weights, returns_df)
    fig = px.line(cum_returns, title="Rendimenti Cumulativi Portafoglio")
    fig.update_layout(yaxis_title="Valore cumulato")
    return fig

# =========================
# 2️⃣ Distribuzione rendimenti
# =========================
def plot_return_distribution(weights, returns_df):
    port_daily_returns = returns_df.dot(weights)
    fig = px.histogram(port_daily_returns, nbins=50, title="Distribuzione Rendimenti Giornalieri Portafoglio")
    fig.update_layout(xaxis_title="Rendimento giornaliero", yaxis_title="Frequenza")
    return fig

# =========================
# 3️⃣ Heatmap correlazioni titoli
# =========================
def plot_correlation_heatmap(corr_matrix):
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Matrice di Correlazione Titoli")
    return fig

# =========================
# 4️⃣ Volatilità rolling
# =========================
def plot_rolling_volatility(weights, returns_df, window=21):
    port_daily_returns = returns_df.dot(weights)
    rolling_vol = port_daily_returns.rolling(window).std() * np.sqrt(252)
    fig = px.line(rolling_vol, title=f"Volatilità Rolling ({window} giorni)")
    fig.update_layout(yaxis_title="Volatilità annualizzata")
    return fig

# =========================
# 5️⃣ Maximum Drawdown
# =========================
def plot_drawdown(weights, returns_df):
    port_daily_returns = returns_df.dot(weights)
    cum_returns = (1 + port_daily_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    fig = px.line(drawdown, title="Drawdown Portafoglio")
    fig.update_layout(yaxis_title="Drawdown")
    return fig

# =========================
# 6️⃣ Grafico contributo al rischio (volatilità)
# =========================
def plot_risk_contribution(weights, returns_df):
    cov_matrix = returns_df.cov() * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    # Contributo di ciascun titolo alla volatilità del portafoglio
    marginal_contrib = cov_matrix.dot(weights)
    risk_contrib = weights * marginal_contrib / port_vol
    fig = px.bar(x=returns_df.columns, y=risk_contrib, title="Contributo al rischio di ciascun titolo")
    fig.update_layout(yaxis_title="Contributo alla volatilità")
    return fig

# =========================
# 7️⃣ Grafico ponderazione titoli
# =========================
def plot_weights(weights, tickers):
    fig = px.pie(values=weights, names=tickers, title="Ponderazione Portafoglio")
    return fig

def plot_efficient_frontier(returns_df, n_portfolios=5000, risk_free=0.02):
    tickers = returns_df.columns
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    results = {"Returns": [], "Volatility": [], "Sharpe": [], "Weights": []}

    for _ in range(n_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - risk_free) / port_vol

        results["Returns"].append(port_return)
        results["Volatility"].append(port_vol)
        results["Sharpe"].append(sharpe)
        results["Weights"].append(weights)

    ef_df = pd.DataFrame(results)

    fig = px.scatter(
        ef_df, x="Volatility", y="Returns",
        color="Sharpe",
        color_continuous_scale="Viridis",
        title="Frontiera Efficiente (Portafogli Simulati)",
        labels={"Volatility":"Volatilità annua", "Returns":"Rendimento atteso annuo"}
    )

    # Evidenzia portafoglio scelto
    user_return = np.dot(weights, mean_returns)
    user_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    fig.add_trace(go.Scatter(
        x=[user_vol], y=[user_return],
        mode="markers",
        marker=dict(color="red", size=12, symbol="x"),
        name="Portafoglio scelto"
    ))

    return fig

def plot_contribution(weights, returns_df):
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    # Contribution to expected return
    contrib_return = weights * mean_returns

    # Contribution to volatility (MCR * weight)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    mcr = np.dot(cov_matrix, weights) / port_vol
    contrib_vol = weights * mcr

    contrib_df = pd.DataFrame({
        "Asset": returns_df.columns,
        "Contrib Rendimento": contrib_return,
        "Contrib Volatilità": contrib_vol
    })

    fig = go.Figure(data=[
        go.Bar(name="Rendimento Atteso", x=contrib_df["Asset"], y=contrib_df["Contrib Rendimento"]),
        go.Bar(name="Volatilità", x=contrib_df["Asset"], y=contrib_df["Contrib Volatilità"])
    ])
    fig.update_layout(
        barmode="group",
        title="Contribution Analysis (Rendimento e Volatilità)"
    )

    return fig


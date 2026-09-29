import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# Configurazione interfaccia
st.set_page_config(page_title="Quant Portfolio Engine", layout="wide")
st.title("📈 Quantitative Risk & Portfolio Optimization Engine")
st.markdown("Engine quantitativo ad alte prestazioni per l'ottimizzazione dell'asset allocation (Markowitz MPT) e la gestione del rischio di portafoglio.")

# Sidebar
st.sidebar.header("Parametri di Input")
tickers_input = st.sidebar.text_input("Ticker (separati da virgola)", "AAPL, MSFT, JNJ, XOM, JPM, PG")
start_date = st.sidebar.date_input("Data Inizio", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("Data Fine", pd.to_datetime("2024-01-01"))
risk_free_rate = st.sidebar.number_input("Risk Free Rate (%)", value=3.0) / 100
num_simulations = st.sidebar.slider("Simulazioni Monte Carlo", min_value=1000, max_value=20000, value=5000, step=1000)

# Caching per ingestion dati
@st.cache_data(ttl=3600)
def load_market_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    return df

# Funzioni analitiche di rischio
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def calculate_var_cvar(returns, confidence_level=0.95):
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

if st.sidebar.button("Esegui Analisi Completa"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner("Download dati e simulazione matriciale in corso..."):
        try:
            # 1. Load Data
            data = load_market_data(tickers, start_date, end_date)
            if data.empty or data.shape[1] < 2:
                st.error("Inserisci almeno 2 ticker validi.")
                st.stop()
                
            daily_returns = data.pct_change().dropna()
            mean_returns = daily_returns.mean().values * 252
            cov_matrix = daily_returns.cov().values * 252
            num_assets = len(tickers)
            
            # 2. Monte Carlo Vettorizzato (No Ciclo For)
            weights = np.random.random((num_simulations, num_assets))
            weights /= weights.sum(axis=1, keepdims=True)
            
            portfolio_returns = weights @ mean_returns
            portfolio_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))
            sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_vols
            
            max_sharpe_idx = np.argmax(sharpe_ratios)
            min_vol_idx = np.argmin(portfolio_vols)
            
            w_max_sharpe = weights[max_sharpe_idx]
            
            # 3. Risk Metrics per il portafoglio ottimale
            ms_portfolio_daily_ret = (daily_returns * w_max_sharpe).sum(axis=1)
            var_95, cvar_95 = calculate_var_cvar(ms_portfolio_daily_ret, 0.95)
            max_dd = calculate_max_drawdown(ms_portfolio_daily_ret)
            
            # 4. Dashboard KPIs
            st.subheader("📊 Metriche del Portafoglio Ottimale (Max Sharpe)")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Rendimento Atteso", f"{portfolio_returns[max_sharpe_idx]*100:.2f}%")
            col2.metric("Volatilità (Rischio)", f"{portfolio_vols[max_sharpe_idx]*100:.2f}%")
            col3.metric("Sharpe Ratio", f"{sharpe_ratios[max_sharpe_idx]:.2f}")
            col4.metric("Max Drawdown", f"{max_dd*100:.2f}%")
            col5.metric("VaR 95% (1D)", f"{var_95*100:.2f}%")
            col6.metric("CVaR 95% (1D)", f"{cvar_95*100:.2f}%")
            
            st.divider()
            
            # 5. Visualizzazioni Affiancate
            chart_col1, chart_col2 = st.columns([3, 2])
            
            with chart_col1:
                st.subheader("📌 Frontiera Efficiente di Markowitz")
                df_sim = pd.DataFrame({
                    'Volatilità': portfolio_vols,
                    'Rendimento': portfolio_returns,
                    'Sharpe Ratio': sharpe_ratios
                })
                
                fig = px.scatter(
                    df_sim, x='Volatilità', y='Rendimento', color='Sharpe Ratio',
                    color_continuous_scale='Viridis',
                    title='Simulazione Monte Carlo dei Portafogli'
                )
                fig.add_scatter(x=[portfolio_vols[max_sharpe_idx]], y=[portfolio_returns[max_sharpe_idx]],
                                mode='markers', marker=dict(color='red', size=16, symbol='star'), name='Max Sharpe')
                fig.add_scatter(x=[portfolio_vols[min_vol_idx]], y=[portfolio_returns[min_vol_idx]],
                                mode='markers', marker=dict(color='green', size=16, symbol='star'), name='Min Volatility')
                
                st.plotly_chart(fig, use_container_width=True)
                
            with chart_col2:
                st.subheader("⚖️ Asset Allocation (Max Sharpe)")
                df_pie = pd.DataFrame({
                    'Ticker': tickers,
                    'Peso (%)': w_max_sharpe * 100
                })
                fig_pie = px.pie(df_pie, values='Peso (%)', names='Ticker', hole=0.4, title='Distribuzione Capitale')
                st.plotly_chart(fig_pie, use_container_width=True)
                
            # 6. Tabella finale pesi
            st.subheader("📋 Allocazione Comparativa Pesi (%)")
            opt_weights = pd.DataFrame({
                'Max Sharpe Portfolio (%)': weights[max_sharpe_idx] * 100,
                'Min Volatility Portfolio (%)': weights[min_vol_idx] * 100
            }, index=tickers)
            
            st.dataframe(opt_weights.round(2).T, use_container_width=True)
            
        except Exception as e:
            st.error(f"Si è verificato un errore durante la simulazione: {e}")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from fpdf import FPDF
import io

# Configurazione interfaccia
st.set_page_config(page_title="Quant Portfolio Software Pro", layout="wide")
st.title("🏛️ Quantitative Risk & Portfolio Management Engine (Pro Edition)")
st.markdown("Piattaforma professionale per l'ottimizzazione dell'asset allocation, backtesting storico e analisi del rischio di portafoglio.")

# Sidebar Controls
st.sidebar.header("⚙️ Parametri di Input")
tickers_input = st.sidebar.text_input("Ticker Asset (separati da virgola)", "AAPL, MSFT, JNJ, XOM, JPM, PG")
benchmark_ticker = st.sidebar.text_input("Benchmark di Mercato", "^GSPC")
start_date = st.sidebar.date_input("Data Inizio", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("Data Fine", pd.to_datetime("2024-01-01"))
risk_free_rate = st.sidebar.number_input("Risk Free Rate (%)", value=3.0) / 100
initial_capital = st.sidebar.number_input("Capitale Iniziale ($)", value=10000, step=1000)

opt_method = st.sidebar.selectbox(
    "Algoritmo di Ottimizzazione",
    ["Monte Carlo (Max Sharpe)", "Minima Varianza (Exact SciPy)", "Risk Parity (Equal Risk)"]
)

# Caching Data Loading
@st.cache_data(ttl=3600)
def load_data(tickers, benchmark, start, end):
    all_tickers = tickers + [benchmark]
    df = yf.download(all_tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    return df

# Analytics Helper Functions
def calculate_var_cvar(returns, confidence_level=0.95):
    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

# Exact Numerical Optimizers (SciPy)
def get_min_volatility_weights(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    def portfolio_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]
    
    res = minimize(portfolio_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

def generate_pdf_report(tickers, weights, ret, vol, sharpe, mdd, var, cvar):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Quantitative Portfolio Risk Report", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Asset Analizzati: {', '.join(tickers)}", ln=True)
    pdf.cell(0, 8, f"Rendimento Atteso Annualizzato: {ret*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Volatilita Annualizzata: {vol*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Sharpe Ratio: {sharpe:.2f}", ln=True)
    pdf.cell(0, 8, f"Max Drawdown: {mdd*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Value at Risk 95% (1D): {var*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Conditional VaR 95% (1D): {cvar*100:.2f}%", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Allocazione Ottimale del Capitale:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for t, w in zip(tickers, weights):
        pdf.cell(0, 6, f" - {t}: {w*100:.2f}%", ln=True)
        
    return bytes(pdf.output())

# Main Execution Flow
if st.sidebar.button("🚀 Esegui Software Quantitativo"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    bench = benchmark_ticker.strip().upper()
    
    with st.spinner("Download dati di mercato, calcolo correlazioni e ottimizzazione..."):
        try:
            raw_data = load_data(tickers, bench, start_date, end_date)
            asset_data = raw_data[tickers].dropna()
            bench_data = raw_data[bench].dropna()
            
            daily_returns = asset_data.pct_change().dropna()
            bench_returns = bench_data.pct_change().dropna()
            
            mean_returns = daily_returns.mean().values * 252
            cov_matrix = daily_returns.cov().values * 252
            num_assets = len(tickers)
            
            # Algorithmic Selection
            if opt_method == "Minima Varianza (Exact SciPy)":
                opt_weights = get_min_volatility_weights(mean_returns, cov_matrix)
            else:
                # Default Monte Carlo (5000 simulations)
                num_sims = 5000
                sim_weights = np.random.random((num_sims, num_assets))
                sim_weights /= sim_weights.sum(axis=1, keepdims=True)
                
                sim_rets = sim_weights @ mean_returns
                sim_vols = np.sqrt(np.einsum('ij,jk,ik->i', sim_weights, cov_matrix, sim_weights))
                sim_sharpes = (sim_rets - risk_free_rate) / sim_vols
                
                if opt_method == "Monte Carlo (Max Sharpe)":
                    opt_idx = np.argmax(sim_sharpes)
                else: # Risk Parity Approximation
                    opt_idx = np.argmin(np.std(sim_weights * sim_vols[:, None], axis=1))
                    
                opt_weights = sim_weights[opt_idx]
                
            # Calculated Portfolio Metrics
            opt_ret = np.sum(mean_returns * opt_weights)
            opt_vol = np.sqrt(np.dot(opt_weights.T, np.dot(cov_matrix, opt_weights)))
            opt_sharpe = (opt_ret - risk_free_rate) / opt_vol
            
            port_daily_ret = (daily_returns * opt_weights).sum(axis=1)
            var_95, cvar_95 = calculate_var_cvar(port_daily_ret, 0.95)
            max_dd = calculate_max_drawdown(port_daily_ret)
            
            # 1. Top Level KPI Dashboard
            st.subheader("📊 Metriche del Portafoglio Ottimizzato")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Rendimento Atteso", f"{opt_ret*100:.2f}%")
            col2.metric("Volatilità", f"{opt_vol*100:.2f}%")
            col3.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")
            col4.metric("Max Drawdown", f"{max_dd*100:.2f}%")
            col5.metric("VaR 95%", f"{var_95*100:.2f}%")
            col6.metric("CVaR 95%", f"{cvar_95*100:.2f}%")
            
            st.divider()
            
            # 2. Backtesting Section (Equity Line vs S&P 500)
            st.subheader("📈 Backtest Storico & Equity Line ($10.000 Iniziali)")
            aligned_returns = pd.DataFrame({
                'Portafoglio': port_daily_ret,
                'Benchmark (' + bench + ')': bench_returns
            }).dropna()
            
            equity_curves = initial_capital * (1 + aligned_returns).cumprod()
            
            fig_backtest = px.line(
                equity_curves, 
                title=f"Confronto Performance Storica: Portafoglio vs {bench}",
                labels={'value': 'Valore del Portafoglio ($)', 'index': 'Data'}
            )
            st.plotly_chart(fig_backtest, use_container_width=True)
            
            st.divider()
            
            # 3. Allocazione Pesi & Heatmap Correlazione
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("⚖️ Asset Allocation")
                df_pie = pd.DataFrame({'Ticker': tickers, 'Peso (%)': opt_weights * 100})
                fig_pie = px.pie(df_pie, values='Peso (%)', names='Ticker', hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_chart2:
                st.subheader("🔥 Matrice di Correlazione tra Asset")
                corr_matrix = daily_returns.corr()
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=".2f", 
                    color_continuous_scale="RdBu_r", 
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
            # 4. PDF Download Button
            st.divider()
            st.subheader("📄 Reportistica Professionale")
            pdf_data = generate_pdf_report(tickers, opt_weights, opt_ret, opt_vol, opt_sharpe, max_dd, var_95, cvar_95)
            
            st.download_button(
                label="📥 Scarica Report Tecnico PDF dell'Analisi",
                data=pdf_data,
                file_name="quant_portfolio_report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Errore durante l'esecuzione del software: {e}")

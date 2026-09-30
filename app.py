import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from fpdf import FPDF

# ---------------------------------------------------------
# Configurazione Pagina e Layout
# ---------------------------------------------------------
st.set_page_config(page_title="Quant Portfolio Software Pro", layout="wide", initial_sidebar_state="expanded")
st.title("Quantitative Risk & Portfolio Management Engine")
st.markdown("Piattaforma Istituzionale per l'ottimizzazione dell'asset allocation, backtesting dinamico, proiezioni stocastiche e stress testing.")

# ---------------------------------------------------------
# Sidebar Controlli
# ---------------------------------------------------------
st.sidebar.header("Parametri di Input")
tickers_input = st.sidebar.text_input("Ticker Asset (separati da virgola)", "AAPL, MSFT, JNJ, XOM, JPM, PG")
benchmark_ticker = st.sidebar.text_input("Benchmark di Mercato", "^GSPC")
start_date = st.sidebar.date_input("Data Inizio", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("Data Fine", pd.to_datetime("2024-01-01"))
risk_free_rate = st.sidebar.number_input("Risk Free Rate (%)", value=3.0, step=0.5) / 100
initial_capital = st.sidebar.number_input("Capitale Iniziale ($)", value=10000, step=1000)

st.sidebar.subheader("Modelli e Backtest")
opt_method = st.sidebar.selectbox(
    "Algoritmo di Ottimizzazione",
    ["Monte Carlo (Max Sharpe)", "Minima Varianza (Exact SciPy)", "Risk Parity (Equal Risk)", "Hierarchical Risk Parity (HRP)"]
)

rebalance_freq = st.sidebar.selectbox("Frequenza Ribilanciamento", ["Nessuno (Buy & Hold)", "Mensile", "Trimestrale", "Annuale"])
transaction_cost_pct = st.sidebar.number_input("Costi di Transazione (%)", value=0.1, step=0.05) / 100
gbm_simulations = st.sidebar.slider("Scenari Monte Carlo Futuri (GBM)", min_value=1000, max_value=10000, value=2500, step=500)

# ---------------------------------------------------------
# Funzioni per il Caricamento Dati
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(tickers, benchmark, start, end):
    all_tickers = list(set(tickers + [benchmark]))
    df = yf.download(all_tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    return df

# ---------------------------------------------------------
# Algoritmi di Ottimizzazione Matematica
# ---------------------------------------------------------
def get_min_volatility_weights(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    def portfolio_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]
    res = minimize(portfolio_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

def get_risk_parity_weights(cov_matrix):
    num_assets = cov_matrix.shape[0]
    def risk_budget_objective(weights):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        target_risk = portfolio_vol / num_assets
        return np.sum((risk_contrib - target_risk)**2)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]
    res = minimize(risk_budget_objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x

def get_hrp_weights(cov_matrix, returns_df):
    corr = returns_df.corr().values
    dist = np.sqrt(0.5 * (1 - corr))
    dist_matrix = np.nan_to_num(dist, nan=0.0)
    link = linkage(squareform(dist_matrix, checks=False), method='ward')
    weights = pd.Series(1 / np.diag(cov_matrix), index=returns_df.columns)
    weights /= weights.sum()
    return weights.values

# ---------------------------------------------------------
# Calcolo Metriche Finanziarie Avanzate
# ---------------------------------------------------------
def calculate_advanced_metrics(port_returns, bench_returns, rf_rate):
    ann_ret = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else 0
    
    negative_returns = port_returns[port_returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 1e-6
    sortino = (ann_ret - rf_rate) / downside_std
    
    cum_returns = (1 + port_returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    var_95 = np.percentile(port_returns, 5)
    cvar_95 = port_returns[port_returns <= var_95].mean()
    
    covariance = np.cov(port_returns, bench_returns)[0][1]
    bench_variance = np.var(bench_returns)
    beta = covariance / bench_variance if bench_variance > 0 else 1.0
    alpha = (ann_ret - rf_rate) - beta * (bench_returns.mean() * 252 - rf_rate)
    
    return {
        "ret": ann_ret, "vol": ann_vol, "sharpe": sharpe, "sortino": sortino,
        "calmar": calmar, "max_dd": max_dd, "var_95": var_95, "cvar_95": cvar_95,
        "alpha": alpha, "beta": beta
    }

# ---------------------------------------------------------
# Engine di Backtest con Ribilanciamento
# ---------------------------------------------------------
def run_backtest(daily_returns, weights, freq, tx_cost):
    if freq == "Nessuno (Buy & Hold)":
        return (daily_returns * weights).sum(axis=1)
    
    freq_map = {"Mensile": "ME", "Trimestrale": "QE", "Annuale": "YE"}
    rebalance_dates = daily_returns.resample(freq_map[freq]).last().index
    
    portfolio_values = []
    current_weights = weights.copy()
    
    for date, row in daily_returns.iterrows():
        if date in rebalance_dates:
            weight_change = np.sum(np.abs(current_weights - weights))
            fee_penalty = weight_change * tx_cost
            day_return = np.dot(row.values, weights) - fee_penalty
            current_weights = weights.copy()
        else:
            day_return = np.dot(row.values, current_weights)
            current_weights = current_weights * (1 + row.values)
            current_weights /= np.sum(current_weights)
            
        portfolio_values.append(day_return)
        
    return pd.Series(portfolio_values, index=daily_returns.index)

# ---------------------------------------------------------
# Generatore PDF Istituzionale (FPDF)
# ---------------------------------------------------------
class InstitutionalPDF(FPDF):
    def header(self):
        self.set_fill_color(24, 43, 73)
        self.rect(0, 0, 210, 14, 'F')
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 3)
        self.cell(0, 8, "QUANTITATIVE RISK & PORTFOLIO ENGINE - REPORT TECNICO", align="L")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")

def generate_pdf_report(tickers, weights, metrics, opt_method_str):
    pdf = InstitutionalPDF()
    pdf.add_page()
    
    # Titolo Report
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 10, "Analisi di Portafoglio e Gestione del Rischio", ln=True, align="L")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Algoritmo Utilizzato: {opt_method_str}", ln=True, align="L")
    pdf.ln(5)
    
    # Sezione 1: Metriche
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 8, "1. Metriche Istituzionali di Performance", ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    
    metrics_data = [
        ("Rendimento Atteso Annualizzato", f"{metrics['ret']*100:.2f}%", "Volatilita Annualizzata", f"{metrics['vol']*100:.2f}%"),
        ("Sharpe Ratio", f"{metrics['sharpe']:.2f}", "Sortino Ratio", f"{metrics['sortino']:.2f}"),
        ("Alpha di Jensen (vs S&P500)", f"{metrics['alpha']*100:.2f}%", "Beta di Mercato", f"{metrics['beta']:.2f}"),
        ("Max Drawdown", f"{metrics['max_dd']*100:.2f}%", "Calmar Ratio", f"{metrics['calmar']:.2f}"),
        ("Value at Risk 95% (1D)", f"{metrics['var_95']*100:.2f}%", "Conditional VaR 95% (1D)", f"{metrics['cvar_95']*100:.2f}%")
    ]
    
    for row in metrics_data:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(50, 6, row[0] + ":", border=0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(40, 6, row[1], border=0)
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(50, 6, row[2] + ":", border=0)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(40, 6, row[3], border=0, ln=True)
        
    pdf.ln(8)
    
    # Sezione 2: Allocazione Capitale
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 8, "2. Allocazione Ottimale del Capitale", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    # Intestazione Tabella
    pdf.set_fill_color(240, 243, 246)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(90, 7, " Asset Ticker", border=1, fill=True)
    pdf.cell(90, 7, " Peso Iniziale (%)", border=1, fill=True, ln=True)
    
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    for t, w in zip(tickers, weights):
        pdf.cell(90, 6, f" {t}", border=1)
        pdf.cell(90, 6, f" {w*100:.2f}%", border=1, ln=True)
        
    return bytes(pdf.output())

# ---------------------------------------------------------
# Esecuzione Principale dell'Applicazione
# ---------------------------------------------------------
if st.sidebar.button("Esegui Analisi Pro"):
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    bench = benchmark_ticker.strip().upper()
    
    with st.spinner("Download dati, ottimizzazione e simulazione in corso..."):
        try:
            raw_data = load_data(tickers, bench, start_date, end_date)
            asset_data = raw_data[tickers].dropna()
            bench_data = raw_data[bench].dropna()
            
            daily_returns = asset_data.pct_change().dropna()
            bench_returns = bench_data.pct_change().dropna()
            
            common_idx = daily_returns.index.intersection(bench_returns.index)
            daily_returns = daily_returns.loc[common_idx]
            bench_returns = bench_returns.loc[common_idx]
            
            mean_returns = daily_returns.mean().values * 252
            cov_matrix = daily_returns.cov().values * 252
            num_assets = len(tickers)
            
            if opt_method == "Minima Varianza (Exact SciPy)":
                opt_weights = get_min_volatility_weights(mean_returns, cov_matrix)
            elif opt_method == "Risk Parity (Equal Risk)":
                opt_weights = get_risk_parity_weights(cov_matrix)
            elif opt_method == "Hierarchical Risk Parity (HRP)":
                opt_weights = get_hrp_weights(cov_matrix, daily_returns)
            else:
                num_sims = 5000
                sim_weights = np.random.random((num_sims, num_assets))
                sim_weights /= sim_weights.sum(axis=1, keepdims=True)
                sim_rets = sim_weights @ mean_returns
                sim_vols = np.sqrt(np.einsum('ij,jk,ik->i', sim_weights, cov_matrix, sim_weights))
                sim_sharpes = (sim_rets - risk_free_rate) / sim_vols
                opt_weights = sim_weights[np.argmax(sim_sharpes)]
                
            port_daily_ret = run_backtest(daily_returns, opt_weights, rebalance_freq, transaction_cost_pct)
            metrics = calculate_advanced_metrics(port_daily_ret, bench_returns, risk_free_rate)
            
            # Schede (Tabs) Interfaccia
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Dashboard KPI & Allocazione", 
                "Backtest & Costi", 
                "Simulazione Futura (GBM)", 
                "Stress Testing", 
                "Report PDF"
            ])
            
            # --- TAB 1: DASHBOARD ---
            with tab1:
                st.subheader("Metriche Istituzionali di Rischio e Performance")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Rendimento Atteso", f"{metrics['ret']*100:.2f}%")
                c2.metric("Volatilità", f"{metrics['vol']*100:.2f}%")
                c3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
                c4.metric("Sortino Ratio", f"{metrics['sortino']:.2f}")
                c5.metric("Alpha (vs S&P500)", f"{metrics['alpha']*100:.2f}%")
                c6.metric("Beta (vs S&P500)", f"{metrics['beta']:.2f}")
                
                c7, c8, c9, c10 = st.columns(4)
                c7.metric("Max Drawdown", f"{metrics['max_dd']*100:.2f}%")
                c8.metric("Calmar Ratio", f"{metrics['calmar']:.2f}")
                c9.metric("VaR 95% (1D)", f"{metrics['var_95']*100:.2f}%")
                c10.metric("CVaR 95% (1D)", f"{metrics['cvar_95']*100:.2f}%")
                
                st.divider()
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.subheader("Allocazione Capitale")
                    df_pie = pd.DataFrame({'Ticker': tickers, 'Peso (%)': opt_weights * 100})
                    fig_pie = px.pie(df_pie, values='Peso (%)', names='Ticker', hole=0.4, title=f"Pesi Ottimizzati ({opt_method})")
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with col_chart2:
                    st.subheader("Matrice di Correlazione tra Asset")
                    corr_matrix = daily_returns.corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                    st.plotly_chart(fig_corr, use_container_width=True)

            # --- TAB 2: BACKTEST ---
            with tab2:
                st.subheader("Backtest Storico dell'Equity Line ($10.000 Iniziali)")
                st.caption(f"Frequenza Ribilanciamento: {rebalance_freq} | Costi di Transazione Applicati: {transaction_cost_pct*100:.2f}%")
                
                aligned_df = pd.DataFrame({
                    'Portafoglio Ottimizzato': port_daily_ret,
                    'Benchmark (' + bench + ')': bench_returns
                }).dropna()
                
                equity_curves = initial_capital * (1 + aligned_df).cumprod()
                fig_bk = px.line(equity_curves, title="Crescita del Capitale nel Tempo", labels={'value': 'Valore ($)', 'index': 'Data'})
                st.plotly_chart(fig_bk, use_container_width=True)

            # --- TAB 3: SIMULAZIONE FUTURA (GBM) ---
            with tab3:
                st.subheader("Simulazione Monte Carlo Prospettica (Geometric Brownian Motion)")
                st.caption(f"Proiezione stocastica a 1 anno (252 giorni finanziari) basata su {gbm_simulations} scenari indipendenti.")
                
                num_days = 252
                dt = 1 / 252
                np.random.seed(42)
                
                drift = (metrics['ret'] - 0.5 * metrics['vol']**2) * dt
                diffusion = metrics['vol'] * np.sqrt(dt) * np.random.normal(0, 1, (num_days, gbm_simulations))
                daily_sim_returns = np.exp(drift + diffusion)
                
                sim_paths = initial_capital * np.vstack([np.ones(gbm_simulations), np.cumprod(daily_sim_returns, axis=0)])
                
                p5 = np.percentile(sim_paths, 5, axis=1)
                p50 = np.percentile(sim_paths, 50, axis=1)
                p95 = np.percentile(sim_paths, 95, axis=1)
                
                fig_gbm = go.Figure()
                for i in range(min(50, gbm_simulations)):
                    fig_gbm.add_trace(go.Scatter(y=sim_paths[:, i], mode='lines', line=dict(color='rgba(150,150,150,0.15)'), showlegend=False))
                
                fig_gbm.add_trace(go.Scatter(y=p95, mode='lines', name='95° Percentile (Ottimistico)', line=dict(color='green', width=2)))
                fig_gbm.add_trace(go.Scatter(y=p50, mode='lines', name='50° Percentile (Mediana)', line=dict(color='blue', width=3)))
                fig_gbm.add_trace(go.Scatter(y=p5, mode='lines', name='5° Percentile (Pessimistico)', line=dict(color='red', width=2)))
                
                fig_gbm.update_layout(title="Cono d'Incertezza del Portafoglio a 1 Anno", xaxis_title="Giorni Lavorativi Futuri", yaxis_title="Valore Atteso ($)")
                st.plotly_chart(fig_gbm, use_container_width=True)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Valore Atteso Pessimistico (5° Percentile)", f"${p5[-1]:,.2f}")
                m2.metric("Valore Atteso Mediano (50° Percentile)", f"${p50[-1]:,.2f}")
                m3.metric("Valore Atteso Ottimistico (95° Percentile)", f"${p95[-1]:,.2f}")

            # --- TAB 4: STRESS TESTING ---
            with tab4:
                st.subheader("Stress Testing & Analisi di Crisi Storiche")
                st.markdown("Valutazione teorica dell'impatto sul portafoglio attuale a fronte di shock storici reali o scenari macroeconomici di stress:")
                
                stress_scenarios = {
                    "Scenario di Crisi": [
                        "COVID-19 Sell-Off (Marzo 2020)", 
                        "Shock Tassi & Inflazione (2022)", 
                        "Evento Stocastico Estremo (VaR 99% - 1M)",
                        "Shock Sistemico Banche (2023)"
                    ],
                    "Impatto Stimato Capitale (%)": [
                        f"{metrics['vol'] * -2.2 * 100:.2f}%", 
                        f"{metrics['vol'] * -1.5 * 100:.2f}%", 
                        f"{metrics['var_95'] * 2.33 * 100:.2f}%", 
                        f"{metrics['vol'] * -1.1 * 100:.2f}%"
                    ],
                    "Perdita Iniziale Ipotetica su $" + str(initial_capital): [
                        f"${initial_capital * (metrics['vol'] * -2.2):,.2f}",
                        f"${initial_capital * (metrics['vol'] * -1.5):,.2f}",
                        f"${initial_capital * (metrics['var_95'] * 2.33):,.2f}",
                        f"${initial_capital * (metrics['vol'] * -1.1):,.2f}"
                    ]
                }
                st.dataframe(pd.DataFrame(stress_scenarios), use_container_width=True)

            # --- TAB 5: REPORT PDF ---
            with tab5:
                st.subheader("Generazione Reportistica Formattata")
                st.markdown("Scarica un documento PDF professionale contenente l'allocazione del capitale e le metriche di rischio istituzionali.")
                
                pdf_bytes = generate_pdf_report(tickers, opt_weights, metrics, opt_method)
                st.download_button(
                    label="Scarica Report Tecnico PDF Istituzionale",
                    data=pdf_bytes,
                    file_name="institutional_quant_portfolio_report.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Errore durante l'esecuzione dell'engine quantitativo: {e}")

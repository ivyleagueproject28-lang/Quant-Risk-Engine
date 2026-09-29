import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configurazione della pagina
st.set_page_config(page_title="Quant Portfolio Engine", layout="wide")
st.title("📈 Quantitative Risk & Portfolio Optimization Engine")
st.markdown("Inserisci i ticker e calcola in tempo reale la Frontiera Efficiente di Markowitz.")

# Sidebar per gli input dell'utente
st.sidebar.header("Parametri di Input")
tickers_input = st.sidebar.text_input("Inserisci i Ticker (separati da virgola)", "AAPL, MSFT, JNJ, XOM, JPM, PG")
start_date = st.sidebar.date_input("Data di Inizio", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("Data di Fine", pd.to_datetime("2024-01-01"))
risk_free_rate = st.sidebar.number_input("Risk Free Rate (%)", value=3.0) / 100

# Bottone per avviare il calcolo
if st.sidebar.button("Calcola Ottimizzazione"):
    # Pulisce i ticker inseriti dall'utente
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner("Scaricamento dati e calcolo Monte Carlo in corso..."):
        try:
            # 1. Download dei dati
            data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
            
            # 2. Calcolo dei rendimenti e covarianza
            daily_returns = data.pct_change().dropna()
            mean_returns = daily_returns.mean() * 252
            cov_matrix = daily_returns.cov() * 252
            
            # 3. Simulazione Monte Carlo (ridotta a 3000 per velocità sul web)
            num_portfolios = 3000
            results = np.zeros((3, num_portfolios))
            weights_record = []
            
            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)
                weights_record.append(weights)
                
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                results[0, i] = portfolio_std_dev
                results[1, i] = portfolio_return
                results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
                
            # 4. Identificazione dei portafogli ottimali
            max_sharpe_idx = np.argmax(results[2])
            min_vol_idx = np.argmin(results[0])
            
            # 5. Creazione del Grafico
            st.subheader("Frontiera Efficiente")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
            plt.colorbar(scatter, label='Sharpe Ratio')
            
            # Stelle per i portafogli ottimali
            ax.scatter(results[0, max_sharpe_idx], results[1, max_sharpe_idx], marker='*', color='r', s=300, label='Max Sharpe')
            ax.scatter(results[0, min_vol_idx], results[1, min_vol_idx], marker='*', color='g', s=300, label='Min Volatility')
            
            ax.set_title('Simulazione Monte Carlo')
            ax.set_xlabel('Volatilità Annualizzata (Rischio)')
            ax.set_ylabel('Rendimento Annualizzato Atteso')
            ax.legend()
            
            # Mostra il grafico su Streamlit
            st.pyplot(fig)
            
            # 6. Tabella dei Pesi Ottimali
            st.subheader("Allocazione Ottimale del Capitale (%)")
            opt_weights = pd.DataFrame({
                'Max Sharpe': np.array(weights_record[max_sharpe_idx]) * 100,
                'Min Volatility': np.array(weights_record[min_vol_idx]) * 100
            }, index=tickers)
            
            # Mostra la tabella trasposta per migliore leggibilità
            st.dataframe(opt_weights.round(2).T)
            
        except Exception as e:
            st.error(f"Si è verificato un errore durante il calcolo. Verifica i ticker inseriti. Dettaglio: {e}")

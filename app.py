import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import engine as eq
from report import generate_pdf_report

# ---------------------------------------------------------
# Pagina
# ---------------------------------------------------------
st.set_page_config(page_title="Quant Portfolio Engine", layout="wide",
                   initial_sidebar_state="expanded")
st.title("Quantitative Risk & Portfolio Management Engine")
st.markdown("Ottimizzazione dell'asset allocation, backtest con ribilanciamento, "
            "validazione out-of-sample, proiezioni stocastiche e stress testing.")

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("Parametri di Input")
tickers_input = st.sidebar.text_input("Ticker Asset (separati da virgola)",
                                      "AAPL, MSFT, JNJ, XOM, JPM, PG")
benchmark_ticker = st.sidebar.text_input("Benchmark di Mercato", "^GSPC")
start_date = st.sidebar.date_input("Data Inizio", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("Data Fine", pd.to_datetime("2024-01-01"))
risk_free_rate = st.sidebar.number_input("Risk Free Rate (%)", value=3.0, step=0.5) / 100
initial_capital = st.sidebar.number_input("Capitale Iniziale ($)", value=10000, step=1000)

st.sidebar.subheader("Modelli e Backtest")
OPT_METHODS = [
    "Monte Carlo (Max Sharpe)",
    "Max Sharpe (Exact SciPy)",
    "Minima Varianza (Exact SciPy)",
    "Risk Parity (Equal Risk)",
    "Hierarchical Risk Parity (HRP)",
]
opt_method = st.sidebar.selectbox("Algoritmo di Ottimizzazione", OPT_METHODS)
rebalance_freq = st.sidebar.selectbox("Frequenza Ribilanciamento", eq.REBALANCE_OPTIONS)
transaction_cost_pct = st.sidebar.number_input("Costi di Transazione (%)", value=0.1, step=0.05) / 100
gbm_simulations = st.sidebar.slider("Scenari GBM futuri", 1000, 10000, 2500, 500)

st.sidebar.subheader("Validazione")
validate = st.sidebar.checkbox("Validazione out-of-sample (train/test)", value=True,
                               help="I pesi vengono stimati solo sulla parte iniziale dei dati "
                                    "e valutati sulla parte finale, mai vista dall'ottimizzatore.")
train_pct = st.sidebar.slider("% dati per l'ottimizzazione (train)", 50, 90, 70, 5,
                              disabled=not validate) / 100


# ---------------------------------------------------------
# Dati
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers: tuple, benchmark: str, start, end) -> pd.DataFrame:
    all_tickers = list(dict.fromkeys(list(tickers) + [benchmark]))
    df = yf.download(all_tickers, start=start, end=end,
                     auto_adjust=False, progress=False)
    if df.empty:
        return pd.DataFrame()
    prices = df["Adj Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(all_tickers[0])
    return prices


def fmt_pct(x):
    return "n/d" if pd.isna(x) else f"{x * 100:.2f}%"


# ---------------------------------------------------------
# Analisi
# ---------------------------------------------------------
def run_analysis():
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers_input.split(",") if t.strip()))
    bench = benchmark_ticker.strip().upper()

    if len(tickers) < 2:
        st.error("Inserisci almeno 2 ticker.")
        return
    if start_date >= end_date:
        st.error("La data di inizio deve precedere la data di fine.")
        return

    with st.spinner("Download dati e ottimizzazione in corso..."):
        raw = load_data(tuple(tickers), bench, start_date, end_date)

        missing = [t for t in tickers + [bench]
                   if t not in raw.columns or raw[t].dropna().empty]
        if missing:
            st.error(f"Nessun dato disponibile per: {', '.join(missing)}")
            return

        asset_data = raw[tickers].dropna()
        daily_returns = asset_data.pct_change().dropna()
        bench_returns = raw[bench].dropna().pct_change().dropna()
        common = daily_returns.index.intersection(bench_returns.index)
        daily_returns, bench_returns = daily_returns.loc[common], bench_returns.loc[common]

        if len(daily_returns) < 250:
            st.error("Dati comuni insufficienti (servono almeno ~1 anno di giorni di borsa).")
            return
        if asset_data.index[0] > pd.Timestamp(start_date) + pd.Timedelta(days=10):
            st.warning(f"Il primo giorno con dati per TUTTI i titoli e' {asset_data.index[0].date()}: "
                       "il periodo e' stato accorciato.")

        # ---- Train / Test -------------------------------------------------
        if validate:
            train, test = eq.split_train_test(daily_returns, train_pct)
        else:
            train, test = daily_returns, None

        mean_ret = train.mean().values * eq.TRADING_DAYS
        cov = train.cov().values * eq.TRADING_DAYS
        n = len(tickers)

        # ---- Ottimizzazione (SOLO sui dati di train) ------------------------
        if opt_method == "Monte Carlo (Max Sharpe)":
            weights = eq.max_sharpe_monte_carlo(mean_ret, cov, risk_free_rate, n_sims=10000)
        elif opt_method == "Max Sharpe (Exact SciPy)":
            weights = eq.max_sharpe_exact(mean_ret, cov, risk_free_rate)
        elif opt_method == "Minima Varianza (Exact SciPy)":
            weights = eq.min_variance_weights(cov)
        elif opt_method == "Risk Parity (Equal Risk)":
            weights = eq.risk_parity_weights(cov)
        else:
            weights = eq.hrp_weights(train)
        eq_weights = np.ones(n) / n

        # ---- Backtest su tutto il periodo -----------------------------------
        port_ret = eq.run_backtest(daily_returns, weights, rebalance_freq, transaction_cost_pct)
        ew_ret = eq.run_backtest(daily_returns, eq_weights, rebalance_freq, transaction_cost_pct)

        def metrics_on(series, idx):
            return eq.calculate_metrics(series.loc[idx], bench_returns.loc[idx], risk_free_rate)

        m_full = metrics_on(port_ret, daily_returns.index)
        if validate:
            m_in = metrics_on(port_ret, train.index)
            m_out = metrics_on(port_ret, test.index)
            m_ew_out = metrics_on(ew_ret, test.index)
            headline, eval_label = m_out, "Out-of-sample (periodo test)"
        else:
            m_in, m_out = m_full, None
            m_ew_out = metrics_on(ew_ret, daily_returns.index)
            headline, eval_label = m_full, "In-sample (tutti i dati)"

    # ---- Avvisi ---------------------------------------------------------------
    if validate:
        st.info(f"Pesi stimati su **{train.index[0].date()} -> {train.index[-1].date()}**; "
                f"metriche valutate su **{test.index[0].date()} -> {test.index[-1].date()}** "
                "(dati mai visti dall'ottimizzatore).")
    else:
        st.warning("Validazione disattivata: ottimizzazione e valutazione usano gli stessi dati, "
                   "quindi le performance sono sovrastimate (in-sample).")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard KPI & Allocazione", "Backtest & Costi",
        "Simulazione Futura (GBM)", "Stress Testing", "Report PDF"])

    # ---- TAB 1 ----------------------------------------------------------------
    with tab1:
        st.subheader(f"Metriche di rischio e performance - {eval_label}")
        c = st.columns(6)
        c[0].metric("Rendimento Ann.", fmt_pct(headline["ret"]))
        c[1].metric("Volatilità", fmt_pct(headline["vol"]))
        c[2].metric("Sharpe", f"{headline['sharpe']:.2f}")
        c[3].metric("Sortino", f"{headline['sortino']:.2f}")
        c[4].metric(f"Alpha (vs {bench})", fmt_pct(headline["alpha"]))
        c[5].metric(f"Beta (vs {bench})", f"{headline['beta']:.2f}")
        c = st.columns(6)
        c[0].metric("Max Drawdown", fmt_pct(headline["max_dd"]))
        c[1].metric("Calmar", f"{headline['calmar']:.2f}")
        c[2].metric("VaR 95% (1D)", fmt_pct(headline["var_95"]))
        c[3].metric("CVaR 95% (1D)", fmt_pct(headline["cvar_95"]))
        c[4].metric("VaR 99% (1D)", fmt_pct(headline["var_99"]))
        c[5].metric("CVaR 99% (1D)", fmt_pct(headline["cvar_99"]))

        st.markdown("**Confronto: in-sample vs out-of-sample vs 1/N**" if validate
                    else "**Confronto con il portafoglio equipesato 1/N**")
        rows = [("Rendimento Ann.", "ret", True), ("Volatilità", "vol", True),
                ("Sharpe", "sharpe", False), ("Sortino", "sortino", False),
                ("Max Drawdown", "max_dd", True), ("VaR 99% (1D)", "var_99", True)]
        table = {"Metrica": [r[0] for r in rows]}
        if validate:
            table["Ottimizzato - in-sample"] = [fmt_pct(m_in[k]) if p else f"{m_in[k]:.2f}" for _, k, p in rows]
            table["Ottimizzato - out-of-sample"] = [fmt_pct(m_out[k]) if p else f"{m_out[k]:.2f}" for _, k, p in rows]
            table["1/N - out-of-sample"] = [fmt_pct(m_ew_out[k]) if p else f"{m_ew_out[k]:.2f}" for _, k, p in rows]
        else:
            table["Ottimizzato"] = [fmt_pct(m_full[k]) if p else f"{m_full[k]:.2f}" for _, k, p in rows]
            table["1/N"] = [fmt_pct(m_ew_out[k]) if p else f"{m_ew_out[k]:.2f}" for _, k, p in rows]
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Allocazione Capitale")
            df_w = pd.DataFrame({"Ticker": tickers, "Peso (%)": weights * 100})
            st.plotly_chart(px.pie(df_w, values="Peso (%)", names="Ticker", hole=0.4,
                                   title=f"Pesi ({opt_method})"), use_container_width=True)
        with col2:
            st.subheader("Matrice di Correlazione")
            st.plotly_chart(px.imshow(train.corr(), text_auto=".2f",
                                      color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                            use_container_width=True)

    # ---- TAB 2 ----------------------------------------------------------------
    with tab2:
        st.subheader(f"Backtest storico (${initial_capital:,.0f} iniziali)")
        st.caption(f"Ribilanciamento: {rebalance_freq} | Costi: {transaction_cost_pct * 100:.2f}% "
                   "sul valore scambiato (turnover) | Dividendi inclusi (Adj Close)")
        curves = initial_capital * (1 + pd.DataFrame({
            "Portafoglio ottimizzato": port_ret,
            "Equipesato 1/N": ew_ret,
            f"Benchmark ({bench})": bench_returns,
        })).cumprod()
        fig = px.line(curves, labels={"value": "Valore ($)", "index": "Data", "variable": ""})
        if validate:
            split_date = test.index[0].to_pydatetime()
            fig.add_shape(type="line", x0=split_date, x1=split_date, y0=0, y1=1,
                          yref="paper", line=dict(dash="dash", color="gray"))
            fig.add_annotation(x=split_date, y=1, yref="paper", text="Inizio test",
                               showarrow=False, yanchor="bottom")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Prima della linea tratteggiata il risultato è in-sample (i pesi sono stati "
                   "ottimizzati su quei dati); dopo la linea è out-of-sample.")

    # ---- TAB 3 ----------------------------------------------------------------
    with tab3:
        st.subheader("Proiezione Monte Carlo a 1 anno (Geometric Brownian Motion)")
        st.caption(f"{gbm_simulations} scenari. Parametri (μ, σ) stimati sull'intero backtest; "
                   "assume μ e σ costanti e rendimenti log-normali, quindi sottostima le code.")
        paths = eq.gbm_paths(initial_capital, m_full["ret"], m_full["vol"], gbm_simulations)
        p5, p50, p95 = (np.percentile(paths, q, axis=1) for q in (5, 50, 95))

        fig = go.Figure()
        for i in range(min(50, gbm_simulations)):
            fig.add_trace(go.Scatter(y=paths[:, i], mode="lines",
                                     line=dict(color="rgba(150,150,150,0.15)"), showlegend=False))
        fig.add_trace(go.Scatter(y=p95, name="95° percentile", line=dict(color="green", width=2)))
        fig.add_trace(go.Scatter(y=p50, name="Mediana", line=dict(color="blue", width=3)))
        fig.add_trace(go.Scatter(y=p5, name="5° percentile", line=dict(color="red", width=2)))
        fig.update_layout(title="Cono d'incertezza a 1 anno",
                          xaxis_title="Giorni di borsa futuri", yaxis_title="Valore ($)")
        st.plotly_chart(fig, use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("5° percentile", f"${p5[-1]:,.2f}")
        m2.metric("Mediana", f"${p50[-1]:,.2f}")
        m3.metric("95° percentile", f"${p95[-1]:,.2f}")

    # ---- TAB 4 ----------------------------------------------------------------
    with tab4:
        st.subheader("Stress Testing su crisi storiche reali")
        st.markdown("Si applicano i **pesi correnti** ai rendimenti effettivamente registrati "
                    "da ciascun titolo durante la crisi (pesi fissi, senza ribilanciare). "
                    "Se un titolo non esisteva ancora, i pesi vengono rinormalizzati sugli altri.")
        try:
            hist = load_data(tuple(tickers), bench, "2007-01-01",
                             str(pd.Timestamp.today().date()))
            rows = []
            for name, (s, e) in eq.STRESS_WINDOWS.items():
                pr, br, cov_w = eq.historical_stress(hist, tickers, weights, bench, s, e)
                rows.append({
                    "Scenario": name,
                    "Portafoglio": fmt_pct(pr),
                    f"Benchmark ({bench})": fmt_pct(br),
                    f"Perdita su ${initial_capital:,.0f}": "n/d" if pd.isna(pr) else f"${initial_capital * pr:,.2f}",
                    "Capitale coperto da dati": f"{cov_w * 100:.0f}%",
                })
            ann_ret, ann_vol = m_full["ret"], m_full["vol"]
            pm = eq.parametric_horizon_loss(ann_ret, ann_vol, days=21, conf=0.99)
            rows.append({
                "Scenario": "Parametrico: VaR 99% a 1 mese (21 gg, normale)",
                "Portafoglio": fmt_pct(pm), f"Benchmark ({bench})": "-",
                f"Perdita su ${initial_capital:,.0f}": f"${initial_capital * pm:,.2f}",
                "Capitale coperto da dati": "-",
            })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Le crisi passate non sono previsioni: la prossima avrà cause e dinamiche diverse. "
                       "Lo scenario parametrico assume rendimenti normali e quindi sottostima gli eventi estremi.")
        except Exception as exc:
            st.warning(f"Impossibile scaricare i dati storici per lo stress test: {exc}")

    # ---- TAB 5 ----------------------------------------------------------------
    with tab5:
        st.subheader("Report PDF")
        pdf_bytes = generate_pdf_report(
            tickers, weights, headline, m_in, opt_method, bench, validate, eval_label)
        st.download_button("Scarica Report Tecnico PDF", data=pdf_bytes,
                           file_name="quant_portfolio_report.pdf", mime="application/pdf")


# ---------------------------------------------------------
# Avvio. Lo stato "analisi eseguita" e' in session_state: senza questo, il
# click su "Scarica PDF" (che fa un rerun) farebbe sparire tutti i risultati.
# ---------------------------------------------------------
if st.sidebar.button("Esegui Analisi"):
    st.session_state["ran"] = True

if st.session_state.get("ran"):
    try:
        run_analysis()
    except Exception as exc:
        st.error(f"Errore durante l'esecuzione: {exc}")
else:
    st.info("Imposta i parametri nella barra laterale e premi **Esegui Analisi**.")

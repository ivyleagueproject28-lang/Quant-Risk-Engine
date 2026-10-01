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
"""
engine.py - Logica quantitativa dell'app (solo numpy / pandas / scipy).

Nessuna dipendenza da Streamlit: ogni funzione e' testabile da sola.
Convenzioni:
  - rendimenti giornalieri semplici (pct_change)
  - annualizzazione: media * 252, deviazione standard * sqrt(252)
  - pesi long-only che sommano a 1
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import norm

TRADING_DAYS = 252

REBALANCE_OPTIONS = [
    "Nessuno (Buy & Hold)",
    "Giornaliero (pesi costanti)",
    "Mensile",
    "Trimestrale",
    "Annuale",
]
_PERIOD_CODE = {"Mensile": "M", "Trimestrale": "Q", "Annuale": "Y"}


# ---------------------------------------------------------------------------
# Utilita'
# ---------------------------------------------------------------------------
def _clean_weights(w):
    """Elimina i piccoli negativi numerici di SLSQP e rinormalizza a 1."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    return w / w.sum()


def portfolio_vol(w, cov):
    return float(np.sqrt(w @ cov @ w))


def split_train_test(returns: pd.DataFrame, train_frac: float):
    """Divisione cronologica (mai casuale: i dati sono serie storiche)."""
    cut = int(len(returns) * train_frac)
    return returns.iloc[:cut], returns.iloc[cut:]


# ---------------------------------------------------------------------------
# Algoritmi di ottimizzazione
# ---------------------------------------------------------------------------
def max_sharpe_monte_carlo(mean_returns, cov, rf, n_sims=10000, seed=42):
    """
    Pesi casuali UNIFORMI sul simplesso (Dirichlet(1,...,1)), poi si sceglie
    il portafoglio con Sharpe massimo. E' un'approssimazione: piu' simulazioni
    = piu' vicino all'ottimo vero, ma non lo raggiunge mai esattamente.
    """
    rng = np.random.default_rng(seed)
    n = len(mean_returns)
    w = rng.dirichlet(np.ones(n), size=n_sims)
    rets = w @ mean_returns
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    sharpes = (rets - rf) / vols
    return w[np.argmax(sharpes)]


def max_sharpe_exact(mean_returns, cov, rf):
    """Massimizza lo Sharpe con SLSQP (soluzione esatta, long-only)."""
    n = len(mean_returns)

    def neg_sharpe(w):
        return -(w @ mean_returns - rf) / portfolio_vol(w, cov)

    res = minimize(
        neg_sharpe, np.ones(n) / n, method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},),
    )
    return _clean_weights(res.x)


def min_variance_weights(cov):
    """Minimizza w' S w con somma pesi = 1 e pesi >= 0 (SLSQP)."""
    n = cov.shape[0]
    res = minimize(
        lambda w: w @ cov @ w, np.ones(n) / n, method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},),
    )
    return _clean_weights(res.x)


def risk_parity_weights(cov):
    """
    Ogni asset contribuisce in modo uguale al rischio totale.
    Contributo relativo: RC_i = w_i * (S w)_i / (w' S w)  (somma = 1).
    Si minimizza la distanza da 1/n. Usare i contributi RELATIVI evita che
    l'obiettivo sia ~1e-7 (varianze al quadrato) e SLSQP si fermi subito.
    """
    n = cov.shape[0]

    def objective(w):
        port_var = w @ cov @ w
        rc = w * (cov @ w) / port_var
        return np.sum((rc - 1.0 / n) ** 2)

    res = minimize(
        objective, np.ones(n) / n, method="SLSQP",
        bounds=[(1e-6, 1)] * n,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},),
        options={"ftol": 1e-12, "maxiter": 500},
    )
    return _clean_weights(res.x)


def hrp_weights(returns: pd.DataFrame):
    """
    Hierarchical Risk Parity (Lopez de Prado, 2016), 3 passi:
      1. Clustering gerarchico sulla distanza d = sqrt(0.5 * (1 - rho)).
      2. Quasi-diagonalizzazione: si riordinano gli asset secondo le foglie
         del dendrogramma (asset simili diventano adiacenti).
      3. Bisezione ricorsiva: si divide la lista a meta' e il capitale
         viene ripartito in proporzione inversa alla varianza di ciascun
         cluster (calcolata con pesi a inverso-varianza dentro il cluster).
    """
    n = returns.shape[1]
    if n < 2:
        return np.ones(n)

    cov = returns.cov().values
    corr = np.clip(returns.corr().values, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0            # simmetria numerica
    link = linkage(squareform(dist, checks=False), method="single")
    order = list(leaves_list(link))          # quasi-diagonalizzazione

    def cluster_var(idx):
        sub = cov[np.ix_(idx, idx)]
        ivp = 1.0 / np.diag(sub)
        ivp /= ivp.sum()
        return float(ivp @ sub @ ivp)

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [
            c[i:j]
            for c in clusters if len(c) > 1
            for i, j in ((0, len(c) // 2), (len(c) // 2, len(c)))
        ]
        for k in range(0, len(clusters), 2):
            c0, c1 = clusters[k], clusters[k + 1]
            v0, v1 = cluster_var(c0), cluster_var(c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w.loc[c0] *= alpha
            w.loc[c1] *= 1.0 - alpha

    return w.sort_index().values


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def rebalance_mask(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """True nell'ULTIMO GIORNO DI BORSA APERTA di ogni periodo (non l'ultimo
    giorno di calendario, che puo' essere weekend/festivo)."""
    if freq == "Nessuno (Buy & Hold)":
        return np.zeros(len(index), dtype=bool)
    if freq.startswith("Giornaliero"):
        return np.ones(len(index), dtype=bool)
    periods = index.to_period(_PERIOD_CODE[freq])
    last_day = pd.Series(index, index=index).groupby(periods).transform("max")
    return (last_day.values == index.values)


def run_backtest(daily_returns: pd.DataFrame, target_weights, freq: str,
                 tx_cost: float) -> pd.Series:
    """
    Simulazione con pesi che DERIVANO coi prezzi:
      - ogni giorno: r_p = w . r_t, poi w_i <- w_i (1+r_i) / (1+r_p)
      - nei giorni di ribilanciamento si torna ai pesi target a fine giornata
        pagando  costo = tx_cost * sum|w_derivati - w_target|.
    Con "Nessuno" i pesi non vengono mai riportati al target (vero Buy & Hold).
    """
    r = daily_returns.values
    target = np.asarray(target_weights, dtype=float)
    mask = rebalance_mask(daily_returns.index, freq)

    w = target.copy()
    out = np.empty(len(r))
    for t in range(len(r)):
        gross = float(w @ r[t])
        w = w * (1.0 + r[t]) / (1.0 + gross)
        cost = 0.0
        if mask[t]:
            cost = tx_cost * float(np.abs(w - target).sum())
            w = target.copy()
        out[t] = gross - cost
    return pd.Series(out, index=daily_returns.index)


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------
def calculate_metrics(port: pd.Series, bench: pd.Series, rf: float) -> dict:
    df = pd.concat([port, bench], axis=1, keys=["p", "b"]).dropna()
    p, b = df["p"], df["b"]

    ann_ret = p.mean() * TRADING_DAYS
    ann_vol = p.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
    cagr = (1 + p).prod() ** (TRADING_DAYS / len(p)) - 1

    # Sortino: downside deviation vs MAR = rf (giornaliero), su TUTTI i giorni
    # (i giorni sopra la soglia contano come 0), non std dei soli giorni negativi.
    shortfall = np.minimum(p - rf / TRADING_DAYS, 0.0)
    downside_dev = np.sqrt((shortfall ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = (ann_ret - rf) / downside_dev if downside_dev > 0 else 0.0

    cum = (1 + p).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    var_95 = np.percentile(p, 5)
    var_99 = np.percentile(p, 1)
    cvar_95 = p[p <= var_95].mean()
    cvar_99 = p[p <= var_99].mean()

    c = np.cov(p, b)                       # ddof=1 sia per cov sia per var
    beta = c[0, 1] / c[1, 1] if c[1, 1] > 0 else 1.0
    alpha = ann_ret - (rf + beta * (b.mean() * TRADING_DAYS - rf))

    return {
        "ret": ann_ret, "cagr": cagr, "vol": ann_vol, "sharpe": sharpe,
        "sortino": sortino, "calmar": calmar, "max_dd": max_dd,
        "var_95": var_95, "cvar_95": cvar_95,
        "var_99": var_99, "cvar_99": cvar_99,
        "alpha": alpha, "beta": beta,
    }


# ---------------------------------------------------------------------------
# Proiezione GBM
# ---------------------------------------------------------------------------
def gbm_paths(initial, mu, sigma, n_sims, days=TRADING_DAYS, seed=42):
    """
    S_t = S_0 * exp( (mu - sigma^2/2) t + sigma W_t )
    mu e' il rendimento ATTESO aritmetico annuo; il termine -sigma^2/2 e' la
    correzione di Ito, perche' E[exp(sigma W_t)] = exp(sigma^2 t / 2).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_normal((days, n_sims))
    log_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    growth = np.exp(np.cumsum(log_ret, axis=0))
    return initial * np.vstack([np.ones(n_sims), growth])


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------
STRESS_WINDOWS = {
    "Crisi finanziaria (9 ott 2007 - 9 mar 2009)": ("2007-10-09", "2009-03-09"),
    "COVID-19 sell-off (19 feb - 23 mar 2020)": ("2020-02-19", "2020-03-23"),
    "Bear market tassi/inflazione (3 gen - 12 ott 2022)": ("2022-01-03", "2022-10-12"),
    "Crisi bancaria SVB (8 - 15 mar 2023)": ("2023-03-08", "2023-03-15"),
}


def _window_return(series: pd.Series, start, end, tol_days=7):
    """Rendimento totale in una finestra; NaN se la serie non esisteva
    all'inizio della finestra (es. titolo quotato dopo)."""
    s = series.loc[start:end].dropna()
    if len(s) < 2 or s.index[0] > pd.Timestamp(start) + pd.Timedelta(days=tol_days):
        return np.nan
    return s.iloc[-1] / s.iloc[0] - 1.0


def historical_stress(prices: pd.DataFrame, tickers, weights, bench: str,
                      start, end):
    """
    Rendimento che avrebbe avuto il portafoglio (pesi fissi all'inizio,
    senza ribilanciare) nella finestra storica REALE.
    Se alcuni titoli non esistevano, i pesi vengono rinormalizzati sui
    disponibili e si restituisce la quota di capitale coperta.
    Ritorna (rend_portafoglio, rend_benchmark, copertura).
    """
    w = np.asarray(weights, dtype=float)
    rets = np.array([_window_return(prices[t], start, end) for t in tickers])
    ok = ~np.isnan(rets)
    bench_ret = _window_return(prices[bench], start, end)
    if not ok.any():
        return np.nan, bench_ret, 0.0
    coverage = w[ok].sum()
    port_ret = float((w[ok] / coverage) @ rets[ok])
    return port_ret, bench_ret, float(coverage)


def parametric_horizon_loss(ann_ret, ann_vol, days=21, conf=0.99):
    """Rendimento al percentile (1-conf) su `days` giorni assumendo
    rendimenti normali: mu*T - z * sigma * sqrt(T)."""
    t = days / TRADING_DAYS
    return ann_ret * t - norm.ppf(conf) * ann_vol * np.sqrt(t)
  """report.py - Report PDF (fpdf2). I font base di FPDF supportano solo latin-1."""
from fpdf import FPDF
from fpdf.enums import XPos, YPos

NAVY = (24, 43, 73)
NEXT = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)   # a capo dopo la cella
RIGHT = dict(new_x=XPos.RIGHT, new_y=YPos.TOP)     # resta sulla stessa riga


def _t(s) -> str:
    """Rende sicuro qualsiasi testo per i font latin-1."""
    return str(s).encode("latin-1", "replace").decode("latin-1")


class InstitutionalPDF(FPDF):
    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 14, "F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 3)
        self.cell(0, 8, "QUANTITATIVE RISK & PORTFOLIO ENGINE - REPORT TECNICO",
                  align="L")
        self.ln(14)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Pagina {self.page_no()}", align="C")


def _section(pdf, title):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, _t(title), **NEXT)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)


def _metrics_table(pdf, metrics, bench_name):
    pct = lambda x: f"{x * 100:.2f}%"
    rows = [
        ("Rendimento annualizzato", pct(metrics["ret"]), "Volatilita annualizzata", pct(metrics["vol"])),
        ("Sharpe Ratio", f"{metrics['sharpe']:.2f}", "Sortino Ratio", f"{metrics['sortino']:.2f}"),
        (f"Alpha di Jensen (vs {bench_name})", pct(metrics["alpha"]), "Beta", f"{metrics['beta']:.2f}"),
        ("Max Drawdown", pct(metrics["max_dd"]), "Calmar Ratio", f"{metrics['calmar']:.2f}"),
        ("VaR 95% (1 giorno)", pct(metrics["var_95"]), "CVaR 95% (1 giorno)", pct(metrics["cvar_95"])),
        ("VaR 99% (1 giorno)", pct(metrics["var_99"]), "CVaR 99% (1 giorno)", pct(metrics["cvar_99"])),
    ]
    pdf.set_text_color(40, 40, 40)
    for a, b, c, d in rows:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(55, 6, _t(a + ":"), **RIGHT)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(35, 6, _t(b), **RIGHT)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(50, 6, _t(c + ":"), **RIGHT)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(40, 6, _t(d), **NEXT)
    pdf.ln(4)


def generate_pdf_report(tickers, weights, metrics_eval, metrics_in, method,
                        bench_name, validated, eval_label):
    pdf = InstitutionalPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 10, "Analisi di Portafoglio e Gestione del Rischio", **NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, _t(f"Algoritmo: {method}"), **NEXT)
    pdf.cell(0, 6, _t(f"Validazione: {'out-of-sample (train/test)' if validated else 'NESSUNA (tutti i dati in-sample)'}"), **NEXT)
    pdf.ln(4)

    _section(pdf, f"1. Metriche - {eval_label}")
    _metrics_table(pdf, metrics_eval, bench_name)

    if validated:
        _section(pdf, "2. Metriche in-sample (periodo di ottimizzazione)")
        _metrics_table(pdf, metrics_in, bench_name)
        n = 3
    else:
        n = 2

    _section(pdf, f"{n}. Allocazione del capitale")
    pdf.set_fill_color(240, 243, 246)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.cell(90, 7, " Asset", border=1, fill=True, **RIGHT)
    pdf.cell(90, 7, " Peso (%)", border=1, fill=True, **NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    for t, w in zip(tickers, weights):
        pdf.cell(90, 6, _t(f" {t}"), border=1, **RIGHT)
        pdf.cell(90, 6, f" {w * 100:.2f}%", border=1, **NEXT)
    pdf.ln(6)

    _section(pdf, f"{n + 1}. Limiti del modello")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    notes = [
        "I rendimenti attesi stimati sul passato sono molto rumorosi: i pesi ottimali sono instabili.",
        "VaR e CVaR sono storici: assumono che il futuro assomigli al campione osservato.",
        "I titoli analizzati sono stati scelti a posteriori (survivorship/selection bias).",
        "I costi di transazione sono un'approssimazione proporzionale al turnover.",
        "Le performance passate non garantiscono risultati futuri.",
    ]
    for line in notes:
        pdf.multi_cell(0, 5, _t("- " + line), **NEXT)

    return bytes(pdf.output())

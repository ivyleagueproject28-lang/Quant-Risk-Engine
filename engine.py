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

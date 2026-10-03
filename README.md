# Covariance shrinkage and the efficient frontier

Walk-forward comparison of sample covariance, Ledoit-Wolf and OAS shrinkage in long-only
mean-variance optimization on 40 ETFs, with turnover limits, transaction costs and a
statistical test on the Sharpe differences.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ibrahim-wq-sys/shrinkage-frontier/blob/main/notebooks/shrinkage_frontier.ipynb)

| | |
|---|---|
| Difficulty | Intermediate |
| Time | 8 to 12 hours |
| Stack | cvxpy, scikit-learn, numpy, pandas, plotly, yfinance |
| Concepts | Markowitz, estimation error, condition number, quadratic programming, shrinkage, walk-forward testing |

---

## 1. Project overview

Markowitz optimization takes two inputs, expected returns and a covariance matrix, and
returns the portfolio with the best risk-return trade-off. In practice both inputs are
estimated from a short window of noisy data. The optimizer cannot tell signal from noise.
It finds directions in the covariance matrix that look nearly riskless and puts large weights
on them. Most of those directions are estimation error. Michaud (1989) called mean-variance
optimizers "estimation-error maximizers" for this reason.

This project measures how much that matters, and how much shrinkage fixes it:

1. Download daily prices for 40 ETFs covering US and international equity, sectors, REITs,
   Treasuries, credit and commodities.
2. Estimate expected returns (James-Stein shrunk means, held fixed across all strategies) and
   three covariance matrices: sample, Ledoit-Wolf and OAS.
3. Solve long-only minimum-variance and maximum-Sharpe portfolios with a 25% position cap and
   a turnover limit, using cvxpy.
4. Rebalance monthly in a walk-forward loop with no look-ahead, charging 10 bps per trade.
5. Compare realized Sharpe, volatility, drawdown, turnover and risk-forecast accuracy, and test
   whether the Sharpe differences are statistically significant.

Every strategy uses the same expected returns, constraints, rebalance dates and costs. The
only thing that changes is the covariance estimator, so any performance gap comes from the
risk model.

## 2. Real-world finance use case

| Who | How they use this |
|---|---|
| Multi-asset allocation desks | Run a constrained optimizer every month or quarter to set strategic and tactical weights. The covariance estimator decides how stable those weights are. |
| Risk teams | Forecast portfolio volatility for limits and VaR. An ill-conditioned covariance matrix under-forecasts the risk of optimized portfolios, which this project measures directly. |
| Quant equity and smart-beta | Minimum-variance and risk-parity indices (MSCI Min Vol, for example) depend on a stable covariance estimate across hundreds of names. |
| Robo-advisors and wealth platforms | Optimize ETF model portfolios under long-only, position-cap and turnover constraints. That is the exact problem set up here. |

Ledoit-Wolf (2003, 2004) style shrinkage, or a factor model that does the same job, is the
default covariance input at most systematic shops. Raw sample covariance is rarely used for
optimization once N is more than a handful of assets.

## 3. System architecture

```
┌────────────────────┐   ┌─────────────────────┐   ┌────────────────────────┐
│ Data collection    │──▶│ Cleaning & features │──▶│ Estimators             │
│ yfinance + retries │   │ missing data, ffill │   │ μ: James-Stein         │
│ CSV cache          │   │ returns, rf, dates  │   │ Σ: sample / LW / OAS   │
│ synthetic fallback │   └─────────────────────┘   │ condition number       │
└────────────────────┘                             └───────────┬────────────┘
                                                               │
                ┌──────────────────────────────────────────────▼─────────────┐
                │ Walk-forward engine (monthly)                              │
                │  for each rebalance date t:                                │
                │    window = returns[t-T+1 .. t]                            │
                │    estimate μ once, Σ per estimator                        │
                │    solve min-var / max-Sharpe  ◀── PortfolioOptimizer      │
                │    (cvxpy, DPP-compiled once, turnover vs drifted weights) │
                │    charge costs, hold to t+1 with daily drift              │
                └──────────────────────────────────────┬─────────────────────┘
                                                       │
        ┌──────────────────────┬───────────────────────┼──────────────────────┐
        ▼                      ▼                       ▼                      ▼
┌───────────────┐   ┌────────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ Frontier study│   │ Metrics            │   │ Sensitivity      │   │ Dashboard export │
│ random ports  │   │ Sharpe, DD, TO,    │   │ T/N sweep,       │   │ report.html,     │
│ promised vs   │   │ risk ratio, JK test│   │ equity sleeve    │   │ CSVs, PNGs       │
│ realized risk │   └────────────────────┘   └──────────────────┘   └──────────────────┘
└───────────────┘
```

## 4. Required APIs and data sources

| Source | What | Access |
|---|---|---|
| Yahoo Finance via `yfinance` | Daily adjusted close (splits and dividends) for 40 ETFs, Jan 2008 onward | Free, no key. Rate-limited, so the code retries with exponential backoff and caches to CSV. |
| Yahoo `^IRX` | 13-week T-bill yield, used as the risk-free rate | Same. Falls back to a constant 1.5% if unavailable. |
| Synthetic generator (built in) | 5-factor model with stochastic volatility and Student-t shocks | Used only if Yahoo is unreachable, so the pipeline always runs. Every chart footer states the data source. |

## 5. Required Python libraries

| Library | Used for |
|---|---|
| `cvxpy` | Quadratic programs: min variance, max Sharpe (Cornuejols-Tutuncu transform), frontier. Solvers: CLARABEL, then OSQP, then SCS. |
| `scikit-learn` | `LedoitWolf` and `OAS` covariance estimators |
| `numpy` | Linear algebra, Cholesky factors, eigenvalues, vectorized backtest accounting |
| `pandas` | Time-series alignment, rebalance calendar, metrics tables |
| `plotly` | Interactive charts and the HTML dashboard |
| `yfinance` | Market data |
| `kaleido` (optional) | PNG export of the figures for the README |
| `nbformat` (optional) | Builds the Colab notebook from the source file |

## 6. Folder and file structure

```
shrinkage-frontier/
├── README.md                     # this file
├── requirements.txt
├── notebooks/
│   └── shrinkage_frontier.ipynb  # Colab notebook, 16 sections (generated from src/)
├── src/
│   └── shrinkage_frontier.py     # single source of truth, percent-format cells (# %%)
├── tools/
│   └── build_notebook.py         # src/*.py  ->  notebooks/*.ipynb
├── data/                         # CSV cache: prices.csv, risk_free.csv (git-ignored)
└── outputs/
    ├── report.html               # self-contained interactive dashboard
    ├── metrics.csv               # raw metrics, one row per strategy
    ├── metrics_formatted.csv
    ├── daily_returns.csv         # net daily returns per strategy
    ├── weights_<est>_<obj>.csv   # target weights at every rebalance
    ├── diagnostics_<est>_<obj>.csv  # turnover, κ, δ, predicted vs realized vol, solver status
    ├── lookback_sensitivity.csv
    ├── data_quality_report.csv
    └── figures/*.png             # if kaleido is installed
```

The `.py` file runs as a plain script (`python src/shrinkage_frontier.py`) and also opens as
a notebook in VS Code or Jupytext. Edit the `.py` file, then run `python tools/build_notebook.py`
to regenerate the `.ipynb`, so the two never drift apart.

## 7. Step-by-step build guide

| Step | Hours | What to do | Done when |
|---|---|---|---|
| 1. Setup | 0.5 | New Colab notebook, install cvxpy and yfinance, write the `Config` dataclass | `CFG.validate()` passes |
| 2. Universe and data | 1.0 | Pick 40 ETFs with history back to 2008. Write `download_prices` with retries and a CSV cache. | Price panel loads twice, the second time from cache |
| 3. Cleaning | 1.0 | Missing-data filter, common start date, limited forward-fill, outlier flags, quality report | No NaNs, report shows each ticker's coverage |
| 4. Features | 0.5 | Daily returns, daily risk-free rate, month-end rebalance positions | Rebalance count about equal to months minus lookback |
| 5. Estimators | 1.0 | Sample, Ledoit-Wolf and OAS covariance, James-Stein means, condition number | LW condition number well below sample on the latest window |
| 6. Optimizer | 2.0 | cvxpy min-var and max-Sharpe with parameters (DPP), turnover constraint, solver fallback chain | Weights sum to 1, respect caps, `is_dpp()` is True |
| 7. Frontier | 1.0 | Parametric frontier solver, random portfolios, promised vs realized risk | Frontier lies above every random portfolio |
| 8. Backtest | 2.0 | Walk-forward loop with drift, costs, turnover vs drifted weights, diagnostics log | 214 rebalances, no look-ahead (window ends at t, holding starts t+1) |
| 9. Metrics | 1.0 | Sharpe, Sortino, drawdown, turnover, effective N, risk forecast ratio, Jobson-Korkie-Memmel test | Metrics table for every strategy plus benchmarks |
| 10. Visuals and export | 1.5 | Nine charts, HTML dashboard, CSV export | `report.html` opens offline |
| 11. Sensitivity and write-up | 1.0 | T/N sweep on the equity sleeve, README results | Gap between estimators narrows as T/N rises |

Total: about 12.5 hours with the sensitivity study and write-up, about 10 without.

To run it:

```bash
# Colab: open notebooks/shrinkage_frontier.ipynb and Runtime > Run all (about 2 minutes)

# Local
pip install -r requirements.txt
python src/shrinkage_frontier.py          # writes outputs/report.html and CSVs
python tools/build_notebook.py            # regenerate the notebook after editing src/
```

## 8. Data collection pipeline

```
load_market_data()
 ├─ cache hit (data/prices.csv younger than 3 days, all tickers present) → return
 ├─ yfinance.download(40 tickers, auto_adjust=True)      3 attempts, backoff 2s/4s/8s
 │   └─ ^IRX download → rf = yield / 100                 2 attempts, else constant 1.5%
 │   └─ write cache
 └─ DataDownloadError and allow_synthetic_fallback → simulate_prices()
```

Design choices:

- `auto_adjust=True` gives total-return prices, so dividends from bond and REIT ETFs are
  counted. Without it, TLT and VNQ look far worse than they were.
- Tickers that fail individually are logged and dropped. They don't kill the run.
- The synthetic generator is there so the code can be tested anywhere, including CI and
  firewalled sandboxes. It is not a substitute for real data, and the dashboard says which
  source produced it.

## 9. Data cleaning and feature engineering

| Step | Rule | Why |
|---|---|---|
| De-duplicate, sort, drop weekends | Keep last duplicate | yfinance occasionally returns duplicate rows around DST changes |
| Non-positive prices | Set to NaN | Bad prints |
| Sparse tickers | Drop if more than 5% missing | One late-listing ETF would otherwise shorten the whole sample |
| Common start | First date every survivor has a price | The covariance matrix needs a full panel |
| Forward-fill | At most 5 days | Covers exchange holidays. Longer gaps stay as gaps and the row is dropped. |
| Outliers | Flag moves above 40%, zero moves above 75% | Adjusted ETF data rarely has real 75% daily moves. USO in April 2020 is close, so check the flag list. |
| Returns | Simple daily returns | Portfolio returns aggregate linearly in simple returns |
| Risk-free | `(1 + y)^(1/252) - 1`, forward-filled to trading days | Daily excess returns for Sharpe |
| Rebalance calendar | Last trading day of each month, after a full lookback | No partial windows, no look-ahead |

## 10. Core models and algorithms

**Covariance estimators.** With N = 40 and T = 252, the sample covariance has 820 free
parameters estimated from about 10,000 numbers. Its eigenvalues are biased: the largest come
out too large and the smallest too small (Marchenko-Pastur). The condition number
κ = λ_max/λ_min is a direct measure of how unstable the inverse, and therefore the optimal
weights, will be.

- **Sample:** `S = 1/(T-1) Σ (r_t - r̄)(r_t - r̄)'`
- **Ledoit-Wolf:** `Σ_LW = δ F + (1 - δ) S`, with δ chosen analytically to minimise expected
  Frobenius loss.
- **OAS:** same form, with a δ that converges faster under Gaussian returns.

*Target choice.* scikit-learn's target is F = λ̄ I, the same variance for every asset. In a
multi-asset universe that is a poor target: SHY's variance is about 1/100th of EWZ's. With
the identity target, Ledoit-Wolf min-variance portfolios came out *worse* than sample in
testing. The code therefore runs the estimators on standardised returns and rescales,
`Σ = D · LW(corr) · D`. Each asset keeps its own variance and only the correlations shrink.
`Config.shrink_correlation = False` restores plain scikit-learn behaviour.

**Expected returns.** James-Stein shrinkage toward the cross-sectional mean:
`δ = min(1, (N + 2) / ((N + 2) + T · d'S⁻¹d))`, `d = m − m̄`. All strategies use the same μ so
the test isolates the covariance estimator.

**Minimum variance (QP):**
```
minimize    ‖U w‖²                 (U'U = Σ, Cholesky)
subject to  1'w = 1,  0 ≤ w ≤ 0.25,  ‖w − w_drifted‖₁ ≤ τ
```

**Maximum Sharpe.** The Sharpe ratio is a ratio, so it is not convex as stated. Substituting
y = k·w (Cornuejols and Tutuncu) gives a QP:
```
minimize    ‖U y‖²
subject to  (μ − r_f)'y = 1,  1'y = k,  0 ≤ y ≤ 0.25·k,  ‖y − k·w_drifted‖₁ ≤ τ·k,  k ≥ 0
w* = y*/k*
```
Every constraint is homogeneous in (y, k), so the position cap and turnover limit carry over
exactly. If no capped portfolio has positive expected excess return, the problem is
infeasible. The month then falls back to minimum variance and the fallback is logged.

**Implementation details:**

- Problems are built once with `cp.Parameter` and re-solved with new data (DPP). That avoids
  recompiling about 1,300 times and keeps the full backtest to around 15 seconds.
- Risk is written as `sum_squares(U @ w)` with a Cholesky factor rather than `quad_form`. It
  is DPP-compliant and numerically better behaved. Near-singular matrices are repaired by
  clipping eigenvalues.
- Solver chain: CLARABEL, then OSQP, then SCS. If the turnover limit makes a problem
  infeasible (drifted weights above the cap after a large move), the solve is retried without
  it and the event is recorded. If every attempt fails, the strategy holds its drifted weights
  for the month rather than crashing.
- Holding-period accounting is exact: `value_t = (Π(1 + r)) · w`, so daily portfolio returns
  and end-of-month drifted weights come from one cumulative product.

**Statistical test.** The Jobson-Korkie test with the Memmel (2003) correction, on daily
excess returns, gives a z-score and p-value for H0: SR_shrinkage = SR_sample.

## 11. Visualizations and dashboard components

| # | Chart | What it shows |
|---|---|---|
| 1 | Efficient frontier with 15,000 random portfolios | Classic frontier from each estimator, ETFs as points, random portfolios coloured by Sharpe, min-var ◆ and max-Sharpe ★ markers |
| 2 | Promised vs realized risk | Frontier portfolios formed on one year and held for the next. Dashed is ex-ante vol, solid is realized. The horizontal gap is estimation error. |
| 3 | Eigenvalue spectrum | Sorted correlation-matrix eigenvalues for each estimator against the Marchenko-Pastur noise band |
| 4 | Conditioning through time | Condition number (log) and shrinkage intensity δ at every rebalance |
| 5 | Weight heatmaps | Max-Sharpe weights by month, sample vs Ledoit-Wolf, ETFs grouped by asset class |
| 6 | Cumulative return and drawdown | Growth of $1 (log) and drawdown, one panel per objective, equal weight as a dashed benchmark |
| 7 | Rolling 12-month Sharpe | Stability of risk-adjusted returns through time |
| 8 | Risk forecast ratio and turnover | Realized ÷ predicted vol (1.0 = perfect), and annual turnover |
| 9 | Metrics table | Full out-of-sample summary |
| 10 | T/N sensitivity | Realized vol and forecast ratio of min-variance as the lookback grows from 63 to 504 days |

`outputs/report.html` is self-contained (plotly.js embedded, about 10 MB). It opens offline
and can be attached to an email or published with GitHub Pages. Colours: Ledoit-Wolf is
blue, sample is orange, OAS is aqua, and benchmarks are grey dashed. The palette is
colour-blind safe for three series.

## 12. Performance metrics

| Metric | Definition | Read it as |
|---|---|---|
| CAGR | `wealth_T^(1/years) − 1` | Net of 10 bps costs |
| Volatility | `std(daily) · √252` | The quantity min-variance is trying to minimise |
| Sharpe | `mean(excess) / std(excess) · √252` | Excess over daily T-bill |
| Sortino | Annualised excess / downside deviation | Penalises losses only |
| Max drawdown, Calmar | Worst peak-to-trough, CAGR / abs(MDD) | Tail pain |
| Annual turnover | Σ‖w_new − w_drifted‖₁ / years | 100% means the whole book turns over once a year. Cost is turnover × bps. |
| Effective N | `1 / Σw²` averaged over rebalances | Diversification. 40 for equal weight. |
| Risk forecast ratio | median(realized vol / predicted vol) per holding period | Above 1 means the risk model was overconfident about the portfolio it chose |
| Condition number | κ of the implied correlation matrix | Numerical health of the optimizer input |
| JK p vs sample | Memmel-corrected Jobson-Korkie p-value | Below 0.05 means the Sharpe difference is unlikely to be noise |
| Fallbacks | Months where max-Sharpe fell back to min-var, turnover was relaxed, or a solve failed | Should be small. Check `diagnostics_*.csv` if not. |

### Results

> The table below comes from the **synthetic fallback data**. Yahoo Finance was blocked in
> the build environment. It confirms the pipeline works end to end. It says nothing about
> markets. Run the notebook in Colab to get real-data results and replace this table.

Main backtest: 40 assets, 252-day lookback, 25% cap, 50% turnover limit, 10 bps, Dec 2008 to Sep 2026, 214 rebalances.

| Strategy | CAGR | Vol | Sharpe | Max DD | Turnover/yr | κ (median) | JK p |
|---|---|---|---|---|---|---|---|
| Sample · Min variance | 2.46% | 3.70% | 0.26 | −10.2% | 107% | 620 | |
| Ledoit-Wolf · Min variance | 2.45% | 3.70% | 0.26 | −10.1% | 102% | 213 | 0.88 |
| OAS · Min variance | 2.46% | 3.70% | 0.26 | −10.1% | 105% | 375 | 0.71 |
| Sample · Max Sharpe | 0.94% | 7.22% | −0.05 | −29.5% | 450% | 620 | |
| Ledoit-Wolf · Max Sharpe | 0.86% | 7.27% | −0.06 | −30.1% | 447% | 213 | 0.57 |
| Equal weight | 2.63% | 13.69% | 0.14 | −53.3% | 36% | | |

Sensitivity study: equity and REIT sleeve (28 ETFs), long-only, no cap, no turnover limit.

| Lookback (T/N) | Realized vol: sample | Realized vol: LW | Turnover: sample | Turnover: LW |
|---|---|---|---|---|
| 63 (2.3) | 14.91% | 14.83% | 768% | 703% |
| 126 (4.5) | 14.72% | 14.67% | 435% | 409% |
| 252 (9.0) | 14.51% | 14.51% | 242% | 229% |
| 504 (18.0) | 14.74% | 14.73% | 123% | 119% |

### Findings from building it

These held during development and are worth checking against real data:

1. **Shrinkage cuts the condition number by about 3x** (620 to 213 on the correlation
   matrix) without changing the large eigenvalues that carry the real factor structure.
2. **Position caps act as shrinkage.** With a 25% cap and a turnover limit, the three
   estimators produce almost identical portfolios. Jagannathan and Ma (2003) proved that
   no-short-sale constraints are equivalent to shrinking the covariance matrix. The estimator
   matters most when constraints are loose.
3. **The benefit scales with N/T.** In the uncapped equity sleeve, Ledoit-Wolf lowers
   realized vol and cuts turnover by about 8% at T/N = 2.3. The gap closes by T/N = 18.
4. **The identity target hurts multi-asset portfolios.** scikit-learn's default target pulls
   SHY's variance up toward EWZ's. Shrinking correlations instead fixed it.
5. **Uncapped min-variance on a multi-asset universe is degenerate.** It puts about 95% in
   the lowest-vol asset (SHY) whatever the estimator, which is why the main backtest uses a cap.

## 13. Final deliverables

- `notebooks/shrinkage_frontier.ipynb`: Colab notebook, runs top to bottom in about 2 minutes
- `src/shrinkage_frontier.py`: the same code as a script, about 1,450 lines with docstrings
- `outputs/report.html`: interactive dashboard with KPI tiles and 10 charts
- `outputs/*.csv`: metrics, daily returns, weights and diagnostics for every strategy
- This README: methodology, results and findings

## 14. Resume description

**Covariance shrinkage in portfolio optimization** | Python, cvxpy, scikit-learn, plotly

- Built a walk-forward backtester for long-only min-variance and max-Sharpe portfolios on 40
  multi-asset ETFs (2008 to 2026, 214 monthly rebalances), comparing sample, Ledoit-Wolf and
  OAS covariance under position caps, turnover limits and transaction costs.
- Formulated max-Sharpe as a convex QP through the Cornuejols-Tutuncu transform and compiled
  both problems once with cvxpy parameters (DPP), solving about 1,300 problems in 15 seconds
  with a three-solver fallback chain.
- Cut the covariance condition number by about 3x with correlation-target shrinkage. Showed
  that sklearn's identity target hurts multi-asset portfolios, and that position caps close
  most of the gap between estimators (Jagannathan-Ma).
- Measured out-of-sample risk-forecast bias and tested Sharpe differences with the
  Memmel-corrected Jobson-Korkie test. Shipped results as a self-contained interactive
  dashboard.

Interview talking points: why sample covariance is ill-conditioned (eigenvalue bias,
Marchenko-Pastur); why max Sharpe needs a variable change to be convex; why turnover is
measured against drifted, not previous target, weights; why constraints act like shrinkage;
why μ is held fixed across strategies.

> After the real-data run, replace "about 3x" and "about 8%" with your own numbers.

## 15. Potential upgrades

| Upgrade | Effort | Why |
|---|---|---|
| Nonlinear shrinkage (Ledoit-Wolf 2017/2020) | Medium | Shrinks each eigenvalue separately. Usually beats linear shrinkage when N is large. |
| Constant-correlation and single-index targets (LW 2003, 2004) | Low | Compare targets directly. Write δ by hand in numpy. |
| Factor-model covariance (PCA or Fama-French) | Medium | The other standard way to tame Σ. Compare against shrinkage. |
| Exponentially weighted covariance | Low | Faster reaction to volatility regimes. Pairs well with shrinkage. |
| Black-Litterman expected returns | Medium | Replaces James-Stein μ with equilibrium plus views |
| Hierarchical Risk Parity (López de Prado) | Medium | No matrix inversion at all. A natural fourth estimator. |
| Resampled efficiency (Michaud) | Low | Bootstrap the frontier, average the weights |
| Transaction-cost term in the objective | Low | `+ λ‖w − w_prev‖₁` instead of a hard cap, then tune λ |
| Bootstrap confidence intervals on Sharpe | Low | Block bootstrap to complement the JK test |
| Larger universe (S&P 500 constituents) | Medium | N = 500 with T = 252 is where shrinkage matters most. Needs survivorship-bias-free data. |
| Streamlit app | Medium | Sliders for lookback, cap and turnover with live re-solve |
| Unit tests and CI | Low | pytest for optimizer feasibility, no look-ahead, weight sums; GitHub Actions on the synthetic data path |

## References

- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*.
- Ledoit, O. and Wolf, M. (2004). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management*.
- Chen, Y., Wiesel, A., Eldar, Y. and Hero, A. (2010). Shrinkage algorithms for MMSE covariance estimation. *IEEE Transactions on Signal Processing* (OAS).
- Jagannathan, R. and Ma, T. (2003). Risk reduction in large portfolios: why imposing the wrong constraints helps. *Journal of Finance*.
- DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive diversification. *Review of Financial Studies*.
- Michaud, R. (1989). The Markowitz optimization enigma: is "optimized" optimal? *Financial Analysts Journal*.
- Memmel, C. (2003). Performance hypothesis testing with the Sharpe ratio. *Finance Letters*.
- Cornuejols, G. and Tutuncu, R. (2007). *Optimization Methods in Finance*. Cambridge University Press.

*Research code. Not investment advice.*

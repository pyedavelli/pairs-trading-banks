# Market-Neutral Pairs Trading on US Bank Cointegration

A statistical arbitrage strategy on cointegrated US bank pairs, built end-to-end in Python with strict in-sample/out-of-sample discipline.

![Portfolio Equity Curve](results/tearsheet_portfolio.png)

## Overview

This project screens a universe of 11 US bank stocks for cointegrated pairs using the Engle-Granger two-step test, then trades the surviving pairs via a rolling z-score mean-reversion signal. The full pipeline runs from raw data download through backtest, performance metrics, and visualization.

**Out-of-sample results (2021–2024):**
- Annualized return: **5.7%** (at 3x leverage)
- Sharpe ratio: **0.39**
- Max drawdown: **-18%**
- Correlation with SPY: **-0.03** (confirming market neutrality)

## Methodology

**Universe:** 11 US banks spanning money center (JPM, BAC, C, WFC), investment banking (GS, MS), and super-regional (USB, PNC, FITB, KEY, RF) segments.

**Data:** Daily adjusted close prices from Yahoo Finance, 2013-01-01 through 2024-12-31 (~3,000 trading days).

**Train/test split:** 2013–2020 in-sample (67%) for pair discovery and parameter selection; 2021–2024 out-of-sample (33%) for evaluation. The out-of-sample window deliberately includes the 2023 regional banking crisis (SVB, Signature, First Republic) as a stress test.

**Pair selection (two-stage filter, applied to in-sample only):**
1. *Statistical filter*: Engle-Granger ADF p-value < 0.05 on the cointegrating regression residuals
2. *Economic filter*: in-sample Sharpe > 0 (a spread can be statistically mean-reverting but unprofitable after transaction costs)

Three pairs cleared Stage 1 (FITB/RF, C/USB, GS/PNC); C/USB failed Stage 2 due to structural break during 2020 COVID, leaving **FITB/RF** and **GS/PNC** as the tradeable pairs.

**Signal:** Rolling 60-day z-score of the spread. Enter at ±2σ, exit at ±0.5σ, stop out at ±4σ.

**Backtest assumptions:**
- 5 bps transaction cost per side per leg
- 3x leverage (conservative vs typical stat arb book of 4-6x)
- 1-day execution lag (signal computed at close, traded at next day's close)
- Equal-weight portfolio across surviving pairs

## Results

| Metric | Portfolio (in-sample) | Portfolio (out-of-sample) |
|---|---|---|
| Total return | +23.5% | +24.9% |
| Annualized return | 2.7% | 5.7% |
| Annualized volatility | 14.6% | 14.7% |
| Sharpe ratio | 0.18 | 0.39 |
| Max drawdown | -28.4% | -18.0% |
| Calmar | 0.09 | 0.32 |
| Correlation with SPY | -0.18 | -0.03 |

**Notable findings:**
- Out-of-sample Sharpe *exceeded* in-sample Sharpe, suggesting the strategy is not overfit
- Portfolio drawdown (-18%) is meaningfully smaller than either individual pair's drawdown (-31% and -28%), demonstrating diversification benefit
- FITB/RF was the dominant contributor; GS/PNC contributed near-zero out-of-sample, illustrating that statistical cointegration does not guarantee tradeable P&L

## Limitations and Extensions

Being explicit about what this project does and does not demonstrate:

- **Multiple testing not corrected.** Screening 55 pairs at p<0.05 yields ~3 false positives by chance; a Bonferroni or FDR correction would be more rigorous.
- **No modeling of margin financing costs.** A levered market-neutral book typically incurs 50-150 bps/year in broker financing, which would reduce annualized return by that amount.
- **Survivorship bias.** Uses currently-listed banks; a fully rigorous study would include delisted names.
- **Static cointegration.** The strategy assumes the cointegration relationship is stable; production systems should test for rolling stability, since relationships can break structurally (as GS/PNC arguably did post-2020).
- **Single parameter set.** Window length (60d), entry/exit thresholds (±2σ / ±0.5σ), and hedge ratios are not cross-validated. Rolling recalibration would be a natural next step.

---

*Built as a portfolio project to demonstrate quantitative research methodology: data pipeline, statistical testing, backtest infrastructure, performance evaluation, and honest reporting of both successes and limitations.*
"""
cointegration.py
================
Screens every possible pair of stocks in our universe for cointegration
using the Engle-Granger two-step test.

Conceptual recap:
- Two stocks are "cointegrated" if their prices wander individually but
  the spread between them is stable (mean-reverting).
- This is what makes pairs trading possible: we bet the spread reverts.
- Engle-Granger procedure:
    Step 1: Regress price_A on price_B → get hedge ratio (beta)
    Step 2: Test the residuals (which ARE the spread) for stationarity
            using the Augmented Dickey-Fuller (ADF) test.
    If ADF rejects the null at p<0.05, residuals are stationary,
    meaning the pair is cointegrated.

This file should be run after data_loader.py has cached the price data.
"""

import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def engle_granger_test(price_a, price_b, significance=0.05):
    """
    Run the Engle-Granger cointegration test on a single pair.

    Parameters
    ----------
    price_a, price_b : pd.Series
        Price series for two stocks. Should be aligned on dates.
    significance : float
        P-value threshold below which we declare the pair cointegrated.
        0.05 is the conventional default (5% false positive rate).

    Returns
    -------
    dict containing:
        beta            : hedge ratio (how many shares of B per share of A)
        alpha           : intercept of the regression
        adf_pvalue      : p-value from ADF test on residuals
        is_cointegrated : bool — True if adf_pvalue < significance
        half_life       : mean-reversion speed, in trading days
    """
    # Align the two series on their common dates and drop any NaN rows.
    # pd.concat with axis=1 puts them side by side as two columns.
    df = pd.concat([price_a, price_b], axis=1).dropna()
    y = df.iloc[:, 0]  # price_a as the dependent variable
    x = df.iloc[:, 1]  # price_b as the independent variable

    # STEP 1: Linear regression y = alpha + beta * x + residuals
    # sm.add_constant() adds a column of 1s so OLS estimates the intercept.
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    alpha = model.params.iloc[0]   # intercept
    beta = model.params.iloc[1]    # slope = hedge ratio

    # The residuals are the spread: what's left after removing the linear
    # relationship. If the pair is cointegrated, this series should be
    # stationary (mean-reverting around zero).
    residuals = y - (alpha + beta * x)

    # STEP 2: Augmented Dickey-Fuller test for stationarity on residuals.
    # Null hypothesis: the series has a unit root (i.e., NOT stationary).
    # If we reject the null (low p-value), the series IS stationary,
    # meaning the pair is cointegrated.
    # autolag='AIC' lets the test pick the optimal lag order automatically.
    adf_result = adfuller(residuals, autolag='AIC')
    adf_pvalue = adf_result[1]  # The p-value is the second element of the tuple

    # Estimate how fast the spread reverts — useful for picking the
    # z-score lookback window later.
    half_life = _estimate_half_life(residuals)

    return {
        'beta': beta,
        'alpha': alpha,
        'adf_pvalue': adf_pvalue,
        'is_cointegrated': adf_pvalue < significance,
        'half_life': half_life,
    }


def _estimate_half_life(spread):
    """
    Estimate the half-life of mean reversion in the spread.

    The math: if the spread follows an AR(1) process, the half-life
    (time to revert halfway to the mean) is:
        half_life = -ln(2) / ln(1 + lambda)
    where lambda is the coefficient on the lagged level in a regression
    of the change in spread on the previous spread level.

    Practical use: if half-life is 10 days, use a z-score window of
    roughly 2-3x that (20-30 days). Too short and you trade noise; too
    long and you miss the reversion.

    Underscore prefix on the function name signals "private helper" —
    you wouldn't import this from another file.
    """
    spread = spread.dropna()
    spread_lag = spread.shift(1).dropna()   # Yesterday's spread
    delta = spread.diff().dropna()           # Today's change in spread
    spread_lag = spread_lag.loc[delta.index]  # Align indices

    # Regress delta on spread_lag: delta = const + lambda * spread_lag
    X = sm.add_constant(spread_lag)
    model = sm.OLS(delta, X).fit()
    lambda_coef = model.params.iloc[1]

    # If lambda >= 0, the spread is NOT mean-reverting (it explodes or
    # random walks). Return NaN to flag this.
    if lambda_coef >= 0:
        return np.nan

    half_life = -np.log(2) / lambda_coef
    return half_life


def screen_all_pairs(prices, significance=0.05):
    """
    Run the cointegration test on every unique pair in the price DataFrame.

    For N tickers, there are N*(N-1)/2 unique unordered pairs. With our
    11 banks, that's 55 pairs to test.

    Note on multiple testing: running 55 tests at p<0.05 means we'd expect
    ~2-3 false positives by random chance even if NO pairs were truly
    cointegrated. Mention this honestly in your README's "Limitations"
    section — it's a real concern that distinguishes thoughtful work.

    Returns
    -------
    pd.DataFrame
        One row per pair, sorted by adf_pvalue ascending (best first).
    """
    tickers = prices.columns.tolist()
    results = []

    # itertools.combinations gives every unique unordered pair.
    # (A,B) is included, (B,A) is not — exactly what we want.
    for ticker_a, ticker_b in itertools.combinations(tickers, 2):
        try:
            test = engle_granger_test(prices[ticker_a], prices[ticker_b],
                                      significance)
            results.append({
                'ticker_a': ticker_a,
                'ticker_b': ticker_b,
                'beta': test['beta'],
                'adf_pvalue': test['adf_pvalue'],
                'is_cointegrated': test['is_cointegrated'],
                'half_life_days': test['half_life'],
            })
        except Exception as e:
            # If a particular pair fails (rare, e.g. perfectly collinear
            # data), skip it rather than crash the whole screen.
            print(f"Failed on {ticker_a}/{ticker_b}: {e}")

    # Sort so the most cointegrated pairs (lowest p-value) come first.
    results_df = pd.DataFrame(results).sort_values('adf_pvalue').reset_index(drop=True)
    return results_df


if __name__ == '__main__':
    # Import from our own data_loader module.
    # This works because both files are in the src/ package.
    from data_loader import download_prices, split_in_out_sample, BANK_TICKERS

    # Load price data (will use cache from previous run)
    prices = download_prices(tickers=BANK_TICKERS)

    # CRITICAL: only use IN-SAMPLE data for pair selection. Looking at
    # out-of-sample data here would constitute lookahead bias — we'd be
    # using future information to pick our pairs, which makes the final
    # evaluation meaningless.
    in_sample, _ = split_in_out_sample(prices)

    print("\nScreening all pairs for cointegration (in-sample only)...\n")
    results = screen_all_pairs(in_sample)

    print("Top 10 most cointegrated pairs:")
    # .to_string(index=False) prints the DataFrame without the row numbers,
    # which is cleaner for console output.
    print(results.head(10).to_string(index=False))

    n_coint = results['is_cointegrated'].sum()
    print(f"\n{n_coint} of {len(results)} pairs are cointegrated at p<0.05")
    print("Pairs used in signals.py were selected via a two-stage filter — see signals.py for details.")
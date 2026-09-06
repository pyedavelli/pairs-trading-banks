"""
signals.py
==========
Converts a cointegrated pair into trading signals.

The pipeline within this file:
    1. Compute the spread: spread_t = price_A_t - beta * price_B_t
    2. Compute the rolling z-score of the spread:
           z_t = (spread_t - rolling_mean) / rolling_std
    3. Generate position signals based on z-score thresholds:
           - Enter long spread when z < -entry_threshold (spread too low)
           - Enter short spread when z > +entry_threshold (spread too high)
           - Exit when |z| < exit_threshold (mean reversion happened)
           - Stop out when |z| > stop_threshold (relationship broke)

Position conventions:
    +1 = LONG the spread  = BUY stock A, SHORT stock B
    -1 = SHORT the spread = SHORT stock A, BUY stock B
     0 = no position
"""

import numpy as np
import pandas as pd


def compute_spread(price_a, price_b, hedge_ratio):
    """
    Compute the spread between two stocks given the hedge ratio.

    spread_t = price_A_t - hedge_ratio * price_B_t

    The hedge ratio comes from the Engle-Granger regression (the 'beta'
    we computed in cointegration.py). It tells us how many units of
    stock B to use to neutralize stock A's exposure.

    Returns
    -------
    pd.Series indexed by date.
    """
    return price_a - hedge_ratio * price_b


def compute_zscore(spread, window=60):
    """
    Compute the rolling z-score of the spread.

    z_t = (spread_t - rolling_mean_t) / rolling_std_t

    The z-score tells us how many standard deviations the current spread
    is from its recent mean. |z| > 2 is conventionally "unusual," meaning
    we'd expect mean reversion soon.

    The 'window' parameter is critical. Rule of thumb: pick window ≈ 2-3x
    the half-life estimated in cointegration.py. Default of 60 days
    (~3 months) is a reasonable starting point for our bank pairs.

    Important: we use a TRAILING window. At time t, the rolling stats
    use only data up to and including t — never future data. This is
    what makes the signal usable in real-time without lookahead bias.

    Returns
    -------
    pd.Series of z-scores. The first `window-1` values will be NaN
    because there isn't enough history yet.
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore


def generate_signals(zscore, entry=2.0, exit=0.5, stop=4.0):
    """
    Convert a z-score series into a position signal series.

    Rules (the "state machine"):
        - If currently flat (position == 0):
            * If z < -entry: go LONG spread  (position = +1)
            * If z > +entry: go SHORT spread (position = -1)
            * Otherwise: stay flat
        - If currently LONG (position == +1):
            * If z > -exit: close out (the spread reverted)  -> position = 0
            * If z < -stop: stop out (relationship broke)    -> position = 0
            * Otherwise: hold the position
        - If currently SHORT (position == -1):
            * If z < +exit: close out (reverted)  -> position = 0
            * If z > +stop: stop out              -> position = 0
            * Otherwise: hold

    Why a state machine and not a vectorized formula? Because the
    position depends on the PREVIOUS position. You can't just compute
    "where should I be today" from the z-score alone — you need to know
    whether you're already in a trade.

    Parameters
    ----------
    zscore : pd.Series
        Output of compute_zscore().
    entry : float
        |z| level at which we open a new position. 2.0 is standard.
    exit : float
        |z| level at which we close a winning position. 0.5 means we
        exit when the spread has reverted most of the way to the mean.
    stop : float
        |z| level at which we close a losing position to limit damage.
        4.0 is loose; tighter stops trade more often but lose more often.

    Returns
    -------
    pd.Series of {-1, 0, +1} integer positions, indexed by date.
    """
    # Initialize the position series with zeros, matching the zscore index
    position = pd.Series(0, index=zscore.index, dtype=int)

    # Iterate day by day. This is one of the rare cases where a Python
    # loop is the clearest way to express the logic — the state dependency
    # makes vectorization awkward.
    current_position = 0
    for t in range(len(zscore)):
        z = zscore.iloc[t]

        # If z is NaN (early in series, before window fills), stay flat
        if pd.isna(z):
            position.iloc[t] = 0
            continue

        if current_position == 0:
            # Currently flat — look for entry
            if z < -entry:
                current_position = +1   # Long the spread
            elif z > +entry:
                current_position = -1   # Short the spread
        elif current_position == +1:
            # Currently long — look for exit or stop
            if z > -exit:
                current_position = 0    # Mean reverted, take profit
            elif z < -stop:
                current_position = 0    # Spread blew out further, stop loss
        elif current_position == -1:
            # Currently short — look for exit or stop
            if z < +exit:
                current_position = 0    # Mean reverted, take profit
            elif z > +stop:
                current_position = 0    # Spread blew out further, stop loss

        position.iloc[t] = current_position

    return position


def build_pair_signals(price_a, price_b, hedge_ratio, window=60,
                      entry=2.0, exit=0.5, stop=4.0):
    """
    Convenience wrapper: takes raw prices and parameters, returns a
    DataFrame with everything we need for the backtest.

    Returns
    -------
    pd.DataFrame with columns:
        spread   : the raw spread
        zscore   : the rolling z-score
        position : the signal (-1, 0, +1)
    """
    spread = compute_spread(price_a, price_b, hedge_ratio)
    zscore = compute_zscore(spread, window=window)
    position = generate_signals(zscore, entry=entry, exit=exit, stop=stop)

    return pd.DataFrame({
        'spread': spread,
        'zscore': zscore,
        'position': position,
    })


# ---------------------------------------------------------------------------
# THE PAIRS WE'RE TRADING
# ---------------------------------------------------------------------------
# Pair selection used a two-stage filter applied to in-sample data only
# (2013-2020), to avoid lookahead bias:
#
#   Stage 1 — Statistical filter: Engle-Granger ADF p-value < 0.05.
#             Identifies pairs with statistically significant cointegration.
#
#   Stage 2 — Economic filter: in-sample Sharpe ratio > 0.
#             Statistical cointegration is necessary but not sufficient for
#             a tradeable signal — a spread can be mean-reverting yet still
#             unprofitable after transaction costs and stop-losses.
#
# Three pairs cleared Stage 1 in our initial screen (FITB/RF, C/USB, GS/PNC).
# C/USB failed Stage 2 (in-sample Sharpe = -0.21), so we drop it. The two
# surviving pairs (FITB/RF, GS/PNC) form our tradeable portfolio.
#
# This C/USB result is itself an interesting finding: Citigroup's 2020 COVID
# drawdown and the 2023 regional banking crisis both hit C disproportionately
# relative to USB, blowing the spread out and triggering stop-losses faster
# than the take-profits could accumulate. Statistical cointegration in
# 2013-2019 broke down structurally after 2020.
PAIRS_TO_TRADE = [
    # (ticker_a, ticker_b, hedge_ratio_from_in_sample_regression)
    ('FITB', 'RF',  1.461526),
    ('GS',   'PNC', 1.158081),
]

# Kept for reference — the full initial screen, before the Sharpe filter.
# Useful if you want to re-run the analysis showing the C/USB pair to
# demonstrate why we filtered it out.
PAIRS_INITIAL_SCREEN = [
    ('FITB', 'RF',  1.461526),
    ('C',    'USB', 1.308450),
    ('GS',   'PNC', 1.158081),
]


if __name__ == '__main__':
    from data_loader import download_prices, split_in_out_sample, BANK_TICKERS

    # Load price data (will use cache from previous run)
    prices = download_prices(tickers=BANK_TICKERS)
    in_sample, _ = split_in_out_sample(prices)

    # Build and inspect signals for each of our two pairs.
    # We do this on in-sample only here — out-of-sample comes in backtest.py.
    for ticker_a, ticker_b, hedge_ratio in PAIRS_TO_TRADE:
        print(f"\n{'='*60}")
        print(f"PAIR: {ticker_a} / {ticker_b}  (hedge ratio = {hedge_ratio:.4f})")
        print(f"{'='*60}")

        signals = build_pair_signals(
            in_sample[ticker_a],
            in_sample[ticker_b],
            hedge_ratio=hedge_ratio,
            window=60,
            entry=2.0,
            exit=0.5,
            stop=4.0,
        )

        # Position distribution: how often are we long/short/flat?
        print("\nPosition distribution (in-sample):")
        pos_counts = signals['position'].value_counts().sort_index()
        for pos, count in pos_counts.items():
            pct = count / len(signals) * 100
            label = {-1: 'SHORT', 0: 'FLAT', +1: 'LONG'}[pos]
            print(f"  {label:6s} ({pos:+d}): {count:4d} days  ({pct:5.1f}%)")

        # Z-score summary stats — sanity check that the distribution looks
        # roughly mean-zero with reasonable variance
        print("\nZ-score summary:")
        print(signals['zscore'].describe().to_string())
"""
backtest.py
===========
Simulates running the pairs trading strategy on historical data and
produces a P&L series.

Key principles enforced here:
    1. NO LOOKAHEAD BIAS: positions are determined using data available
       up to time t, but we earn P&L on the position from t-1 (i.e., we
       can't trade on a signal until the day after it appears).
    2. TRANSACTION COSTS: every change in position costs money. We use
       5 basis points (0.05%) per dollar of notional traded, which is
       a realistic estimate for retail equity execution.
    3. DOLLAR-NEUTRAL POSITIONS: when we're "long the spread," we put
       equal dollar amounts long stock A and short stock B (adjusted by
       hedge ratio). This makes the strategy market-neutral.

This file runs the backtest for each of our two pairs and then combines
them into a portfolio (equal-weighted across the two pairs).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# LEVERAGE
# ---------------------------------------------------------------------------
# Statistical arbitrage strategies inherently generate small unlevered
# returns because the spread movements they capture are small relative to
# the underlying stocks' price levels. Professional stat arb funds run
# these strategies at 3-5x leverage to scale the dollar returns while
# preserving the Sharpe ratio.
#
# We assume 3x leverage here, which is conservative relative to industry
# practice (typical equity stat arb books run 4-6x gross exposure). This
# means every $1 of capital controls $3 of long-short notional exposure.
#
# Important caveats (mention in any writeup):
#   1. Leverage scales both returns AND drawdowns linearly. Our 3x-levered
#      drawdowns are 3x larger than the unlevered figures.
#   2. We do not model financing costs (broker margin interest). For a
#      market-neutral book financing costs are typically 50-150 bps/year,
#      which would reduce annualized return by that amount.
#   3. Sharpe ratio is unchanged by leverage in theory (return and vol
#      scale equally); in practice slight differences from financing.
LEVERAGE = 3.0


def backtest_pair(price_a, price_b, signals, hedge_ratio, cost_bps=5):
    """
    Run a backtest on a single pair given pre-computed signals.

    Parameters
    ----------
    price_a, price_b : pd.Series
        Aligned price series.
    signals : pd.DataFrame
        Output of build_pair_signals() — must contain 'position' column.
    hedge_ratio : float
        Hedge ratio (beta) from the cointegration regression.
    cost_bps : float
        Transaction cost in basis points per side. 5 bps = 0.05% per
        unit of notional traded.

    Returns
    -------
    pd.DataFrame with columns:
        position    : position from signals (-1, 0, +1)
        ret_a       : daily return of stock A
        ret_b       : daily return of stock B
        strategy_ret: daily strategy return (after costs)
        cum_return  : cumulative return (starts at 1.0, grows from there)
    """
    # Align everything on common dates
    df = pd.concat([price_a, price_b, signals['position']], axis=1).dropna()
    df.columns = ['price_a', 'price_b', 'position']

    # Compute daily simple returns for each stock.
    # pct_change() gives (today - yesterday) / yesterday.
    df['ret_a'] = df['price_a'].pct_change()
    df['ret_b'] = df['price_b'].pct_change()

    # CRITICAL: shift position by 1 day before applying to returns.
    # Reasoning: we observe today's close prices, compute the z-score,
    # and decide on a position. But we can only act on that decision
    # tomorrow (you can't trade at a close you've already observed).
    # So today's signal earns tomorrow's return.
    position_yesterday = df['position'].shift(1).fillna(0)

    # Strategy return formula:
    #   When long spread  (position = +1): long A, short B (scaled by hedge)
    #     -> return = ret_a - hedge_ratio * ret_b
    #   When short spread (position = -1): short A, long B
    #     -> return = -ret_a + hedge_ratio * ret_b
    # We combine these by multiplying by position. We also normalize by
    # (1 + |hedge_ratio|) so that the return is on a per-dollar-of-capital
    # basis rather than scaled up by the hedge ratio.
    # Unlevered gross return from the spread move
    gross_ret_unlevered = position_yesterday * (df['ret_a'] - hedge_ratio * df['ret_b']) / (1 + abs(hedge_ratio))

    # Apply leverage. With LEVERAGE=3, $1 of capital controls $3 of notional,
    # so a 1% spread move generates a 3% return on capital.
    gross_ret = gross_ret_unlevered * LEVERAGE

    # Compute transaction costs whenever position changes.
    # cost_per_change accounts for the fact that flipping from +1 to -1 means
    # closing one position AND opening the opposite one (twice the turnover).
    # Costs also scale with leverage — bigger positions, bigger commissions.
    position_change = position_yesterday.diff().abs().fillna(0)
    cost_per_change = (cost_bps / 10000) * 2 * LEVERAGE  # bps to decimal, scaled
    costs = position_change * cost_per_change

    # Net strategy return = levered gross return minus levered costs
    df['strategy_ret'] = gross_ret - costs

    # Cumulative return: start at 1.0, multiply by (1 + daily return).
    # cumprod() multiplies all elements up to each point.
    df['cum_return'] = (1 + df['strategy_ret'].fillna(0)).cumprod()

    return df


def trade_log(backtest_df):
    """
    Extract a list of individual trades from the backtest DataFrame.

    A "trade" is a sequence of consecutive days with the same non-zero
    position. We log entry date, exit date, position type, and P&L.

    Useful for computing hit rate and average trade length.

    Returns
    -------
    pd.DataFrame with one row per completed trade.
    """
    trades = []
    in_trade = False
    entry_idx = None
    entry_position = 0
    entry_cum_return = 1.0

    for i in range(len(backtest_df)):
        pos = backtest_df['position'].iloc[i]
        cum = backtest_df['cum_return'].iloc[i]

        if not in_trade and pos != 0:
            # Trade begins
            in_trade = True
            entry_idx = i
            entry_position = pos
            entry_cum_return = cum
        elif in_trade and pos == 0:
            # Trade ends
            trades.append({
                'entry_date': backtest_df.index[entry_idx],
                'exit_date': backtest_df.index[i],
                'position': entry_position,  # +1 or -1
                'duration_days': i - entry_idx,
                'pnl_pct': (cum / entry_cum_return - 1) * 100,
            })
            in_trade = False

    return pd.DataFrame(trades)


def build_portfolio(pair_backtests):
    """
    Combine multiple pair backtests into an equal-weighted portfolio.

    Each pair contributes its own daily return series. We average them
    across pairs each day. This is equivalent to allocating 1/N of capital
    to each pair and rebalancing daily.

    Why equal-weight? It's the simplest sensible choice and avoids the
    appearance of overfitting weights to the in-sample data. In a more
    advanced project you could use risk-parity (inverse-volatility) weights.

    Parameters
    ----------
    pair_backtests : dict of {pair_name: backtest_df}
        Each backtest_df is the output of backtest_pair().

    Returns
    -------
    pd.DataFrame with combined portfolio returns and cumulative curve.
    """
    # Stack the per-pair strategy returns into one DataFrame
    returns_df = pd.DataFrame({
        name: bt['strategy_ret']
        for name, bt in pair_backtests.items()
    })

    # Equal-weighted daily return = mean across pairs
    portfolio_ret = returns_df.mean(axis=1)

    # Build cumulative curve
    portfolio_cum = (1 + portfolio_ret.fillna(0)).cumprod()

    return pd.DataFrame({
        'strategy_ret': portfolio_ret,
        'cum_return': portfolio_cum,
    })


if __name__ == '__main__':
    from data_loader import download_prices, split_in_out_sample, BANK_TICKERS
    from signals import build_pair_signals, PAIRS_TO_TRADE

    # Load full price history
    prices = download_prices(tickers=BANK_TICKERS)
    in_sample, out_sample = split_in_out_sample(prices)

    # Backtest each pair on the FULL period using the in-sample hedge ratio.
    # We compute signals on the full period so we have a continuous P&L
    # curve from 2013 through 2024.
    pair_backtests = {}

    for ticker_a, ticker_b, hedge_ratio in PAIRS_TO_TRADE:
        signals = build_pair_signals(
            prices[ticker_a],
            prices[ticker_b],
            hedge_ratio=hedge_ratio,
            window=60,
        )
        bt = backtest_pair(prices[ticker_a], prices[ticker_b],
                           signals, hedge_ratio)
        pair_backtests[f'{ticker_a}/{ticker_b}'] = bt

        # Print per-pair summary
        print(f"\n{'='*60}")
        print(f"PAIR: {ticker_a}/{ticker_b}")
        print(f"{'='*60}")

        bt_in = bt.loc[in_sample.index.min():in_sample.index.max()]
        bt_out = bt.loc[out_sample.index.min():out_sample.index.max()]

        in_ret = (bt_in['cum_return'].iloc[-1] / bt_in['cum_return'].iloc[0] - 1) * 100
        out_ret = (bt_out['cum_return'].iloc[-1] / bt_out['cum_return'].iloc[0] - 1) * 100

        print(f"  In-sample  return: {in_ret:+7.2f}%")
        print(f"  Out-sample return: {out_ret:+7.2f}%")

        trades = trade_log(bt)
        if len(trades) > 0:
            win_rate = (trades['pnl_pct'] > 0).mean() * 100
            avg_dur = trades['duration_days'].mean()
            print(f"  Trades: {len(trades)}, win rate: {win_rate:.1f}%, avg duration: {avg_dur:.1f} days")

    # Build the portfolio (equal-weight across pairs)
    portfolio = build_portfolio(pair_backtests)

    print(f"\n{'='*60}")
    print("EQUAL-WEIGHTED PORTFOLIO (all 2 pairs)")
    print(f"{'='*60}")

    port_in = portfolio.loc[in_sample.index.min():in_sample.index.max()]
    port_out = portfolio.loc[out_sample.index.min():out_sample.index.max()]

    in_ret = (port_in['cum_return'].iloc[-1] / port_in['cum_return'].iloc[0] - 1) * 100
    out_ret = (port_out['cum_return'].iloc[-1] / port_out['cum_return'].iloc[0] - 1) * 100

    print(f"  In-sample  return: {in_ret:+7.2f}%")
    print(f"  Out-sample return: {out_ret:+7.2f}%")
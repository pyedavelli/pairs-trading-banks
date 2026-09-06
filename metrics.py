"""
metrics.py
==========
Computes standard performance metrics from a backtest return series.

These are the metrics any quantitative finance recruiter will look at,
so it's worth computing them properly and understanding what they say.
"""

import numpy as np
import pandas as pd


# Number of trading days in a year (used for annualizing).
# 252 is the standard convention (~21 trading days/month × 12 months).
TRADING_DAYS_PER_YEAR = 252


def annualized_return(daily_returns):
    """
    Annualized geometric return.

    Geometric (not arithmetic) because returns compound. The formula:
        (1 + total_return) ^ (252 / num_days) - 1

    Returns a decimal (e.g., 0.08 = 8% annualized).
    """
    daily_returns = daily_returns.dropna()
    total_return = (1 + daily_returns).prod() - 1
    n_days = len(daily_returns)
    if n_days == 0:
        return np.nan
    return (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1


def annualized_volatility(daily_returns):
    """
    Annualized standard deviation of returns.

    Daily std × sqrt(252). The sqrt(252) scaling comes from the fact
    that variance scales linearly with time, so std scales with sqrt(time).
    """
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns, risk_free_rate=0.0):
    """
    Sharpe ratio: excess return per unit of volatility.

    Sharpe = (annualized_return - risk_free_rate) / annualized_volatility

    Interpretation:
        < 1 : mediocre
        1-2 : decent
        2-3 : strong
        > 3 : raise eyebrows (usually means a bug or overfitting)

    We default risk_free_rate=0 for simplicity. In practice you'd subtract
    the T-bill rate, but for a market-neutral strategy the risk-free
    adjustment makes less difference than for a long-only portfolio.
    """
    ann_ret = annualized_return(daily_returns)
    ann_vol = annualized_volatility(daily_returns)
    if ann_vol == 0 or pd.isna(ann_vol):
        return np.nan
    return (ann_ret - risk_free_rate) / ann_vol


def max_drawdown(cumulative_returns):
    """
    Maximum peak-to-trough decline in the equity curve.

    For each point in time, compute how far the equity curve is below
    its running peak. The max drawdown is the worst such decline.

    Returned as a negative decimal (e.g., -0.15 = 15% drawdown).
    """
    # Running maximum of the equity curve up to each point
    running_max = cumulative_returns.cummax()
    # Drawdown at each point = (current - peak) / peak
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def calmar_ratio(daily_returns, cumulative_returns):
    """
    Calmar = annualized return / |max drawdown|.

    A complement to Sharpe. Sharpe penalizes all volatility; Calmar only
    cares about downside (drawdown). Some investors prefer it because they
    care a lot more about drawdowns than upside volatility.
    """
    ann_ret = annualized_return(daily_returns)
    mdd = max_drawdown(cumulative_returns)
    if mdd == 0 or pd.isna(mdd):
        return np.nan
    return ann_ret / abs(mdd)


def hit_rate(trade_log_df):
    """
    Percentage of completed trades that were profitable.
    """
    if len(trade_log_df) == 0:
        return np.nan
    return (trade_log_df['pnl_pct'] > 0).mean()


def summary_stats(backtest_df, trade_log_df, label=''):
    """
    Build a one-row summary of all key metrics.

    The `label` parameter lets you tag the row (e.g., 'in-sample' vs
    'out-of-sample') when stacking multiple summaries into a comparison
    table.

    Returns a dict (easy to convert to a DataFrame row).
    """
    daily_ret = backtest_df['strategy_ret']
    cum_ret = backtest_df['cum_return']

    return {
        'label': label,
        'total_return_pct': (cum_ret.iloc[-1] / cum_ret.iloc[0] - 1) * 100,
        'annualized_return_pct': annualized_return(daily_ret) * 100,
        'annualized_vol_pct': annualized_volatility(daily_ret) * 100,
        'sharpe': sharpe_ratio(daily_ret),
        'max_drawdown_pct': max_drawdown(cum_ret) * 100,
        'calmar': calmar_ratio(daily_ret, cum_ret),
        'num_trades': len(trade_log_df),
        'hit_rate_pct': hit_rate(trade_log_df) * 100 if len(trade_log_df) > 0 else np.nan,
    }


def correlation_with_benchmark(strategy_returns, benchmark_returns):
    """
    Correlation between strategy daily returns and benchmark daily returns.

    For a market-neutral strategy, this should be close to zero. A low
    correlation is actually the SELLING POINT — the strategy provides
    returns uncorrelated with the broader market.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])


if __name__ == '__main__':
    from data_loader import download_prices, split_in_out_sample, BANK_TICKERS, BENCHMARK
    from signals import build_pair_signals, PAIRS_TO_TRADE
    from backtest import backtest_pair, trade_log, build_portfolio

    # Load price data (banks + SPY benchmark)
    prices = download_prices(tickers=BANK_TICKERS + [BENCHMARK])
    in_sample, out_sample = split_in_out_sample(prices)

    # Re-run backtest for each pair so we have everything in scope
    pair_backtests = {}
    pair_trades = {}

    for ticker_a, ticker_b, hedge_ratio in PAIRS_TO_TRADE:
        signals = build_pair_signals(
            prices[ticker_a], prices[ticker_b],
            hedge_ratio=hedge_ratio, window=60,
        )
        bt = backtest_pair(prices[ticker_a], prices[ticker_b],
                           signals, hedge_ratio)
        pair_backtests[f'{ticker_a}/{ticker_b}'] = bt
        pair_trades[f'{ticker_a}/{ticker_b}'] = trade_log(bt)

    portfolio = build_portfolio(pair_backtests)

    # ---- Build per-pair summary table ----
    print("=" * 90)
    print("PER-PAIR PERFORMANCE METRICS")
    print("=" * 90)

    # Bucket trades by EXIT date rather than entry date. A trade that opens
    # in-sample but closes out-of-sample generates its P&L primarily in the
    # out-of-sample period, so it belongs to the out-of-sample bucket.
    # Using exit_date also matches how the daily return series is naturally
    # split by date, keeping the trade log consistent with the return series.
    split_boundary = in_sample.index.max()

    rows = []
    for pair_name, bt in pair_backtests.items():
        bt_in = bt.loc[in_sample.index.min():in_sample.index.max()]
        bt_out = bt.loc[out_sample.index.min():out_sample.index.max()]

        trades_full = pair_trades[pair_name]
        if len(trades_full) > 0:
            trades_in = trades_full[trades_full['exit_date'] <= split_boundary]
            trades_out = trades_full[trades_full['exit_date'] > split_boundary]
        else:
            trades_in = trades_full
            trades_out = trades_full

        rows.append(summary_stats(bt_in, trades_in, label=f'{pair_name} (in)'))
        rows.append(summary_stats(bt_out, trades_out, label=f'{pair_name} (out)'))

    pair_table = pd.DataFrame(rows)
    print(pair_table.to_string(index=False))

    # ---- Portfolio-level summary (all pairs combined) ----
    print("\n" + "=" * 90)
    print("EQUAL-WEIGHTED PORTFOLIO METRICS")
    print("=" * 90)

    port_in = portfolio.loc[in_sample.index.min():in_sample.index.max()]
    port_out = portfolio.loc[out_sample.index.min():out_sample.index.max()]

    # Combine all trades across pairs into one log for portfolio-level
    # hit rate. Same exit-date bucketing as above for consistency.
    all_trades_in = pd.concat([
        pair_trades[name][pair_trades[name]['exit_date'] <= split_boundary]
        for name in pair_trades if len(pair_trades[name]) > 0
    ], ignore_index=True) if any(len(t) > 0 for t in pair_trades.values()) else pd.DataFrame()

    all_trades_out = pd.concat([
        pair_trades[name][pair_trades[name]['exit_date'] > split_boundary]
        for name in pair_trades if len(pair_trades[name]) > 0
    ], ignore_index=True) if any(len(t) > 0 for t in pair_trades.values()) else pd.DataFrame()

    port_rows = [
        summary_stats(port_in, all_trades_in, label='Portfolio (in-sample)'),
        summary_stats(port_out, all_trades_out, label='Portfolio (out-of-sample)'),
    ]
    port_table = pd.DataFrame(port_rows)
    print(port_table.to_string(index=False))

    # ---- Market-neutrality check: correlation with SPY ----
    print("\n" + "=" * 90)
    print("MARKET-NEUTRALITY CHECK (correlation with SPY)")
    print("=" * 90)

    spy_returns = prices[BENCHMARK].pct_change()
    corr_in = correlation_with_benchmark(port_in['strategy_ret'], spy_returns)
    corr_out = correlation_with_benchmark(port_out['strategy_ret'], spy_returns)

    print(f"  In-sample correlation with SPY:     {corr_in:+.3f}")
    print(f"  Out-of-sample correlation with SPY: {corr_out:+.3f}")
    print("\n  Interpretation: values close to 0 confirm market neutrality.")
    print("  |corr| < 0.2 is a strong selling point for the strategy.")
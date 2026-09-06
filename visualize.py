"""
visualize.py
============
Generates charts for the pairs trading strategy.

Two outputs:
    1. Per-pair tearsheets: 3-panel chart showing the strategy equity curve,
       spread z-score over time, and drawdown.
    2. Portfolio tearsheet: 2-panel layout (equity curve + drawdown) for the
       equal-weighted combined portfolio.

Design principle: we deliberately do NOT plot SPY on the equity curve
panels. Visually comparing a market-neutral strategy to a long-only index
is misleading — they have entirely different risk profiles and return
objectives. Instead we show a flat reference line at 1.0 ("cash baseline").
Market neutrality is reported separately via correlation with SPY in
metrics.py.

All charts save to the /results folder as PNG files.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Output directory for saved charts. Resolves to <project_root>/results.
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def plot_pair_tearsheet(backtest_df, signals_df, pair_name='Pair',
                        entry=2.0, exit=0.5, save_path=None):
    """
    Generate a 3-panel tearsheet figure for a single pair.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of backtest_pair(). Must contain 'cum_return' and 'strategy_ret'.
    signals_df : pd.DataFrame
        Output of build_pair_signals(). Must contain 'zscore'.
    pair_name : str
        Used in the chart title (e.g. 'FITB/RF').
    entry, exit : float
        Threshold lines to draw on the z-score chart.
    save_path : Path or None
        If provided, save the figure to this path. Always displays inline.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # ---- PANEL 1: Strategy equity curve ----
    ax1 = axes[0]
    strategy_curve = backtest_df['cum_return'] / backtest_df['cum_return'].iloc[0]

    ax1.plot(strategy_curve.index, strategy_curve.values,
             label='Pairs Strategy (3x levered)', linewidth=1.8, color='steelblue')
    # Flat reference line at 1.0 = "did nothing" baseline
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.6,
                label='Cash baseline (no return)')
    ax1.axvline(pd.Timestamp('2021-01-01'), color='black', linestyle=':',
                alpha=0.5, label='In/Out-of-Sample split')
    ax1.set_ylabel('Cumulative Return (starts at 1.0)')
    ax1.set_title(f'{pair_name}: Market-Neutral Pairs Strategy Equity Curve')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # ---- PANEL 2: Z-score with thresholds ----
    ax2 = axes[1]
    ax2.plot(signals_df.index, signals_df['zscore'].values,
             linewidth=0.8, color='steelblue')
    ax2.axhline(y=entry, color='red', linestyle='--', alpha=0.6,
                label=f'Entry threshold (±{entry})')
    ax2.axhline(y=-entry, color='red', linestyle='--', alpha=0.6)
    ax2.axhline(y=exit, color='green', linestyle='--', alpha=0.6,
                label=f'Exit threshold (±{exit})')
    ax2.axhline(y=-exit, color='green', linestyle='--', alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Z-score')
    ax2.set_title('Spread Z-Score Over Time')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # ---- PANEL 3: Drawdown ----
    ax3 = axes[2]
    running_max = backtest_df['cum_return'].cummax()
    drawdown = (backtest_df['cum_return'] - running_max) / running_max * 100
    ax3.fill_between(drawdown.index, drawdown.values, 0,
                     color='red', alpha=0.3)
    ax3.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_xlabel('Date')
    ax3.set_title('Strategy Drawdown')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_portfolio_tearsheet(portfolio_df, save_path=None):
    """
    2-panel tearsheet for the combined equal-weighted portfolio.

    Same design principle as the pair tearsheets: no SPY comparison on the
    equity curve, since a market-neutral portfolio shouldn't be visually
    compared to a long-only index.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Output of build_portfolio(). Must contain 'cum_return'.
    save_path : Path or None
        If provided, save the figure to this path. Always displays inline.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ---- Cumulative equity curve ----
    ax1 = axes[0]
    port_curve = portfolio_df['cum_return'] / portfolio_df['cum_return'].iloc[0]

    ax1.plot(port_curve.index, port_curve.values,
             label='Equal-Weight Pairs Portfolio (3x levered)',
             linewidth=1.8, color='steelblue')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.6,
                label='Cash baseline (no return)')
    ax1.axvline(pd.Timestamp('2021-01-01'), color='black', linestyle=':',
                alpha=0.5, label='In/Out-of-Sample split')
    ax1.set_ylabel('Cumulative Return (starts at 1.0)')
    ax1.set_title('Equal-Weighted Pairs Portfolio: Market-Neutral Equity Curve')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # ---- Drawdown ----
    ax2 = axes[1]
    running_max = portfolio_df['cum_return'].cummax()
    drawdown = (portfolio_df['cum_return'] - running_max) / running_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color='red', alpha=0.3)
    ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('Portfolio Drawdown')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


if __name__ == '__main__':
    from data_loader import download_prices, BANK_TICKERS
    from signals import build_pair_signals, PAIRS_TO_TRADE
    from backtest import backtest_pair, build_portfolio

    # Load bank price data. We no longer need SPY here since the equity
    # curve panels don't plot it — market-neutrality is confirmed in
    # metrics.py via correlation instead.
    prices = download_prices(tickers=BANK_TICKERS)

    # Build backtests for each pair and generate per-pair tearsheets
    pair_backtests = {}

    for ticker_a, ticker_b, hedge_ratio in PAIRS_TO_TRADE:
        signals = build_pair_signals(
            prices[ticker_a], prices[ticker_b],
            hedge_ratio=hedge_ratio, window=60,
        )
        bt = backtest_pair(prices[ticker_a], prices[ticker_b],
                           signals, hedge_ratio)
        pair_name = f'{ticker_a}/{ticker_b}'
        pair_backtests[pair_name] = bt

        # Generate per-pair tearsheet
        save_path = RESULTS_DIR / f'tearsheet_{ticker_a}_{ticker_b}.png'
        plot_pair_tearsheet(
            bt, signals,
            pair_name=pair_name, save_path=save_path,
        )

    # Generate portfolio tearsheet
    portfolio = build_portfolio(pair_backtests)
    save_path = RESULTS_DIR / 'tearsheet_portfolio.png'
    plot_portfolio_tearsheet(portfolio, save_path=save_path)

    print(f"\nAll charts saved to {RESULTS_DIR}")

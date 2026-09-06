"""
data_loader.py
==============
Responsible for downloading historical price data from Yahoo Finance and
caching it locally so we don't hit the API every time we run something.

This is the first module in the pipeline. Every other file depends on having
clean price data, so we build this first and test it before moving on.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# ---------------------------------------------------------------------------
# These live at module level so that any other file can import them and we
# have one source of truth for the universe of stocks and the date ranges.

# The 11 bank tickers we'll test for cointegration.
# Note: TFC (Truist) was excluded because the BB&T/SunTrust merger that created
# Truist closed in Dec 2019, so TFC's price history is too short for our
# in-sample window (2013-2020).
BANK_TICKERS = [
    # Money center banks — large, diversified, similar business mix
    'JPM',   # JPMorgan Chase
    'BAC',   # Bank of America
    'C',     # Citigroup
    'WFC',   # Wells Fargo
    # Investment banks — capital markets focused
    'GS',    # Goldman Sachs
    'MS',    # Morgan Stanley
    # Super-regional banks — commercial banking focused, similar geographies
    'USB',   # US Bancorp
    'PNC',   # PNC Financial
    'FITB',  # Fifth Third
    'KEY',   # KeyCorp
    'RF',    # Regions Financial
]

# The S&P 500 ETF — used as a benchmark to compare our strategy against
# a passive "just buy the market" approach.
BENCHMARK = 'SPY'

# Date range configuration.
# Total: 12 years of daily data.
# In-sample (2013-2020):  used to discover cointegrated pairs and tune params.
# Out-of-sample (2021-2024): used ONCE at the end to evaluate the strategy
# honestly. This includes the 2023 regional banking crisis as a stress test.
START_DATE = '2013-01-01'
END_DATE = '2024-12-31'
SPLIT_DATE = '2021-01-01'

# Path to the data cache folder. Path(__file__) gives this file's location,
# .parent goes up one level (to /src), .parent.parent goes up to project root,
# then we append /data to land in the cache folder.
DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)  # Create the folder if it doesn't exist yet


# ---------------------------------------------------------------------------
# MAIN DOWNLOAD FUNCTION
# ---------------------------------------------------------------------------
def download_prices(tickers, start_date=START_DATE, end_date=END_DATE,
                    cache_file='prices.csv', force_refresh=False):
    """
    Download daily adjusted close prices for a list of tickers.

    'Adjusted' close means the prices have been corrected for stock splits
    and dividends, so we can compute returns honestly. This is essential —
    using raw close prices would create fake price jumps on split days.

    The function caches results to a CSV file. Yahoo Finance is free but
    rate-limited, so we don't want to re-download every time we run the
    pipeline. If the cache already covers the date range and tickers we
    need, we load from disk instead.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols, e.g. ['JPM', 'BAC'].
    start_date, end_date : str
        Date range in 'YYYY-MM-DD' format.
    cache_file : str
        Filename inside DATA_DIR to read/write.
    force_refresh : bool
        If True, ignore the cache and re-download.

    Returns
    -------
    pd.DataFrame
        Rows = trading days, columns = tickers, values = adjusted close prices.
    """
    cache_path = DATA_DIR / cache_file

    # CACHE CHECK: if we already have the data on disk and it covers what
    # we need, just load it and return early. This is the common case once
    # you've run the script at least once.
    if cache_path.exists() and not force_refresh:
        print(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)

        # Verify the cache actually covers everything we're asking for
        cache_covers_dates = (df.index.min() <= pd.Timestamp(start_date) and
                              df.index.max() >= pd.Timestamp(end_date))
        cache_covers_tickers = all(t in df.columns for t in tickers)

        if cache_covers_dates and cache_covers_tickers:
            # .loc slices by date label (inclusive on both ends);
            # the column selection filters to only the requested tickers.
            return df.loc[start_date:end_date, tickers]
        else:
            print("Cache exists but is incomplete — re-downloading...")

    # DOWNLOAD: hit the Yahoo Finance API
    print(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}")
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,   # Adjusts for splits/dividends automatically
        progress=False,     # Suppresses the progress bar (cleaner output)
    )

    # yfinance returns a multi-level column DataFrame when you pass multiple
    # tickers (one level for OHLCV fields, one for tickers). We only want
    # the 'Close' column for each ticker, so we extract that level.
    # If only one ticker is passed, the structure is different — handle both.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    else:
        prices = raw[['Close']].rename(columns={'Close': tickers[0]})

    # DATA QUALITY CHECK: drop tickers with too much missing data.
    # A ticker missing >5% of its observations probably has a listing issue
    # (e.g., didn't trade during our entire window) and shouldn't be used.
    missing_pct = prices.isna().mean()
    bad_tickers = missing_pct[missing_pct > 0.05].index.tolist()
    if bad_tickers:
        print(f"Warning: dropping tickers with >5% missing data: {bad_tickers}")
        prices = prices.drop(columns=bad_tickers)

    # Forward-fill small gaps (e.g., a single missing day from an API hiccup)
    # then drop any rows that still have NaN. ffill propagates the last
    # known price forward, which is the conventional way to handle this.
    prices = prices.ffill().dropna()

    # Save to disk so the next run uses the cache
    prices.to_csv(cache_path)
    print(f"Saved to {cache_path}")
    return prices


# ---------------------------------------------------------------------------
# IN-SAMPLE / OUT-OF-SAMPLE SPLIT
# ---------------------------------------------------------------------------
def split_in_out_sample(prices, split_date=SPLIT_DATE):
    """
    Split price data into in-sample (training) and out-of-sample (testing).

    Why split at all? Because if we pick pairs and tune parameters using
    the same data we evaluate on, our results will look great but won't
    survive contact with reality. The split forces us to be honest:
    discover the strategy on one period, evaluate it on another.

    Default split:
        In-sample:     2013-01-01 to 2020-12-31  (~2,015 trading days)
        Out-of-sample: 2021-01-01 to 2024-12-31  (~1,007 trading days)
        Ratio: roughly 67/33

    The 2021+ out-of-sample window deliberately includes the 2023 regional
    banking crisis (SVB, Signature, First Republic). If our strategy breaks
    here, that's important to know and discuss honestly rather than hide.

    Returns
    -------
    (in_sample, out_sample) : tuple of pd.DataFrame
    """
    # .loc[:split_date] grabs everything up to and including split_date.
    # We then drop the last row with .iloc[:-1] so the split_date itself
    # is the FIRST observation of out-of-sample, not the last of in-sample.
    in_sample = prices.loc[:split_date].iloc[:-1]
    out_sample = prices.loc[split_date:]
    return in_sample, out_sample


# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------
# The `if __name__ == '__main__':` block only runs when you execute this
# file directly (e.g., right-click → Run in PyCharm). It does NOT run when
# another file imports from this one. This is the standard Python pattern
# for "if I'm being run as a script, do this; if I'm being imported as a
# module, don't."
if __name__ == '__main__':
    # Download everything: banks + SPY for benchmark comparison
    all_tickers = BANK_TICKERS + [BENCHMARK]
    prices = download_prices(tickers=all_tickers)

    print(f"\nLoaded {len(prices)} trading days across {len(prices.columns)} tickers")
    print("Last 5 rows of data:")
    print(prices.tail())

    # Show how the in/out split looks
    in_sample, out_sample = split_in_out_sample(prices)
    print(f"\nIn-sample:  {in_sample.index.min().date()} to "
          f"{in_sample.index.max().date()} ({len(in_sample)} days)")
    print(f"Out-sample: {out_sample.index.min().date()} to "
          f"{out_sample.index.max().date()} ({len(out_sample)} days)")
    total = len(in_sample) + len(out_sample)
    print(f"Split ratio: {len(in_sample)/total:.1%} / {len(out_sample)/total:.1%}")
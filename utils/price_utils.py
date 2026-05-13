"""
Price data utilities: yfinance wrappers, call timing classification, label logic.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time, timedelta
import pytz
import config

ET = pytz.timezone("US/Eastern")


def fetch_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a ticker in a date range.
    Returns DataFrame indexed by trading dates with 'Adj Close' column.
    """
    data = yf.download(
        ticker, start=start_date, end=end_date,
        progress=False, auto_adjust=False
    )
    if data.empty:
        return pd.DataFrame()
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data[["Adj Close"]].copy()


def get_price_window(ticker: str, call_date: str, window_days: int = 10) -> pd.DataFrame:
    """
    Fetch prices in a window around the earnings call date.
    Pads extra calendar days to account for weekends/holidays.
    """
    dt = pd.Timestamp(call_date)
    start = (dt - timedelta(days=window_days + 10)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=window_days + 10)).strftime("%Y-%m-%d")
    return fetch_prices(ticker, start, end)


def classify_call_timing(call_datetime_str: str) -> str:
    """
    Classify earnings call timing relative to market hours.

    Args:
        call_datetime_str: "YYYY-MM-DD HH:MM:SS" (assumed ET if no tz)

    Returns:
        "pre_market" | "during_market" | "after_close" | "unknown"
    """
    try:
        dt = pd.Timestamp(call_datetime_str)
    except (ValueError, TypeError):
        return "unknown"

    # If time is midnight (00:00:00), timestamp likely missing time component
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        return "unknown"

    call_time = dt.time()
    market_open = time(9, 30)
    market_close = time(16, 0)

    if call_time < market_open:
        return "pre_market"
    elif call_time <= market_close:
        return "during_market"
    else:
        return "after_close"


def get_pre_post_prices(prices_df: pd.DataFrame, call_date: str,
                        timing: str) -> tuple[float | None, float | None]:
    """
    Get pre-price and post-price based on call timing logic.

    Returns:
        (pre_price, post_price) or (None, None) if data unavailable
    """
    if prices_df.empty:
        return None, None

    call_dt = pd.Timestamp(call_date).normalize()
    trading_dates = prices_df.index.normalize()

    def nearest_trading_day_on_or_before(dt):
        mask = trading_dates <= dt
        if mask.any():
            return prices_df.index[mask][-1]
        return None

    def nearest_trading_day_on_or_after(dt):
        mask = trading_dates >= dt
        if mask.any():
            return prices_df.index[mask][0]
        return None

    def next_trading_day_after(dt):
        mask = trading_dates > dt
        if mask.any():
            return prices_df.index[mask][0]
        return None

    def prev_trading_day_before(dt):
        mask = trading_dates < dt
        if mask.any():
            return prices_df.index[mask][-1]
        return None

    if timing in ("pre_market", "during_market"):
        # Pre-price: previous trading day's close
        # Post-price: same day's close
        pre_day = prev_trading_day_before(call_dt)
        post_day = nearest_trading_day_on_or_after(call_dt)

    elif timing == "after_close":
        # Pre-price: same day's close (or most recent if not trading day)
        # Post-price: next trading day's close
        pre_day = nearest_trading_day_on_or_before(call_dt)
        post_day = next_trading_day_after(call_dt)

    elif timing == "unknown":
        # Heuristic: treat as after_close (most common for earnings calls)
        pre_day = nearest_trading_day_on_or_before(call_dt)
        post_day = next_trading_day_after(call_dt)
        # If that doesn't work, try during-market logic
        if pre_day is None or post_day is None:
            pre_day = prev_trading_day_before(call_dt)
            post_day = nearest_trading_day_on_or_after(call_dt)

    else:
        return None, None

    if pre_day is None or post_day is None:
        return None, None

    pre_price = float(prices_df.loc[pre_day, "Adj Close"])
    post_price = float(prices_df.loc[post_day, "Adj Close"])
    return pre_price, post_price


def compute_return(pre_price: float, post_price: float) -> float:
    """Compute simple return."""
    if pre_price == 0 or pre_price is None or post_price is None:
        return np.nan
    return (post_price - pre_price) / pre_price


def assign_label_ternary(ret: float, threshold: float = 0.005) -> str | None:
    """Assign ternary label: UP / DOWN / FLAT."""
    if np.isnan(ret):
        return None
    if ret > threshold:
        return "UP"
    elif ret < -threshold:
        return "DOWN"
    else:
        return "FLAT"


def assign_label_binary(ret: float, threshold: float = 0.005) -> str | None:
    """Assign binary label: UP / DOWN (FLAT excluded → None)."""
    if np.isnan(ret):
        return None
    if ret > threshold:
        return "UP"
    elif ret < -threshold:
        return "DOWN"
    else:
        return None  # FLAT excluded from binary


# ── Market-Adjusted Returns ──────────────────────────────────────────────────

def fetch_spy_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SPY (S&P 500 ETF) prices for market adjustment."""
    return fetch_prices("SPY", start_date, end_date)


def compute_excess_return(stock_return: float, market_return: float) -> float:
    """Compute market-adjusted (excess) return: stock_return - market_return."""
    if np.isnan(stock_return) or np.isnan(market_return):
        return np.nan
    return stock_return - market_return


def assign_label_ternary_excess(excess_ret: float, threshold: float = 0.005) -> str | None:
    """Assign ternary label based on excess return."""
    return assign_label_ternary(excess_ret, threshold)


def assign_label_binary_excess(excess_ret: float, threshold: float = 0.005) -> str | None:
    """Assign binary label based on excess return."""
    return assign_label_binary(excess_ret, threshold)


# ── Multi-Day Returns ────────────────────────────────────────────────────────

def get_post_price_nday(prices_df: pd.DataFrame, call_date: str,
                        timing: str, n_days: int = 1) -> float | None:
    """
    Get closing price n trading days after the earnings call.

    Args:
        prices_df: DataFrame with 'Adj Close' indexed by trading dates
        call_date: Date of the earnings call
        timing: "pre_market", "during_market", "after_close", or "unknown"
        n_days: Number of trading days after the event to get the price

    Returns:
        Closing price n_days after the event, or None if unavailable
    """
    if prices_df.empty:
        return None

    call_dt = pd.Timestamp(call_date).normalize()
    trading_dates = prices_df.index.normalize()

    # Determine the anchor day (same logic as get_pre_post_prices)
    if timing in ("pre_market", "during_market"):
        # Event before/during market: anchor = call_date
        mask = trading_dates >= call_dt
    elif timing in ("after_close", "unknown"):
        # Event after close: anchor = next trading day
        mask = trading_dates > call_dt
    else:
        return None

    future_dates = prices_df.index[mask]
    if len(future_dates) < n_days:
        return None

    target_day = future_dates[n_days - 1]
    return float(prices_df.loc[target_day, "Adj Close"])


def compute_multi_day_returns(prices_df: pd.DataFrame, call_date: str,
                              timing: str, pre_price: float,
                              horizons: list[int] = None) -> dict:
    """
    Compute returns for multiple horizons (1-day, 2-day, 3-day, etc.).

    Returns:
        {1: 0.012, 2: -0.005, 3: 0.023, ...} mapping horizon → return
    """
    horizons = horizons or config.RETURN_HORIZONS

    results = {}
    for h in horizons:
        post_price = get_post_price_nday(prices_df, call_date, timing, n_days=h)
        if post_price is not None and pre_price is not None and pre_price != 0:
            results[h] = (post_price - pre_price) / pre_price
        else:
            results[h] = np.nan
    return results

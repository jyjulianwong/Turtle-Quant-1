import pandas as pd
import numpy as np
from typing import Literal
from scipy.signal import savgol_filter  # pyrefly: ignore[missing-module-attribute]
from sklearn.linear_model import LinearRegression
from math import sqrt


def get_wick_direction(row: pd.Series) -> Literal["up", "down", "neutral"]:
    """Determine the direction of the wick.

    Args:
        row: The row to check.

    Returns:
        The direction of the wick.
    """

    upper_wick = row["High"] - max(row["Close"], row["Open"])
    lower_wick = min(row["Close"], row["Open"]) - row["Low"]
    if upper_wick > lower_wick * 1.2:
        return "up"
    if lower_wick > upper_wick * 1.2:
        return "down"
    return "neutral"


def is_support_resistance_zone(
    data: pd.DataFrame,
    idx: int,
    threshold: float = 0.005,
    method: Literal["pivot", "moving_average", "extrema"] = "extrema",
) -> bool:
    """Check if the current price is in a support or resistance zone.

    TODO: Doesn't work yet.

    Supports three methods:
    1. Pivot Point: Uses standard pivot point calculations based on previous day's H/L/C
    2. Moving Average: Uses moving average and standard deviation bands
    3. Extrema: Uses local minima and maxima, and draws lines through points with linear regression

    Args:
        data: The data to check, must contain 'High', 'Low', 'Close' columns.
        idx: The index of the current price.
        threshold: The threshold for determining if price is near a support/resistance level.
        method: Method to use ('pivot', 'moving_average', 'extrema').

    Returns:
        True if the current price is in a support or resistance zone, False otherwise.
    """
    # Minimum data requirements based on method
    min_data_required = 1 if method == "pivot" else 20
    if idx < min_data_required:
        return False

    # Calculate support/resistance levels using the specified method
    levels_df = _calc_support_resistance_levels(data, method=method)

    # Get S1 and R1 for the current index
    s1 = levels_df.loc[data.index[idx], "S1"]
    r1 = levels_df.loc[data.index[idx], "R1"]

    # Check for NaN values (can happen with moving average method for early data points)
    if pd.isna(s1) or pd.isna(r1):
        return False

    # Get current price
    current_price = data.loc[data.index[idx], "Close"]

    # Check if current price is near level-1 support or resistance only
    level_1_zones = [s1, r1]  # Only check S1 and R1
    return any(
        abs(current_price - level) / current_price < threshold  # noqa: E501 # pyrefly: ignore[bad-argument-type, unsupported-operation]
        for level in level_1_zones
    )


def get_support_resistance_level(
    data: pd.DataFrame,
    method: Literal["pivot", "moving_average", "extrema"] = "extrema",
    level: Literal["S1", "S2", "R1", "R2", "Pivot"] = "S1",
) -> pd.Series:
    """Get a specific support/resistance level as a pandas Series for plotting.

    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns.
        level: Which level to return ('S1', 'S2', 'R1', 'R2', or 'Pivot').
        method: Method to use for calculation ('pivot', 'moving_average', 'extrema').

    Returns:
        pandas Series with the specified support/resistance level values.
    """
    levels_df = _calc_support_resistance_levels(data, method=method)
    return levels_df[level]


def _calc_support_resistance_levels(
    data: pd.DataFrame,
    method: Literal["pivot", "moving_average", "extrema"] = "extrema",
) -> pd.DataFrame:
    """Calculate support and resistance levels using the specified method.

    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns.
        method: Method to use for calculation ('pivot', 'moving_average', 'extrema').

    Returns:
        DataFrame with columns: 'Pivot', 'S1', 'S2', 'R1', 'R2' aligned with the original data index.
    """
    if method == "pivot":
        return _calc_sup_res_levels_with_pivots(data)
    if method == "moving_average":
        return _calc_sup_res_levels_with_moving_average(data)
    if method == "extrema":
        return _calc_sup_res_levels_with_extrema(data)
    raise ValueError(f"Invalid method: {method}")


def _calc_sup_res_levels_with_moving_average(
    data: pd.DataFrame, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Calculate support and resistance levels using moving averages and standard deviation bands.

    This method uses moving averages to identify dynamic support and resistance levels:
    - Pivot (Middle Line): Simple Moving Average of Close prices
    - S1/R1: ±1 standard deviation from the moving average
    - S2/R2: ±2 standard deviations from the moving average

    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns.
        window: Period for moving average calculation (default: 20).
        num_std: Number of standard deviations for S2/R2 levels (default: 2.0).

    Returns:
        DataFrame with columns: 'Pivot', 'S1', 'S2', 'R1', 'R2' aligned with the original data index.
    """
    if len(data) < window:
        raise ValueError(
            f"Need at least {window} rows of data for moving average calculation"
        )

    # Calculate moving average as the pivot/middle line
    pivot = data["Close"].rolling(window=window).mean()

    # Calculate rolling standard deviation
    rolling_std = data["Close"].rolling(window=window).std()

    # Calculate support and resistance levels using standard deviations
    s1 = pivot - rolling_std  # Support 1 (1 std below MA)
    r1 = pivot + rolling_std  # Resistance 1 (1 std above MA)
    s2 = pivot - (num_std * rolling_std)  # Support 2 (2 std below MA)
    r2 = pivot + (num_std * rolling_std)  # Resistance 2 (2 std above MA)

    # Create result DataFrame
    result = pd.DataFrame(
        {"Pivot": pivot, "S1": s1, "S2": s2, "R1": r1, "R2": r2}, index=data.index
    )

    return result


def _calc_sup_res_levels_with_pivots(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate pivot point support and resistance levels for the entire dataset.

    This function computes pivot points, support levels (S1, S2), and resistance levels (R1, R2)
    for each day based on the previous day's High, Low, and Close values using vectorized operations.

    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns and datetime index.

    Returns:
        DataFrame with columns: 'Pivot', 'S1', 'S2', 'R1', 'R2' aligned with the original data index.
        The first row will contain NaN values since pivot calculations require previous day data.
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 rows of data to calculate pivot levels")

    # TODO: Respect CANDLE_UNIT.
    # Shift data by 1 to get previous day's values
    prev_high = data["High"].shift(6)
    prev_low = data["Low"].shift(6)
    prev_close = data["Close"].shift(6)

    # Calculate pivot point using vectorized operations
    pivot = (prev_high + prev_low + prev_close) / 3

    # Calculate support and resistance levels using vectorized operations
    s1 = 2 * pivot - prev_high  # Support 1
    r1 = 2 * pivot - prev_low  # Resistance 1
    s2 = pivot - (prev_high - prev_low)  # Support 2
    r2 = pivot + (prev_high - prev_low)  # Resistance 2

    # Create result DataFrame
    result = pd.DataFrame(
        {"Pivot": pivot, "S1": s1, "S2": s2, "R1": r1, "R2": r2}, index=data.index
    )

    return result


def _calc_sup_res_levels_with_extrema(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate support and resistance levels using extrema.

    This function computes support and resistance levels based on the highest and lowest prices in a given period.
    Refer to https://github.com/sohandillikar/SupportResistance.

    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns and datetime index.

    Returns:
        DataFrame with columns: 'Pivot', 'S1', 'S2', 'R1', 'R2' aligned with the original data index.
        The first row will contain NaN values since pivot calculations require previous day data.
    """

    def pythag(pt1, pt2):
        a_sq = (pt2[0] - pt1[0]) ** 2
        b_sq = (pt2[1] - pt1[1]) ** 2
        return sqrt(a_sq + b_sq)

    def regression_ceof(pts):
        X = np.array([pt[0] for pt in pts]).reshape(-1, 1)
        y = np.array([pt[1] for pt in pts])
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0], model.intercept_

    def local_min_max(pts):
        local_min = []
        local_max = []
        prev_pts = [(0, pts[0]), (1, pts[1])]
        for i in range(1, len(pts) - 1):
            append_to = ""
            if pts[i - 1] > pts[i] < pts[i + 1]:
                append_to = "min"
            elif pts[i - 1] < pts[i] > pts[i + 1]:
                append_to = "max"
            if append_to:
                if local_min or local_max:
                    prev_distance = pythag(prev_pts[0], prev_pts[1]) * 0.5
                    curr_distance = pythag(prev_pts[1], (i, pts[i]))
                    if curr_distance >= prev_distance:
                        prev_pts[0] = prev_pts[1]
                        prev_pts[1] = (i, pts[i])
                        if append_to == "min":
                            local_min.append((i, pts[i]))
                        else:
                            local_max.append((i, pts[i]))
                else:
                    prev_pts[0] = prev_pts[1]
                    prev_pts[1] = (i, pts[i])
                    if append_to == "min":
                        local_min.append((i, pts[i]))
                    else:
                        local_max.append((i, pts[i]))
        return local_min, local_max

    series = data["Close"]
    series.index = np.arange(series.shape[0])  # pyrefly: ignore[bad-argument-type]

    # TODO: Respect CANDLE_UNIT.
    n_months = max(series.shape[0] // (30 * 6), 1)
    smooth_window_size = int(2 * n_months)

    # TODO: Error processing strategy MultiplePattern: polyorder must be less than window_length.
    pts = savgol_filter(
        x=np.array(series), window_length=smooth_window_size, polyorder=3
    )

    local_min, local_max = local_min_max(pts)

    local_min_slope, local_min_int = regression_ceof(local_min)
    local_max_slope, local_max_int = regression_ceof(local_max)
    support = (local_min_slope * np.array(series.index)) + local_min_int
    resistance = (local_max_slope * np.array(series.index)) + local_max_int

    return pd.DataFrame(
        {
            "Pivot": series,
            "S1": support,
            "S2": support,
            "R1": resistance,
            "R2": resistance,
        },
        index=data.index,
    )

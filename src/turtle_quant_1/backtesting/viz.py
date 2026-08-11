"""Visualisation helpers for inspecting backtesting results in Jupyter notebooks."""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from turtle_quant_1.backtesting.models import TestCaseResults

_SIGNAL_COLORS = {"BUY": "limegreen", "HOLD": "steelblue", "SELL": "tomato"}


def _normalize_dt(series: pd.Series) -> pd.Series:
    """Return a tz-naive UTC pandas Series of datetimes for consistent comparison."""
    converted = pd.to_datetime(series, utc=True)
    return converted.dt.tz_localize(None).astype("datetime64[s]")


def _build_tick_params(
    price_df: pd.DataFrame, n_ticks: int = 10
) -> tuple[list[int], list[str]]:
    """Compute evenly-spaced integer tick positions and date-string labels.

    Args:
        price_df: Price DataFrame with a 'datetime' column and integer index.
        n_ticks: Approximate number of ticks to show on the x-axis.

    Returns:
        Tuple of (tick_positions, tick_labels).
    """
    n = len(price_df)
    step = max(1, n // n_ticks)
    positions = list(range(0, n, step))
    labels = (
        pd.to_datetime(price_df.loc[positions, "datetime"], utc=True)
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    return positions, labels


def _map_timestamps_to_index(
    timestamps: pd.Series, price_df: pd.DataFrame
) -> np.ndarray:
    """Map a series of datetimes to the nearest integer row index in price_df.

    Uses an asof-merge so each timestamp resolves to the closest preceding (or
    following) row in price_df, avoiding gaps caused by public holidays or
    irregular tick intervals.

    Args:
        timestamps: Series of datetime values to map.
        price_df: Reference price DataFrame with integer index and 'datetime' column.

    Returns:
        NumPy array of integer indices aligned with the input timestamps.
    """
    ref = (
        pd.DataFrame(
            {
                "ts": _normalize_dt(price_df["datetime"]).values,
                "row_idx": price_df.index.values,
            }
        )
        .sort_values("ts")
        .reset_index(drop=True)
    )

    ts_frame = (
        pd.DataFrame(
            {
                "ts": _normalize_dt(timestamps).values,
                "orig_order": np.arange(len(timestamps)),
            }
        )
        .sort_values("ts")
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(ts_frame, ref, on="ts", direction="nearest")
    return merged.sort_values("orig_order")["row_idx"].values


def plot_signals(
    results: TestCaseResults,
    symbol: str,
    show_portfolio_value: bool = True,
    title: Optional[str] = None,
    height: int = 600,
) -> go.Figure:
    """Plot the close price for a symbol with buy/sell markers overlaid.

    The x-axis uses the integer row index of the price DataFrame rather than
    wall-clock time, which eliminates visual gaps caused by weekends, public
    holidays, and other periods where no market data exists.

    Optionally shows a second panel with portfolio value over time.
    The returned Plotly figure is interactive: zoom, pan, and hover are all
    enabled by default in Jupyter notebooks.

    Args:
        results: The TestCaseResults returned by BacktestingEngine.run_backtest().
        symbol: The ticker symbol to plot (must be present in results.price_history).
        show_portfolio_value: Whether to show a second panel with portfolio value.
        title: Optional figure title. Defaults to "{symbol} – Buy / Sell signals".
        height: Total figure height in pixels.

    Returns:
        A plotly Figure object. Call .show() in a notebook to render it.

    Example (Jupyter notebook)::

        from turtle_quant_1.backtesting.viz import plot_signals

        results = engine.run_backtest()
        plot_signals(results, "AAPL").show()
    """
    if symbol not in results.price_history:
        raise ValueError(
            f"Symbol '{symbol}' not found in results.price_history. "
            f"Available symbols: {list(results.price_history.keys())}"
        )

    price_df = results.price_history[symbol].copy()
    # price_df already carries a clean integer index from reset_index(drop=True)
    # in BacktestingEngine.run_backtest() — use it directly as the x-axis.
    price_dates = pd.to_datetime(price_df["datetime"], utc=True).dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    txn_df = results.transaction_history.copy()
    if not txn_df.empty:
        txn_df = txn_df[txn_df["symbol"] == symbol]
    buys = txn_df[txn_df["action"] == "BUY"] if not txn_df.empty else pd.DataFrame()
    sells = txn_df[txn_df["action"] == "SELL"] if not txn_df.empty else pd.DataFrame()

    tick_positions, tick_labels = _build_tick_params(price_df)

    n_rows = 2 if show_portfolio_value else 1
    row_heights = [0.7, 0.3] if show_portfolio_value else [1.0]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
    )

    # Close price line
    fig.add_trace(
        go.Scatter(
            x=price_df.index,
            y=price_df["Close"],
            mode="lines",
            name="Close",
            line=dict(color="steelblue", width=1),
            customdata=price_dates,
            hovertemplate="%{customdata}<br>Close: $%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Buy markers
    if not buys.empty:
        buy_idx = _map_timestamps_to_index(buys["timestamp"], price_df)
        buy_dates = pd.to_datetime(buys["timestamp"], utc=True).dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        fig.add_trace(
            go.Scatter(
                x=buy_idx,
                y=buys["price"].values,
                mode="markers",
                name="BUY",
                marker=dict(symbol="triangle-up", color="limegreen", size=10),
                customdata=buy_dates.values,
                hovertemplate="%{customdata}<br>BUY @ $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Sell markers
    if not sells.empty:
        sell_idx = _map_timestamps_to_index(sells["timestamp"], price_df)
        sell_dates = pd.to_datetime(sells["timestamp"], utc=True).dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        fig.add_trace(
            go.Scatter(
                x=sell_idx,
                y=sells["price"].values,
                mode="markers",
                name="SELL",
                marker=dict(symbol="triangle-down", color="tomato", size=10),
                customdata=sell_dates.values,
                hovertemplate="%{customdata}<br>SELL @ $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_xaxes(
        tickvals=tick_positions,
        ticktext=tick_labels,
        row=1,
        col=1,
    )

    # Optional second panel: portfolio value
    if show_portfolio_value and not results.portfolio_value_history.empty:
        pv_df = results.portfolio_value_history.copy()
        pv_idx = _map_timestamps_to_index(pv_df["datetime"], price_df)
        pv_dates = pd.to_datetime(pv_df["datetime"], utc=True).dt.strftime(
            "%Y-%m-%d %H:%M"
        )

        fig.add_trace(
            go.Scatter(
                x=pv_idx,
                y=pv_df["portfolio_value"].values,
                mode="lines",
                name="Portfolio value",
                line=dict(color="darkorange", width=1),
                customdata=pv_dates.values,
                hovertemplate="%{customdata}<br>Portfolio: $%{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Portfolio ($)", row=2, col=1)
        fig.update_xaxes(
            tickvals=tick_positions,
            ticktext=tick_labels,
            row=2,
            col=1,
        )

    fig.update_layout(
        title=title or f"{symbol} – Buy / Sell signals",
        height=height,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def plot_signal_distribution(
    results: TestCaseResults,
    symbol: str,
    height: int = 400,
) -> go.Figure:
    """Plot a timeline scatter of BUY / HOLD / SELL signals against the price index.

    The x-axis uses the integer row index of the price DataFrame for the given
    symbol, eliminating time gaps caused by weekends and public holidays. Each
    signal is drawn as a coloured marker at its mapped row index; the y-axis
    shows the signal score. Aggregate counts per action are shown in the legend
    labels.

    Args:
        results: The TestCaseResults returned by BacktestingEngine.run_backtest().
        symbol: Ticker symbol used to resolve the integer price index. Must be
            present in results.price_history.
        height: Figure height in pixels.

    Returns:
        A plotly Figure object. Call .show() in a notebook to render it.
    """
    if symbol not in results.price_history:
        raise ValueError(
            f"Symbol '{symbol}' not found in results.price_history. "
            f"Available symbols: {list(results.price_history.keys())}"
        )

    price_df = results.price_history[symbol].copy()

    sig_df = results.signal_history.copy()
    sig_df = sig_df[sig_df["symbol"] == symbol].copy()

    tick_positions, tick_labels = _build_tick_params(price_df)

    fig = go.Figure()

    for action in ["BUY", "HOLD", "SELL"]:
        subset = sig_df[sig_df["action"] == action]
        if subset.empty:
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    name=f"{action} (0)",
                    marker=dict(color=_SIGNAL_COLORS[action], size=6, opacity=0.7),
                )
            )
            continue

        sig_idx = _map_timestamps_to_index(subset["datetime"], price_df)
        sig_dates = pd.to_datetime(subset["datetime"], utc=True).dt.strftime(
            "%Y-%m-%d %H:%M"
        )

        fig.add_trace(
            go.Scatter(
                x=sig_idx,
                y=subset["score"].values,
                mode="markers",
                name=f"{action} ({len(subset)})",
                marker=dict(color=_SIGNAL_COLORS[action], size=6, opacity=0.7),
                customdata=sig_dates.values,
                hovertemplate="%{customdata}<br>"
                + f"{action}"
                + "<br>Score: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Signal distribution ({symbol})",
        xaxis=dict(
            title="Row index",
            tickvals=tick_positions,
            ticktext=tick_labels,
        ),
        yaxis_title="Score",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig

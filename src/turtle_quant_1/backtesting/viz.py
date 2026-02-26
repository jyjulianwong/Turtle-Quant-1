"""Visualisation helpers for inspecting backtesting results in Jupyter notebooks."""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from turtle_quant_1.backtesting.models import TestCaseResults


def _to_utc(series: pd.Series) -> pd.Series:
    """Normalise a datetime series to UTC-aware timestamps."""
    converted = pd.to_datetime(series, utc=True)
    return converted


def plot_signals(
    results: TestCaseResults,
    symbol: str,
    show_portfolio_value: bool = True,
    title: Optional[str] = None,
    height: int = 600,
) -> go.Figure:
    """Plot the close price for a symbol with buy/sell markers overlaid.

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
    price_df["datetime"] = _to_utc(price_df["datetime"])

    txn_df = results.transaction_history.copy()
    if not txn_df.empty:
        txn_df["timestamp"] = _to_utc(txn_df["timestamp"])
        txn_df = txn_df[txn_df["symbol"] == symbol]
    buys = txn_df[txn_df["action"] == "BUY"] if not txn_df.empty else pd.DataFrame()
    sells = txn_df[txn_df["action"] == "SELL"] if not txn_df.empty else pd.DataFrame()

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
            x=price_df["datetime"],
            y=price_df["Close"],
            mode="lines",
            name="Close",
            line=dict(color="steelblue", width=1),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Close: $%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Buy markers
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["timestamp"],
                y=buys["price"],
                mode="markers",
                name="BUY",
                marker=dict(symbol="triangle-up", color="limegreen", size=10),
                hovertemplate=("%{x|%Y-%m-%d %H:%M}<br>BUY @ $%{y:.2f}<extra></extra>"),
            ),
            row=1,
            col=1,
        )

    # Sell markers
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["timestamp"],
                y=sells["price"],
                mode="markers",
                name="SELL",
                marker=dict(symbol="triangle-down", color="tomato", size=10),
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M}<br>SELL @ $%{y:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)

    # Optional second panel: portfolio value
    if show_portfolio_value and not results.portfolio_value_history.empty:
        pv_df = results.portfolio_value_history.copy()
        pv_df["datetime"] = _to_utc(pv_df["datetime"])

        fig.add_trace(
            go.Scatter(
                x=pv_df["datetime"],
                y=pv_df["portfolio_value"],
                mode="lines",
                name="Portfolio value",
                line=dict(color="darkorange", width=1),
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M}<br>Portfolio: $%{y:.2f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Portfolio ($)", row=2, col=1)

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
    symbol: Optional[str] = None,
    height: int = 400,
) -> go.Figure:
    """Plot a bar chart showing the distribution of BUY / HOLD / SELL signals.

    Args:
        results: The TestCaseResults returned by BacktestingEngine.run_backtest().
        symbol: If provided, only include signals for this symbol.
        height: Figure height in pixels.

    Returns:
        A plotly Figure object. Call .show() in a notebook to render it.
    """
    sig_df = results.signal_history.copy()

    if symbol is not None:
        sig_df = sig_df[sig_df["symbol"] == symbol]

    counts = (
        sig_df["action"].value_counts().reindex(["BUY", "HOLD", "SELL"], fill_value=0)
    )

    colors = {"BUY": "limegreen", "HOLD": "steelblue", "SELL": "tomato"}

    fig = go.Figure(
        go.Bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker_color=[colors[a] for a in counts.index],
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )

    subtitle = f" ({symbol})" if symbol else ""
    fig.update_layout(
        title=f"Signal distribution{subtitle}",
        xaxis_title="Signal action",
        yaxis_title="Count",
        height=height,
    )

    return fig

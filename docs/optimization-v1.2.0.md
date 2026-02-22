# Optimization (v1.2.0)

This document explains compute optimization work that has been done for v1.2.0.

## Pre-Optimization Program Profiling Analysis

The program profiling notebook ([notebooks/snakeviz.ipynb](notebooks/snakeviz.ipynb)) was run with `TURTLEQUANT1_MAX_WORKERS=0` to ensure all methods were profiled.

| Strategies | Symbols | Timestamps | Run Time (mins) |
|------------|---------|------------|-----------------|
| 12         | 9       | 1344       | 73.5            |

The method names that were consuming the majority of "unexplained" time were:
- `convert_to_daily_data`
    - `DataFrame.apply`
- `_is_price_in_sup_res_zone_vecd`
    - `stack`
    - `cumsum`
    - `join`
- `get_wick_direction_vecd`
- `_get_divergence_signals`
    - `__itemget__`

## Improvements Implemented

- Removed unnecessary DataFrame copying operations as most Pandas operations are now implemented with Copy-on-Write (CoW) or an equivalent guarantee.
- Removed DataFrame sorting operations by making the assumption that all input data to strategies have already been sorted by `datetime`.
- Reduced the number of data points each strategy has to process for prediction by capping history length depending on each strategy's lookback requirements.
- Enabled caching of daily and weekly OHLCV data for each symbol to avoid re-aggregating the same data across all strategies.

## Post-Optimization Program Profiling Analysis

The program profiling notebook ([notebooks/snakeviz.ipynb](notebooks/snakeviz.ipynb)) was run with `TURTLEQUANT1_MAX_WORKERS=0` to ensure all methods were profiled.

| Strategies | Symbols | Timestamps | Run Time (mins) |
|------------|---------|------------|-----------------|
| 12         | 9       | 1344       | 47.5            |
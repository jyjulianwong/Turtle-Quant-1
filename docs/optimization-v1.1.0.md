# Optimization (v1.1.0)

This document explains compute optimization work that has been done since the release of v1.1.0.

## Program Profiling Analysis

The program profiling notebook ([notebooks/snakeviz.ipynb](notebooks/snakeviz.ipynb)) was run with `TURTLEQUANT1_MAX_WORKERS=0` to ensure all methods were profiled.

| Strategies | Symbols | Timestamps | Run Time (mins) |
|------------|---------|------------|-----------------|
| 12         | 9       | 1344       | 73.5            |

The method names that were consuming the majority of "unexplained" time were:
- `convert_to_daily_data`
    - `DataFrame.apply`
- `is_price_in_sup_res_zone_vectorized`
    - `stack`
    - `cumsum`
    - `join`
- `get_wick_direction_vectorized`
- `_get_divergence_signals`
    - `__itemget__`

## Improvements

### Miscellaneous

- Removed unnecessary DataFrame copying operations as most Pandas operations are now implemented with Copy-on-Write (CoW) or an equivalent guarantee.
- Removed DataFrame sorting operations and made assumption that all input data to strategies have already been sorted by `datetime`.
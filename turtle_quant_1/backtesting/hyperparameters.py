"""Hyperparameter tuning script for trading strategies using Optuna."""

import logging
from typing import Dict, Any

import optuna
from turtle_quant_1.backtesting.engine import BacktestingEngine
from turtle_quant_1.backtesting.models import TestCaseResults
from turtle_quant_1.strategies.engine import StrategyEngine

# NOTE: Import all strategy classes so they're available in globals()
from turtle_quant_1.strategies.mean_reversion import *  # noqa: F401, F403
from turtle_quant_1.strategies.momentum import *  # noqa: F401, F403

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config(trial: optuna.Trial) -> Dict[str, Any]:
    """Create a strategy engine configuration for the given trial.

    Args:
        trial: Optuna trial object for hyperparameter suggestions

    Returns:
        Dictionary containing strategy engine configuration
    """

    # NOTE: All strategies are included by default.
    # TODO: Add a way to exclude strategies.
    strategies = {
        "BollingerBand": {
            "window": trial.suggest_int("bb_window", 20, 200),
            "n_std": trial.suggest_int("bb_n_std", 1, 3),
        },
        "MovingAverageCrossover": {
            "sma_candles": trial.suggest_int("mac_sma_candles", 5, 50),
            "lma_candles": trial.suggest_int("mac_lma_candles", 5, 50),
        },
        "RelativeStrengthIndex": {"candles": trial.suggest_int("rsi_candles", 14, 200)},
        "LinearRegression": {
            "lookback_candles": trial.suggest_int("lr_lookback_candles", 20, 200)
        },
    }

    # Strategy engine configuration
    # NOTE: Weights are not part of the hyperparameters of the strategy engine.
    config = {
        "strategies": strategies,
        "buy_unit_threshold": trial.suggest_float("buy_unit_threshold", 0.1, 0.9),
        "sell_threshold": trial.suggest_float("sell_threshold", -0.9, -0.1),
    }

    return config


def run_backtest(
    config: Dict[str, Any], initial_capital: float = 10000.0
) -> TestCaseResults:
    """Run a backtest with the given strategy engine configuration.

    Args:
        config: Strategy engine configuration dictionary
        initial_capital: Starting capital for the backtest

    Returns:
        BacktestingResults object
    """
    # Create strategy engine from config
    strategy_engine = StrategyEngine.from_config(config)

    # Create backtesting engine
    backtesting_engine = BacktestingEngine(
        strategy_engine=strategy_engine,
        initial_capital=initial_capital,
        max_lookback_days=60,  # NOTE: Shorter for faster optimization
        max_lookforward_days=60,  # NOTE: Shorter for faster optimization
    )

    # Run backtest
    results = backtesting_engine.run_backtest()

    # Get evaluation metrics
    metrics = backtesting_engine.get_metrics(benchmark="SPY")

    # Combine results
    results.metrics = metrics

    return results


def get_objective_metric(
    results: TestCaseResults, metric_name: str = "sharpe_ratio"
) -> float:
    """Extract the objective metric from backtest results.

    NOTE: This is currently a wrapper around the quantstats metrics in the results.
    NOTE: Ensure that metric_name is a valid quantstats metric.
    TODO: Implement our own metrics.

    Args:
        results: BacktestingResults object
        metric_name: Name of the metric to optimize

    Returns:
        Metric value (returns large negative value if error occurred)
    """
    metrics = results.metrics
    if not metrics:
        return -1000.0  # Large negative value for failed runs

    # Try to get the requested metric from quantstats
    if metric_name in metrics:
        value = metrics[metric_name]
        # Handle potential NaN or infinite values
        if isinstance(value, (int, float)) and not (
            value != value or abs(value) == float("inf")
        ):
            return float(value)

    # Fallback metrics from basic backtest results
    fallback_metrics = {
        "total_return_percent": results.total_return_percent,
        "total_return_dollars": results.total_return_dollars,
    }

    if metric_name in fallback_metrics:
        return fallback_metrics[metric_name]

    # Default to total return percentage if metric not found
    return results.total_return_percent


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object

    Returns:
        Objective metric value to maximize
    """
    # Create strategy configuration
    config = get_config(trial)

    # Run backtest
    results = run_backtest(config)

    # Extract objective metric (try Sharpe ratio, fallback to total return)
    sharpe = get_objective_metric(results, "sharpe_ratio")
    if sharpe == -1000.0:  # Error occurred, try fallback
        return get_objective_metric(results, "total_return_percent")

    return sharpe


def run_hyperparameter_tuning(
    n_trials: int,
    study_name: str = "strategy_engine_optimization",
    direction: str = "maximize",
) -> optuna.Study:
    """Run hyperparameter optimization using Optuna.

    Args:
        n_trials: Number of optimization trials
        study_name: Name for the study
        direction: Optimization direction ("maximize" or "minimize")

    Returns:
        Completed Optuna study object
    """
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")

    # Create study
    study = optuna.create_study(direction=direction, study_name=study_name)

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Print results
    logger.info("Optimization complete!")
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best value: {study.best_value}")

    # Print top 5 trials
    logger.info("Top 5 trials:")
    top_trials = study.trials_dataframe().nlargest(5, "value")
    for i, (_, trial_row) in enumerate(top_trials.iterrows(), 1):
        # Get the trial object to access parameters
        trial_number = int(trial_row["number"])
        trial_obj = study.trials[trial_number]
        logger.info(
            f"  {i}. Value: {trial_row['value']:.4f}, Parameters: {trial_obj.params}"
        )

    return study


if __name__ == "__main__":
    # Run optimization
    study = run_hyperparameter_tuning(n_trials=10)

    # Test the best configuration
    logger.info("Testing best configuration...")
    best_config = get_config(study.best_trial)  # pyrefly: ignore[bad-argument-type]
    best_results = run_backtest(best_config, initial_capital=10000.0)

    logger.info("Best configuration test results:")
    logger.info(f"  Total Return: ${best_results.total_return_dollars:.2f}")
    logger.info(f"  Return Percentage: {best_results.total_return_percent:.2f}%")
    logger.info(f"  Final Portfolio Value: ${best_results.final_portfolio_value:.2f}")

    metrics = best_results.metrics
    if metrics:
        logger.info("  Metrics:")
        for key, value in metrics.items():
            logger.info(f"    {key}: {value}")

"""Strategy engine for aggregating multiple strategies and generating trading signals."""

from typing import Dict, List, Optional

import pandas as pd

from turtle_quant_1.strategies.base import (
    BaseStrategy,
    BaseStrategyEngine,
    Signal,
    SignalAction,
)


class StrategyEngine(BaseStrategyEngine):
    """Strategy engine that aggregates multiple strategies to generate trading signals.

    The engine combines multiple strategy scores using weighted averages and converts
    the final aggregated score to actionable trading signals (BUY, HOLD, SELL).
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        weights: Optional[List[float]] = None,
        buy_threshold: float = 0.3,
        sell_threshold: float = -0.3,
    ):
        """Initialize the strategy engine.

        Args:
            strategies: List of strategies to aggregate.
            weights: Optional list of weights for each strategy. If None, equal weights are used.
            buy_threshold: Minimum aggregated score to generate BUY signal (default: 0.3).
            sell_threshold: Maximum aggregated score to generate SELL signal (default: -0.3).
        """
        super().__init__(strategies, weights)

        if not (-1.0 <= sell_threshold <= buy_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy: -1.0 <= sell_threshold <= buy_threshold <= 1.0"
            )

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def get_configuration(self) -> Dict:
        """Get current engine configuration.

        Returns:
            Dictionary with current configuration settings.
        """
        return {
            "num_strategies": len(self.strategies),
            "strategy_names": [strategy.name for strategy in self.strategies],
            "weights": self.weights,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
        }

    def update_thresholds(self, buy_threshold: float, sell_threshold: float) -> None:
        """Update the buy and sell thresholds.

        Args:
            buy_threshold: New buy threshold.
            sell_threshold: New sell threshold.
        """
        if not (-1.0 <= sell_threshold <= buy_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy: -1.0 <= sell_threshold <= buy_threshold <= 1.0"
            )

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def update_weights(self, weights: List[float]) -> None:
        """Update strategy weights.

        Args:
            weights: New list of weights for each strategy.
        """
        if len(weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        self.weights = weights

    def analyze_symbol(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Perform comprehensive analysis of a symbol using all strategies.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Dictionary containing detailed analysis results.
        """
        # Get individual strategy scores
        strategy_scores = self.get_breakdown(data, symbol)

        # Get aggregated score and signal
        aggregated_score = self.aggregate_scores(data, symbol)
        signal = self.generate_signal(data, symbol)

        # Calculate weighted contributions
        weighted_contributions = {}
        for strategy, weight in zip(self.strategies, self.weights):
            strategy_name = strategy.name
            if strategy_name in strategy_scores:
                weighted_contributions[strategy_name] = {
                    "raw_score": strategy_scores[strategy_name],
                    "weight": weight,
                    "weighted_score": strategy_scores[strategy_name] * weight,
                }

        return {
            "symbol": symbol,
            "signal": signal.action.value,
            "signal_object": signal,
            "aggregated_score": aggregated_score,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "strategy_scores": strategy_scores,
            "weighted_contributions": weighted_contributions,
            "num_strategies": len(self.strategies),
            "data_points": len(data),
        }

    def get_signal_confidence(self, data: pd.DataFrame, symbol: str) -> float:
        """Post-analysis metric. Calculate confidence level for the generated signal.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Confidence level between 0.0 (low confidence) and 1.0 (high confidence).
        """
        aggregated_score = self.aggregate_scores(data, symbol)

        # Calculate distance from neutral (0) as a proxy for confidence
        # The further from 0, the higher the confidence
        confidence = abs(aggregated_score)

        return min(confidence, 1.0)

    def get_signal_agreement(self, data: pd.DataFrame, symbol: str) -> float:
        """Post-analysis metric. Calculate agreement level between strategies.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Agreement level between 0.0 (complete disagreement) and 1.0 (perfect agreement).
        """
        strategy_scores = self.get_breakdown(data, symbol)
        scores = list(strategy_scores.values())

        if len(scores) <= 1:
            return 1.0  # Perfect agreement with only one strategy

        # Calculate variance of scores as a measure of disagreement
        import numpy as np

        score_variance = np.var(scores)

        # Convert variance to agreement (lower variance = higher agreement)
        # Max possible variance for scores in [-1, 1] is 1.0 (when scores are -1 and +1)
        max_variance = 1.0
        agreement = 1.0 - min(score_variance / max_variance, 1.0)

        return agreement

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        """Generate a trading signal by aggregating all strategies.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Signal object containing action, score, and contributing strategies.
        """
        # Get aggregated score from all strategies
        aggregated_score = self.aggregate_scores(data, symbol)

        # Convert score to trading signal based on thresholds
        if aggregated_score >= self.buy_threshold:
            action = SignalAction.BUY
        elif aggregated_score <= self.sell_threshold:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        # Get strategy names
        strategy_names = [strategy.name for strategy in self.strategies]

        # Create and return Signal object
        return Signal(strategies=strategy_names, action=action, score=aggregated_score)

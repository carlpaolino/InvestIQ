"""
Trading strategy implementations for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for trading strategies."""
    
    @abstractmethod
    def generate_signal(self, prediction: float) -> float:
        """
        Generate trading signal based on prediction.
        
        Args:
            prediction: Model prediction (expected return)
            
        Returns:
            Position size (-1 to 1, where 1 = fully invested, 0 = cash, -1 = short)
        """
        pass
    
    def get_name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__


class SimpleStrategy(BaseStrategy):
    """Simple threshold-based strategy."""
    
    def __init__(self, threshold: float = 0.02, max_position: float = 1.0):
        """
        Initialize simple strategy.
        
        Args:
            threshold: Minimum prediction threshold for taking position
            max_position: Maximum position size
        """
        self.threshold = threshold
        self.max_position = max_position
    
    def generate_signal(self, prediction: float) -> float:
        """Generate signal based on prediction threshold."""
        if prediction > self.threshold:
            return self.max_position
        elif prediction < -self.threshold:
            return -self.max_position
        else:
            return 0.0


class ThresholdStrategy(BaseStrategy):
    """Multi-threshold strategy with different position sizes."""
    
    def __init__(
        self,
        strong_buy_threshold: float = 0.05,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.02,
        strong_sell_threshold: float = -0.05,
        strong_position: float = 1.0,
        normal_position: float = 0.5
    ):
        """
        Initialize threshold strategy.
        
        Args:
            strong_buy_threshold: Threshold for strong buy signal
            buy_threshold: Threshold for buy signal
            sell_threshold: Threshold for sell signal
            strong_sell_threshold: Threshold for strong sell signal
            strong_position: Position size for strong signals
            normal_position: Position size for normal signals
        """
        self.strong_buy_threshold = strong_buy_threshold
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.strong_sell_threshold = strong_sell_threshold
        self.strong_position = strong_position
        self.normal_position = normal_position
    
    def generate_signal(self, prediction: float) -> float:
        """Generate signal based on multiple thresholds."""
        if prediction >= self.strong_buy_threshold:
            return self.strong_position
        elif prediction >= self.buy_threshold:
            return self.normal_position
        elif prediction <= self.strong_sell_threshold:
            return -self.strong_position
        elif prediction <= self.sell_threshold:
            return -self.normal_position
        else:
            return 0.0


class MomentumStrategy(BaseStrategy):
    """Momentum-based strategy using prediction trends."""
    
    def __init__(
        self,
        lookback_periods: int = 5,
        momentum_threshold: float = 0.01,
        max_position: float = 1.0
    ):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_periods: Number of periods to look back for momentum
            momentum_threshold: Minimum momentum threshold
            max_position: Maximum position size
        """
        self.lookback_periods = lookback_periods
        self.momentum_threshold = momentum_threshold
        self.max_position = max_position
        self.prediction_history = []
    
    def generate_signal(self, prediction: float) -> float:
        """Generate signal based on prediction momentum."""
        self.prediction_history.append(prediction)
        
        # Keep only recent history
        if len(self.prediction_history) > self.lookback_periods:
            self.prediction_history = self.prediction_history[-self.lookback_periods:]
        
        # Need enough history for momentum calculation
        if len(self.prediction_history) < self.lookback_periods:
            return 0.0
        
        # Calculate momentum (slope of predictions)
        x = np.arange(len(self.prediction_history))
        y = np.array(self.prediction_history)
        momentum = np.polyfit(x, y, 1)[0]  # Linear regression slope
        
        # Generate signal based on momentum
        if momentum > self.momentum_threshold:
            return self.max_position
        elif momentum < -self.momentum_threshold:
            return -self.max_position
        else:
            return 0.0


class AdaptiveStrategy(BaseStrategy):
    """Adaptive strategy that adjusts based on recent performance."""
    
    def __init__(
        self,
        base_threshold: float = 0.02,
        performance_lookback: int = 20,
        adaptation_rate: float = 0.1,
        min_threshold: float = 0.005,
        max_threshold: float = 0.05
    ):
        """
        Initialize adaptive strategy.
        
        Args:
            base_threshold: Base prediction threshold
            performance_lookback: Periods to look back for performance
            adaptation_rate: Rate of threshold adaptation
            min_threshold: Minimum threshold value
            max_threshold: Maximum threshold value
        """
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold
        self.performance_lookback = performance_lookback
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.recent_performance = []
    
    def generate_signal(self, prediction: float) -> float:
        """Generate signal with adaptive threshold."""
        # Simple signal generation
        if prediction > self.current_threshold:
            signal = 1.0
        elif prediction < -self.current_threshold:
            signal = -1.0
        else:
            signal = 0.0
        
        return signal
    
    def update_performance(self, actual_return: float, strategy_return: float):
        """Update strategy with recent performance for adaptation."""
        self.recent_performance.append({
            'actual_return': actual_return,
            'strategy_return': strategy_return
        })
        
        # Keep only recent performance
        if len(self.recent_performance) > self.performance_lookback:
            self.recent_performance = self.recent_performance[-self.performance_lookback:]
        
        # Adapt threshold based on performance
        if len(self.recent_performance) >= self.performance_lookback:
            self._adapt_threshold()
    
    def _adapt_threshold(self):
        """Adapt threshold based on recent performance."""
        # Calculate strategy vs buy-and-hold performance
        strategy_returns = [p['strategy_return'] for p in self.recent_performance]
        actual_returns = [p['actual_return'] for p in self.recent_performance]
        
        strategy_cumulative = np.prod([1 + r for r in strategy_returns])
        actual_cumulative = np.prod([1 + r for r in actual_returns])
        
        # If strategy underperforming, increase threshold (be more selective)
        if strategy_cumulative < actual_cumulative:
            self.current_threshold = min(
                self.max_threshold,
                self.current_threshold * (1 + self.adaptation_rate)
            )
        else:
            # If strategy outperforming, decrease threshold (be less selective)
            self.current_threshold = max(
                self.min_threshold,
                self.current_threshold * (1 - self.adaptation_rate)
            )


class RiskAdjustedStrategy(BaseStrategy):
    """Risk-adjusted strategy that considers volatility."""
    
    def __init__(
        self,
        base_threshold: float = 0.02,
        volatility_lookback: int = 20,
        risk_adjustment_factor: float = 1.0,
        max_position: float = 1.0
    ):
        """
        Initialize risk-adjusted strategy.
        
        Args:
            base_threshold: Base prediction threshold
            volatility_lookback: Periods for volatility calculation
            risk_adjustment_factor: Factor to adjust for risk
            max_position: Maximum position size
        """
        self.base_threshold = base_threshold
        self.volatility_lookback = volatility_lookback
        self.risk_adjustment_factor = risk_adjustment_factor
        self.max_position = max_position
        self.recent_returns = []
    
    def generate_signal(self, prediction: float) -> float:
        """Generate risk-adjusted signal."""
        # Calculate current volatility
        if len(self.recent_returns) >= self.volatility_lookback:
            volatility = np.std(self.recent_returns[-self.volatility_lookback:])
            # Adjust threshold based on volatility
            adjusted_threshold = self.base_threshold * (1 + volatility * self.risk_adjustment_factor)
        else:
            adjusted_threshold = self.base_threshold
        
        # Generate signal
        if prediction > adjusted_threshold:
            return self.max_position
        elif prediction < -adjusted_threshold:
            return -self.max_position
        else:
            return 0.0
    
    def update_returns(self, actual_return: float):
        """Update with recent returns for volatility calculation."""
        self.recent_returns.append(actual_return)
        
        # Keep only recent returns
        if len(self.recent_returns) > self.volatility_lookback:
            self.recent_returns = self.recent_returns[-self.volatility_lookback:]

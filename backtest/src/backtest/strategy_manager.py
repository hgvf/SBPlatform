import backtrader as bt
from typing import Dict, Type

from .strategies import *

class strategyManager:
    """
    負責 load strategy + 動態隨使用者調整策略相關參數
    """

    STRATEGY_MAP: Dict[str, Type[bt.Strategy]] = {
        'MultiIndicatorStrategy': MultiIndicatorStrategy,
    }

    def __init__(self, 
                 strategy_func: str = None):
        
        self.strategy = None
        self.load_strategy(strategy_func)

    def getStrategy(self) -> bt.Strategy:
        """回傳 bt.Strategy class object"""

        return self.strategy
    
    def load_strategy(self, strategy_func):
        """宣告 bt.Strategy class object"""

        if strategy_func is not None:
            self.strategy = self.STRATEGY_MAP[strategy_func]

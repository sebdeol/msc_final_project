from collections import deque
from typing import Tuple, Union

import numpy as np


class FixedAgent:
    def __init__(self, agent_id, short_window=10, long_window=50, threshold=0.01) -> None:
        """
        Initialise the fixed agent class.
        - short_window: Short-term moving average window
        - long_window: Long-term moving average window
        - threshold: Threshold for mean reversion
        """

        self.agent_id = agent_id
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold
        self.price_history = deque(maxlen=max(long_window, short_window))

    def update(self, price: float) -> None:
        """
        Update price history.
        """

        self.price_history.append(price)

    def get_action(self, current_price: float) -> Union[Tuple[str, float, int], None]:
        """
        Get action based on mean-reversion strategy.
        Returns (order_type, price, quantity) or None.
        """

        if len(self.price_history) < self.long_window:
            return None

        prices = np.array(self.price_history)
        short_ma = np.mean(prices[-self.short_window :])

        if current_price < short_ma * (1 - self.threshold):
            return ("buy", current_price, 10)
        elif current_price > short_ma * (1 + self.threshold):
            return ("sell", current_price, 10)

        return None

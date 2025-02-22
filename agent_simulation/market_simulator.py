from typing import Tuple

import numpy as np


class MarketSimulator:
    def __init__(self, monthly_data, num_ticks_per_month=1000, dt=1 / 252) -> None:
        """
        Initialise the market simulator class.
        - monthly_data: DataFrame with columns
        - num_ticks_per_month: Number of ticks to simulate per month
        - dt: Time step (fraction of a day)
        """

        self.monthly_data = monthly_data
        self.num_ticks = num_ticks_per_month
        self.dt = dt
        self.current_month_idx = 0
        self.current_tick = 0
        self.price = 100.0
        self.prices = []
        self.flash_crash_prob = None
        self.volatility = None
        self.bid_ask_spread = None

    def reset(self, month_idx: int) -> None:
        """
        Reset the simulator for a new month.
        """
        self.current_month_idx = month_idx
        self.current_tick = 0
        self.price = 100.0
        self.prices = [self.price]
        row = self.monthly_data.iloc[month_idx]
        self.volatility = row["intraday_volatility"] / 100
        self.bid_ask_spread = row["bid_ask_spread"] / 10000
        self.flash_crash_prob = row["flash_crash_events_count"] / self.num_ticks

    def step(self) -> Tuple[float, float]:
        """
        Generate the next price tick.
        """

        if self.current_tick >= self.num_ticks:
            raise ValueError("End of ticks for this month.")

        # Simulate price using GBM
        drift = 0.0  # No drift for simplicity
        sigma = self.volatility * np.sqrt(252)  # Annualised volatility
        dW = np.random.normal(0, np.sqrt(self.dt))
        self.price *= np.exp((drift - 0.5 * sigma**2) * self.dt + sigma * dW)

        # Simulate flash crash
        if np.random.random() < self.flash_crash_prob:
            self.price *= 0.9  # 10% drop

        self.prices.append(self.price)
        self.current_tick += 1
        return self.price, self.bid_ask_spread

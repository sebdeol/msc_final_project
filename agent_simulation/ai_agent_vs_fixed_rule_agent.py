from collections import deque
from typing import Tuple

import gym
import numpy as np
from gym import spaces

from agent_simulation.fixed_agent import FixedAgent
from agent_simulation.market_simulator import MarketSimulator
from agent_simulation.order_book import OrderBook
from agent_simulation.portfolio import Portfolio


class AIAgentVsFixedRuleAgent(gym.Env):
    def __init__(self, market_data, num_ticks_per_month=1000) -> None:
        """
        Initialise the AIAgentVsFixedRuleAgent class.
        """

        super(AIAgentVsFixedRuleAgent, self).__init__()

        self.market_data = market_data
        self.num_ticks = num_ticks_per_month
        self.market_sim = MarketSimulator(market_data, num_ticks_per_month)
        self.order_book = OrderBook()
        self.ai_portfolio = Portfolio()
        self.fixed_portfolio = Portfolio()
        self.fixed_agent = FixedAgent(agent_id=1)
        self.current_month = 0
        self.current_tick = 0
        self.price_history = deque(maxlen=50)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0, -0.05, 0]), high=np.array([2, 0.05, 100]), dtype=np.float32)

    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new month.
        """

        self.current_month = np.random.randint(0, len(self.market_data))
        self.market_sim.reset(self.current_month)
        self.order_book = OrderBook()
        self.ai_portfolio = Portfolio()
        self.fixed_portfolio = Portfolio()
        self.fixed_agent = FixedAgent(agent_id=1)
        self.current_tick = 0
        self.price_history.clear()
        return self._get_observation()

    def step(self, action) -> Tuple:
        """
        Step the environment.
        - action: [order_type, price_offset, quantity]
        """

        current_price, bid_ask_spread = self.market_sim.step()
        self.price_history.append(current_price)
        self.fixed_agent.update(current_price)

        # Process fixed agent action
        fixed_action = self.fixed_agent.get_action(current_price)
        if fixed_action:
            order_type, price, qty = fixed_action
            trades = self.order_book.place_order(1, order_type, price, qty)
            for trade_price, trade_qty, buyer_id, _ in trades:
                trade_type = "buy" if buyer_id == 1 else "sell"
                self.fixed_portfolio.update(trade_price, trade_qty, trade_type, current_price)

        # Process AI agent action
        order_type_idx, price_offset, qty = action
        order_type_idx = int(order_type_idx)
        qty = int(qty)

        if order_type_idx < 2 and qty > 0:
            order_type = "buy" if order_type_idx == 0 else "sell"
            price = current_price * (1 + price_offset)
            trades = self.order_book.place_order(0, order_type, price, qty)
            for trade_price, trade_qty, buyer_id, _ in trades:
                trade_type = "buy" if buyer_id == 0 else "sell"
                self.ai_portfolio.update(trade_price, trade_qty, trade_type, current_price)

        # Compute AI reward
        ai_prev_pnl = self.ai_portfolio.pnl
        ai_reward = self.ai_portfolio.pnl - ai_prev_pnl

        self.current_tick += 1
        done = self.current_tick >= self.num_ticks
        obs = self._get_observation()
        info = {
            "ai_pnl": self.ai_portfolio.pnl,
            "fixed_pnl": self.fixed_portfolio.pnl,
            "current_price": current_price,
        }

        return obs, ai_reward, done, info

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        """

        current_price = self.market_sim.price
        bid_ask_spread = self.market_sim.bid_ask_spread
        prices = np.array(list(self.price_history)) if self.price_history else np.array([current_price])
        short_ma = np.mean(prices[-10:]) if len(prices) >= 10 else current_price
        long_ma = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
        position = self.ai_portfolio.position
        cash = self.ai_portfolio.cash
        pnl = self.ai_portfolio.pnl
        order_book_depth = len(self.order_book.buy_orders) + len(self.order_book.sell_orders)

        return np.array(
            [
                current_price,
                bid_ask_spread,
                short_ma,
                long_ma,
                position,
                cash,
                pnl,
                order_book_depth,
            ],
            dtype=np.float32,
        )

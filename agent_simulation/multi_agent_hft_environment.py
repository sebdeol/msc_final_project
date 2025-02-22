from collections import deque
from typing import List, Tuple

import gym
import numpy as np

from agent_simulation.market_simulator import MarketSimulator
from agent_simulation.order_book import OrderBook
from agent_simulation.portfolio import Portfolio


class MultiAgentHFTEnv(gym.Env):
    def __init__(self, market_data, num_ticks_per_month=1000) -> None:
        super(MultiAgentHFTEnv, self).__init__()
        self.market_data = market_data
        self.num_ticks = num_ticks_per_month
        self.market_sim = MarketSimulator(market_data, num_ticks_per_month)
        self.order_book = OrderBook()
        self.ai_portfolio1 = Portfolio()
        self.ai_portfolio2 = Portfolio()
        self.current_tick = 0
        self.price_history = deque(maxlen=50)

    def step(self, actions) -> Tuple:
        action1, action2 = actions
        current_price, _ = self.market_sim.step()
        self.price_history.append(current_price)

        # Process AI agent 1 action
        order_type1 = "no_action"
        order_type_idx1, price_offset1, qty1 = action1
        order_type_idx1 = int(order_type_idx1)
        qty1 = int(qty1)

        if order_type_idx1 < 2 and qty1 > 0:
            order_type1 = "buy" if order_type_idx1 == 0 else "sell"
            if order_type1 == "buy":
                price1 = current_price * (1 - abs(price_offset1))
            else:
                price1 = current_price * (1 + abs(price_offset1))

            trades1 = self.order_book.place_order(0, order_type1, price1, qty1)

            for trade_price, trade_qty, buyer_id, seller_id in trades1:
                trade_type = "buy" if buyer_id == 0 else "sell"
                self.ai_portfolio1.update(trade_price, trade_qty, trade_type, current_price)

        # Process AI agent 2 action
        order_type2 = "no_action"
        order_type_idx2, price_offset2, qty2 = action2
        order_type_idx2 = int(order_type_idx2)
        qty2 = int(qty2)

        if order_type_idx2 < 2 and qty2 > 0:
            order_type2 = "buy" if order_type_idx2 == 0 else "sell"

            if order_type2 == "buy":
                price2 = current_price * (1 - abs(price_offset2))
            else:
                price2 = current_price * (1 + abs(price_offset2))

            trades2 = self.order_book.place_order(1, order_type2, price2, qty2)
            for trade_price, trade_qty, buyer_id, seller_id in trades2:
                trade_type = "buy" if buyer_id == 1 else "sell"
                self.ai_portfolio2.update(trade_price, trade_qty, trade_type, current_price)

        self.current_tick += 1
        done = self.current_tick >= self.num_ticks
        obs1 = self._get_observation(1)
        obs2 = self._get_observation(2)
        info = {
            "ai_pnl1": self.ai_portfolio1.pnl,
            "ai_pnl2": self.ai_portfolio2.pnl,
            "current_price": current_price,
            "action1": order_type1,
            "action2": order_type2,
        }

        return [obs1, obs2], done, info

    def reset(self) -> List:
        self.current_tick = 0
        self.price_history.clear()
        self.ai_portfolio1 = Portfolio()
        self.ai_portfolio2 = Portfolio()
        self.order_book = OrderBook()
        self.market_sim.reset(0)
        obs1 = self._get_observation(1)
        obs2 = self._get_observation(2)
        return [obs1, obs2]

    def _get_observation(self, agent_id) -> np.ndarray:
        current_price = self.market_sim.price
        bid_ask_spread = self.market_sim.bid_ask_spread
        prices = np.array(list(self.price_history)) if self.price_history else np.array([current_price])
        short_ma = np.mean(prices[-10:]) if len(prices) >= 10 else current_price
        long_ma = np.mean(prices[-50:]) if len(prices) >= 50 else current_price

        if agent_id == 1:
            position = self.ai_portfolio1.position
            cash = self.ai_portfolio1.cash
            pnl = self.ai_portfolio1.pnl
        else:  # agent_id == 2
            position = self.ai_portfolio2.position
            cash = self.ai_portfolio2.cash
            pnl = self.ai_portfolio2.pnl
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

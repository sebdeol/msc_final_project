from typing import Dict


class Portfolio:
    def __init__(self, initial_cash=100000, max_position=1000) -> None:
        """
        Initialise the portfolio class.
        - initial_cash: Starting cash balance
        - max_position: Maximum allowed position (absolute value)
        """

        self.cash = initial_cash
        self.position = 0
        self.max_position = max_position
        self.pnl = 0
        self.trades = []

    def update(self, trade_price: float, trade_qty: int, trade_type: str, current_price: float) -> bool:
        """
        Update portfolio after a trade.
        - trade_type: 'buy' or 'sell'
        """

        if trade_type == "buy":
            cost = trade_price * trade_qty
            if self.cash < cost or abs(self.position + trade_qty) > self.max_position:
                return False
            self.cash -= cost
            self.position += trade_qty
        else:
            proceeds = trade_price * trade_qty
            if self.position < -trade_qty or abs(self.position - trade_qty) > self.max_position:
                return False
            self.cash += proceeds
            self.position -= trade_qty

        self.trades.append((trade_price, trade_qty, trade_type, current_price))
        self.pnl = self.cash + self.position * current_price - 100000

        return True

    def get_state(self, current_price: float) -> Dict:
        """
        Get portfolio state for observation.
        """

        return {
            "cash": self.cash,
            "position": self.position,
            "pnl": self.pnl,
            "value": self.cash + self.position * current_price,
        }

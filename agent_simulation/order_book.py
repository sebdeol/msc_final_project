from typing import List


class OrderBook:
    def __init__(self) -> None:
        """
        Initialise the order book class.
        """

        self.buy_orders = []
        self.sell_orders = []
        self.order_id = 0

    def place_order(self, agent_id: int, order_type: str, price: float, quantity: int) -> List:
        """
        Place a limit order.
        - order_type: 'buy' or 'sell'
        - price: Limit price
        - quantity: Number of shares
        """

        timestamp = self.order_id
        self.order_id += 1
        order = (price, quantity, timestamp, agent_id)

        if order_type == "buy":
            self.buy_orders.append(order)
            self.buy_orders.sort(reverse=True)
        else:
            self.sell_orders.append(order)
            self.sell_orders.sort()

        return self.match_orders()

    def cancel_order(self, agent_id: int, order_type: str, price: float, quantity: int, timestamp: int) -> None:
        """
        Cancel an order.
        """

        orders = self.buy_orders if order_type == "buy" else self.sell_orders
        order = (price, quantity, timestamp, agent_id)

        if order in orders:
            orders.remove(order)

    def match_orders(self) -> List:
        """
        Match buy and sell orders.
        Returns list of trades: [(price, quantity, buyer_id, seller_id)]
        """

        trades = []

        while self.buy_orders and self.sell_orders:
            buy_order = self.buy_orders[0]
            sell_order = self.sell_orders[0]
            buy_price, buy_qty, buy_time, buy_agent = buy_order
            sell_price, sell_qty, sell_time, sell_agent = sell_order

            if buy_price >= sell_price:
                trade_qty = min(buy_qty, sell_qty)
                trade_price = sell_price
                trades.append((trade_price, trade_qty, buy_agent, sell_agent))

                # Update orders
                self.buy_orders[0] = (
                    buy_price,
                    buy_qty - trade_qty,
                    buy_time,
                    buy_agent,
                )
                self.sell_orders[0] = (
                    sell_price,
                    sell_qty - trade_qty,
                    sell_time,
                    sell_agent,
                )

                # Remove filled orders
                if self.buy_orders[0][1] == 0:
                    self.buy_orders.pop(0)

                if self.sell_orders[0][1] == 0:
                    self.sell_orders.pop(0)
            else:
                break

        return trades

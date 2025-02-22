from typing import Generator

import pytest

from agent_simulation.order_book import OrderBook


@pytest.fixture
def order_book() -> Generator:
    """Create a test order book instance."""

    yield OrderBook()


def test_order_book_initialisation() -> None:
    """Test order book initialisation."""

    ob = OrderBook()
    assert len(ob.buy_orders) == 0
    assert len(ob.sell_orders) == 0
    assert ob.order_id == 0


def test_place_buy_order(order_book) -> None:
    """Test placing a buy order."""

    trades = order_book.place_order(agent_id=1, order_type="buy", price=100.0, quantity=10)
    assert len(order_book.buy_orders) == 1
    assert len(trades) == 0
    assert order_book.buy_orders[0][0] == 100.0  # price
    assert order_book.buy_orders[0][1] == 10  # quantity
    assert order_book.buy_orders[0][3] == 1  # agent_id


def test_place_sell_order(order_book) -> None:
    """Test placing a sell order."""

    trades = order_book.place_order(agent_id=2, order_type="sell", price=110.0, quantity=5)
    assert len(order_book.sell_orders) == 1
    assert len(trades) == 0
    assert order_book.sell_orders[0][0] == 110.0  # price
    assert order_book.sell_orders[0][1] == 5  # quantity
    assert order_book.sell_orders[0][3] == 2  # agent_id


def test_order_matching(order_book) -> None:
    """Test matching of buy and sell orders."""

    # Place a buy order
    order_book.place_order(agent_id=1, order_type="buy", price=100.0, quantity=10)
    # Place a matching sell order
    trades = order_book.place_order(agent_id=2, order_type="sell", price=95.0, quantity=5)

    assert len(trades) == 1
    assert trades[0][0] == 95.0  # trade price
    assert trades[0][1] == 5  # trade quantity
    assert trades[0][2] == 1  # buyer id
    assert trades[0][3] == 2  # seller id
    assert len(order_book.buy_orders) == 1
    assert order_book.buy_orders[0][1] == 5  # remaining quantity


def test_order_sorting(order_book) -> None:
    """Test that orders are properly sorted."""

    # Add buy orders
    order_book.place_order(agent_id=1, order_type="buy", price=100.0, quantity=10)
    order_book.place_order(agent_id=2, order_type="buy", price=102.0, quantity=5)
    order_book.place_order(agent_id=3, order_type="buy", price=101.0, quantity=7)

    # Check buy orders are sorted by price (highest first)
    assert order_book.buy_orders[0][0] == 102.0
    assert order_book.buy_orders[1][0] == 101.0
    assert order_book.buy_orders[2][0] == 100.0

    # Add sell orders
    order_book.place_order(agent_id=4, order_type="sell", price=105.0, quantity=3)
    order_book.place_order(agent_id=5, order_type="sell", price=103.0, quantity=4)
    order_book.place_order(agent_id=6, order_type="sell", price=104.0, quantity=6)

    # Check sell orders are sorted by price (lowest first)
    assert order_book.sell_orders[0][0] == 103.0
    assert order_book.sell_orders[1][0] == 104.0
    assert order_book.sell_orders[2][0] == 105.0


def test_cancel_order(order_book) -> None:
    """Test order cancellation."""

    # Place orders
    order_book.place_order(agent_id=1, order_type="buy", price=100.0, quantity=10)
    timestamp = 0  # First order gets timestamp 0

    # Cancel the order
    order_book.cancel_order(agent_id=1, order_type="buy", price=100.0, quantity=10, timestamp=timestamp)
    assert len(order_book.buy_orders) == 0


def test_multiple_matches(order_book) -> None:
    """Test multiple order matches."""

    # Place buy orders
    order_book.place_order(agent_id=1, order_type="buy", price=100.0, quantity=10)
    order_book.place_order(agent_id=2, order_type="buy", price=101.0, quantity=5)

    # Place a large sell order that matches both
    trades = order_book.place_order(agent_id=3, order_type="sell", price=99.0, quantity=15)

    assert len(trades) == 2
    assert trades[0][2] == 2  # First trade with highest buy price
    assert trades[1][2] == 1  # Second trade with lower buy price
    assert sum(trade[1] for trade in trades) == 15  # Total quantity traded
    assert len(order_book.buy_orders) == 0


def test_partial_match(order_book) -> None:
    """Test partial order matching."""

    # Place a large buy order
    order_book.place_order(agent_id=1, order_type="buy", price=100.0, quantity=20)

    # Place a smaller sell order
    trades = order_book.place_order(agent_id=2, order_type="sell", price=98.0, quantity=5)

    assert len(trades) == 1
    assert trades[0][1] == 5  # Traded quantity
    assert order_book.buy_orders[0][1] == 15  # Remaining quantity

from typing import Generator

import pytest

from agent_simulation.portfolio import Portfolio


@pytest.fixture
def portfolio() -> Generator:
    """Create a test portfolio instance."""

    yield Portfolio(initial_cash=100000, max_position=1000)


def test_portfolio_initialisation() -> None:
    """Test portfolio initialisation."""

    portfolio = Portfolio(initial_cash=50000, max_position=500)
    assert portfolio.cash == 50000
    assert portfolio.position == 0
    assert portfolio.max_position == 500
    assert portfolio.pnl == 0
    assert len(portfolio.trades) == 0


def test_successful_buy(portfolio) -> None:
    """Test successful buy trade."""

    result = portfolio.update(trade_price=100, trade_qty=10, trade_type="buy", current_price=100)
    assert result is True
    assert portfolio.cash == 99000  # 100000 - (100 * 10)
    assert portfolio.position == 10
    assert len(portfolio.trades) == 1
    assert portfolio.pnl == 0  # (99000 + 10 * 100) - 100000


def test_successful_sell(portfolio) -> None:
    """Test successful sell trade."""

    # First buy some shares
    portfolio.update(trade_price=100, trade_qty=10, trade_type="buy", current_price=100)
    # Then sell them
    result = portfolio.update(trade_price=110, trade_qty=5, trade_type="sell", current_price=110)
    assert result is True
    assert portfolio.cash == 99550  # 99000 + (110 * 5)
    assert portfolio.position == 5
    assert len(portfolio.trades) == 2


def test_insufficient_cash(portfolio) -> None:
    """Test buy with insufficient cash."""

    result = portfolio.update(trade_price=100000, trade_qty=2, trade_type="buy", current_price=100000)
    assert result is False
    assert portfolio.cash == 100000  # Unchanged
    assert portfolio.position == 0
    assert len(portfolio.trades) == 0


def test_position_limit(portfolio) -> None:
    """Test position limit enforcement."""

    # Try to exceed max position
    result = portfolio.update(trade_price=10, trade_qty=1100, trade_type="buy", current_price=10)
    assert result is False
    assert portfolio.position == 0

    # Try with short position
    result = portfolio.update(trade_price=10, trade_qty=1100, trade_type="sell", current_price=10)
    assert result is False
    assert portfolio.position == 0


def test_get_state(portfolio) -> None:
    """Test get_state method."""

    portfolio.update(trade_price=100, trade_qty=10, trade_type="buy", current_price=110)
    state = portfolio.get_state(current_price=110)

    assert isinstance(state, dict)
    assert "cash" in state
    assert "position" in state
    assert "pnl" in state
    assert "value" in state
    assert state["cash"] == 99000
    assert state["position"] == 10
    assert state["value"] == 100100  # 99000 + (10 * 110)
    assert state["pnl"] == 100  # 100100 - 100000


def test_multiple_trades(portfolio) -> None:
    """Test multiple trades sequence."""

    # Buy 10 shares at 100
    portfolio.update(trade_price=100, trade_qty=10, trade_type="buy", current_price=100)
    # Sell 5 shares at 110
    portfolio.update(trade_price=110, trade_qty=5, trade_type="sell", current_price=110)
    # Buy 3 shares at 105
    portfolio.update(trade_price=105, trade_qty=3, trade_type="buy", current_price=105)

    assert portfolio.position == 8  # 10 - 5 + 3
    assert portfolio.cash == 99235  # 100000 - (100*10) + (110*5) - (105*3)
    assert len(portfolio.trades) == 3

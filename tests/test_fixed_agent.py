import numpy as np
import pytest

from agent_simulation.fixed_agent import FixedAgent


@pytest.fixture
def agent() -> FixedAgent:
    """Create a test fixed agent instance."""

    return FixedAgent(agent_id=1, short_window=5, long_window=10, threshold=0.01)


def test_fixed_agent_initialisation() -> None:
    """Test fixed agent initialisation."""

    agent = FixedAgent(agent_id=1, short_window=5, long_window=10, threshold=0.02)

    assert agent.agent_id == 1
    assert agent.short_window == 5
    assert agent.long_window == 10
    assert agent.threshold == 0.02
    assert len(agent.price_history) == 0


def test_price_history_update(agent) -> None:
    """Test price history updates."""

    prices = [100.0, 101.0, 99.0, 100.5]
    for price in prices:
        agent.update(price)

    assert len(agent.price_history) == 4
    assert list(agent.price_history) == prices


def test_price_history_max_length(agent) -> None:
    """Test price history maximum length."""

    # Add more prices than the long window
    prices = list(range(15))  # 15 > long_window (10)
    for price in prices:
        agent.update(price)

    assert len(agent.price_history) == 10  # Should be capped at long_window
    assert list(agent.price_history) == prices[-10:]  # Should keep most recent prices


def test_no_action_insufficient_history(agent) -> None:
    """Test that no action is taken with insufficient history."""

    # Add fewer prices than required
    for price in [100.0, 101.0, 99.0]:
        agent.update(price)

    action = agent.get_action(current_price=100.0)
    assert action is None


def test_buy_signal(agent) -> None:
    """Test buy signal generation."""

    # Create a scenario where price drops below MA
    prices = [100.0] * 8 + [101.0, 102.0]  # Short MA will be higher
    for price in prices:
        agent.update(price)

    # Price significantly below short MA should trigger buy
    action = agent.get_action(current_price=98.0)  # About 3% below MA

    assert action is not None
    assert action[0] == "buy"
    assert action[1] == 98.0
    assert action[2] == 10


def test_sell_signal(agent) -> None:
    """Test sell signal generation."""

    # Create a scenario where price rises above MA
    prices = [100.0] * 8 + [99.0, 98.0]  # Short MA will be lower
    for price in prices:
        agent.update(price)

    # Price significantly above short MA should trigger sell
    action = agent.get_action(current_price=102.0)  # About 3% above MA

    assert action is not None
    assert action[0] == "sell"
    assert action[1] == 102.0
    assert action[2] == 10


def test_no_action_within_threshold(agent) -> None:
    """Test no action when price is within threshold."""
    # Create stable price history
    prices = [100.0] * 10
    for price in prices:
        agent.update(price)

    # Price within threshold should not trigger action
    action = agent.get_action(current_price=100.5)  # Only 0.5% above MA
    assert action is None


def test_threshold_boundary_conditions(agent) -> None:
    """Test behavior at threshold boundaries."""

    # Set up price history
    prices = [100.0] * 10
    for price in prices:
        agent.update(price)

    # Test exactly at threshold (should not trigger)
    action = agent.get_action(current_price=101.0)  # Exactly 1% above MA
    assert action is None

    # Test just above threshold (should trigger sell)
    action = agent.get_action(current_price=101.1)  # Just over 1% above MA
    assert action is not None
    assert action[0] == "sell"

    # Test just below threshold (should trigger buy)
    action = agent.get_action(current_price=98.9)  # Just over 1% below MA
    assert action is not None
    assert action[0] == "buy"

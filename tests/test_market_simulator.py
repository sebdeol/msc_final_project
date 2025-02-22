from typing import Generator

import numpy as np
import pandas as pd
import pytest

from agent_simulation.market_simulator import MarketSimulator


@pytest.fixture
def sample_data() -> Generator:
    """Create sample market data for testing."""

    data = {
        "date": pd.date_range(start="2020-01-01", end="2020-12-31", freq="ME"),
        "intraday_volatility": [1.5, 2.0, 1.8, 1.6, 1.7, 1.9, 2.1, 1.8, 1.7, 1.6, 1.5, 1.4],
        "bid_ask_spread": [10, 12, 11, 9, 10, 11, 13, 12, 10, 9, 8, 9],
        "flash_crash_events_count": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    }

    yield pd.DataFrame(data)


@pytest.fixture
def simulator(sample_data) -> Generator:
    """Create a test market simulator instance."""

    yield MarketSimulator(monthly_data=sample_data, num_ticks_per_month=100, dt=1 / 252)


@pytest.fixture
def long_simulator(sample_data) -> Generator:
    """Create a test market simulator instance with more ticks for volatility testing."""

    yield MarketSimulator(monthly_data=sample_data, num_ticks_per_month=1000, dt=1 / 252)


def test_market_simulator_initialisation(simulator, sample_data) -> None:
    """Test market simulator initialisation."""

    assert simulator.num_ticks == 100
    assert simulator.dt == 1 / 252
    assert simulator.current_month_idx == 0
    assert simulator.current_tick == 0
    assert simulator.price == 100.0
    assert len(simulator.prices) == 0
    assert simulator.monthly_data.equals(sample_data)


def test_reset(simulator) -> None:
    """Test reset functionality."""

    simulator.reset(month_idx=1)

    assert simulator.current_month_idx == 1
    assert simulator.current_tick == 0
    assert simulator.price == 100.0
    assert len(simulator.prices) == 1
    assert simulator.prices[0] == 100.0
    assert simulator.volatility == 0.02  # 2.0/100
    assert simulator.bid_ask_spread == 0.0012  # 12/10000
    assert simulator.flash_crash_prob == 0.01  # 1/100


def test_step_basic_functionality(simulator) -> None:
    """Test basic step functionality."""

    simulator.reset(month_idx=0)
    price, spread = simulator.step()

    assert isinstance(price, float)
    assert isinstance(spread, float)
    assert simulator.current_tick == 1
    assert len(simulator.prices) == 2  # Initial price + new price
    assert spread == 0.001  # 10/10000


def test_step_price_evolution(simulator) -> None:
    """Test price evolution over multiple steps."""

    simulator.reset(month_idx=0)
    prices = [simulator.price]

    for _ in range(10):
        price, _ = simulator.step()
        prices.append(price)

    # Check price movement
    price_changes = np.diff(np.log(prices))
    assert len(price_changes) == 10
    assert np.std(price_changes) > 0  # Prices should move


def test_flash_crash_probability(simulator) -> None:
    """Test flash crash probability."""

    np.random.seed(42)  # For reproducibility
    simulator.reset(month_idx=1)  # Month with flash crash event

    # Run many steps to test flash crash probability
    crash_count = 0
    n_trials = 1000

    for _ in range(n_trials):
        simulator.reset(month_idx=1)
        initial_price = simulator.price
        price, _ = simulator.step()
        if price < initial_price * 0.95:  # Significant price drop
            crash_count += 1

    # Check if crash frequency is roughly as expected
    expected_crashes = n_trials * (1 / 100)  # 1 crash per 100 ticks
    assert abs(crash_count - expected_crashes) < expected_crashes * 0.5  # Within 50% of expected


def test_end_of_month(simulator) -> None:
    """Test behavior at end of month."""

    simulator.reset(month_idx=0)

    # Run all ticks for the month
    for _ in range(simulator.num_ticks):
        simulator.step()

    # Next step should raise error
    with pytest.raises(ValueError, match="End of ticks for this month."):
        simulator.step()


def test_volatility_scaling(long_simulator) -> None:
    """Test volatility scaling and price distribution."""

    np.random.seed(42)  # For reproducibility
    long_simulator.reset(month_idx=0)
    initial_price = long_simulator.price
    prices = [initial_price]

    # Use all available ticks for better statistical significance
    for _ in range(long_simulator.num_ticks):
        price, _ = long_simulator.step()
        prices.append(price)

    # Calculate realised volatility (no need to annualise as it's already annualised in our test data)
    returns = np.diff(np.log(prices))
    realised_vol = np.std(returns) / np.sqrt(long_simulator.dt)
    target_vol = long_simulator.volatility * np.sqrt(252)

    # Check if realised volatility is roughly as expected (within 20% of target)
    assert abs(realised_vol - target_vol) < target_vol * 0.2

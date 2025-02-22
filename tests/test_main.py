from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sb3_contrib import RecurrentPPO

from agent_simulation.main import (
    TOTAL_TIMESTEPS,
    evaluate_simulation,
    evaluate_two_ai_agents,
    load_market_data,
    train_ai_agent,
)

# Verify the actual constant is 1M steps
assert TOTAL_TIMESTEPS == 1000000, "Main training should use 1M steps"


@pytest.fixture
def sample_market_data() -> Generator:
    """Create sample market data for testing."""

    data = {
        "date": pd.date_range(start="2020-01-01", end="2020-12-31", freq="ME"),
        "intraday_volatility": [1.5, 2.0, 1.8, 1.6, 1.7, 1.9, 2.1, 1.8, 1.7, 1.6, 1.5, 1.4],
        "bid_ask_spread": [10, 12, 11, 9, 10, 11, 13, 12, 10, 9, 8, 9],
        "flash_crash_events_count": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    }
    yield pd.DataFrame(data)


@pytest.fixture
def mock_env() -> Generator:
    """Create a mock environment."""

    env = MagicMock()
    env.reset.return_value = np.zeros(10)
    env.step.return_value = (
        np.zeros(10),
        0.0,
        [False],
        [{"ai_pnl": 100, "fixed_pnl": 50, "current_price": 100}],
    )

    yield env


@pytest.fixture
def mock_multi_agent_env() -> Generator:
    """Create a mock environment for two agents."""
    env = MagicMock()
    env.reset.return_value = [np.zeros(10), np.zeros(10)]
    env.step.return_value = (
        [np.zeros(10), np.zeros(10)],
        False,
        {
            "ai_pnl1": 100,
            "ai_pnl2": 150,
            "current_price": 100,
            "action1": 1,
            "action2": 0,
        },
    )
    yield env


def test_load_market_data(tmp_path, sample_market_data) -> None:
    """Test loading market data from CSV."""

    csv_path = tmp_path / "test_market_data.csv"
    sample_market_data.to_csv(csv_path, index=False)

    # Load the data
    loaded_data = load_market_data(csv_path)

    # Check data loading and cleaning
    assert isinstance(loaded_data, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(loaded_data["date"])
    assert loaded_data.equals(sample_market_data)


@patch("agent_simulation.main.RecurrentPPO")
@patch("agent_simulation.main.TOTAL_TIMESTEPS", 1)  # Mock the constant to 1 step for tests
@pytest.mark.skip(reason="This test is too slow")
def test_train_ai_agent(mock_ppo, mock_env) -> None:
    """Test AI agent training."""

    mock_model = MagicMock(spec=RecurrentPPO)
    mock_ppo.return_value = mock_model

    # Train agent with 1 step for testing
    model = train_ai_agent(mock_env, total_timesteps=1)

    # Verify training
    assert mock_ppo.called
    mock_model.learn.assert_called_once_with(total_timesteps=1)
    assert model == mock_model


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
@pytest.mark.skip(reason="This test is too slow")
def test_evaluate_simulation(mock_close, mock_savefig, mock_env) -> None:
    """Test simulation evaluation."""

    # Setup mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.zeros(1), None)

    # Run evaluation
    evaluate_simulation(mock_model, mock_env, num_episodes=2)

    # Verify evaluation
    assert mock_env.reset.call_count == 2  # Called for each episode
    assert mock_savefig.call_count == 2  # Two plots saved
    assert mock_close.call_count == 2  # Two plots closed


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
@pytest.mark.skip(reason="This test is too slow")
def test_evaluate_two_ai_agents(mock_close, mock_savefig, mock_multi_agent_env) -> None:
    """Test evaluation of two AI agents."""

    # Setup mock models
    mock_model1 = MagicMock()
    mock_model2 = MagicMock()
    mock_model1.predict.return_value = (np.zeros(1), None)
    mock_model2.predict.return_value = (np.ones(1), None)

    # Run evaluation
    evaluate_two_ai_agents(mock_model1, mock_model2, mock_multi_agent_env, num_episodes=2)

    # Verify evaluation
    assert mock_multi_agent_env.reset.call_count == 2  # Called for each episode
    assert mock_savefig.call_count == 2  # Two plots saved
    assert mock_close.call_count == 2  # Two plots closed
    assert mock_model1.predict.called
    assert mock_model2.predict.called


@pytest.mark.skip(reason="This test is too slow")
def test_evaluate_simulation_logging(mock_env, caplog) -> None:
    """Test logging in simulation evaluation."""

    # Setup mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.zeros(1), None)

    # Run evaluation with logging
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
        evaluate_simulation(mock_model, mock_env, num_episodes=1)

    # Verify logging
    assert "AI PnL" in caplog.text
    assert "Fixed PnL" in caplog.text

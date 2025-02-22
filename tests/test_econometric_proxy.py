import io
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture
def mock_market_data() -> Generator:
    """Create a mock DataFrame that mimics market_data.csv."""

    data = {
        "date": [
            "2017-12",
            "2018-01",
            "2018-02",
            "2018-03",
            "2019-01",
            "2019-02",
            "2019-03",
            "2019-04",
        ],
        "bid_ask_spread": [4.8, 5.0, 8.0, 6.5, 6.5, 6.0, 5.8, 5.5],
        "intraday_volatility": [0.5, 0.6, 2.5, 1.8, 1.6, 1.2, 1.0, 0.8],
        "flash_crash_events_count": [0, 0, 1, 0, 0, 0, 0, 0],
    }
    yield pd.DataFrame(data)


@pytest.fixture
def mock_csv_content() -> Generator:
    yield """date,bid_ask_spread,intraday_volatility,flash_crash_events_count
            2017-12,4.8,0.5,0
            2018-01,5.0,0.6,0
            2018-02,8.0,2.5,1
            2018-03,6.5,1.8,0
            2019-01,6.5,1.6,0
            2019-02,6.0,1.2,0
            2019-03,5.8,1.0,0
            2019-04,5.5,0.8,0"""


def test_data_loading_and_cleaning(mock_csv_content) -> None:
    with patch("builtins.open", mock_open(read_data=mock_csv_content)):
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.read_csv(io.StringIO(mock_csv_content))

            from econometric_proxy_test import df

            assert pd.api.types.is_datetime64_any_dtype(df["date"])

            assert "Spread" in df.columns
            assert "Volatility" in df.columns
            assert "FlashCrashes" in df.columns


def test_synthetic_algo_msg_generation(mock_market_data) -> None:
    """Test the synthetic AlgoMsg variable generation"""

    df = mock_market_data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    n = len(df)
    # Generate synthetic AlgoMsg
    df["AlgoMsg"] = np.linspace(1000, 10000, n) + np.random.normal(0, 300, n)

    # Test the general characteristics of AlgoMsg
    assert len(df["AlgoMsg"]) == n
    assert df["AlgoMsg"].min() > 0  # Should be positive
    assert all(df["AlgoMsg"].diff().iloc[1:] > -1000)  # Allow for some noise but generally increasing


def test_post2018_dummy_creation(mock_market_data) -> None:
    """Test the creation of post-2018 dummy variable"""

    df = mock_market_data.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Create post2018 dummy
    df["post2018"] = (df["date"].dt.year >= 2018).astype(int)

    # Test dummy variable characteristics
    assert df[df["date"].dt.year < 2018]["post2018"].eq(0).all()
    assert df[df["date"].dt.year >= 2018]["post2018"].eq(1).all()


def test_regression_models(mock_market_data) -> None:
    """Test the structure and basic properties of regression models"""

    df = mock_market_data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.rename(
        columns={
            "bid_ask_spread": "Spread",
            "intraday_volatility": "Volatility",
            "flash_crash_events_count": "FlashCrashes",
        },
        inplace=True,
    )

    n = len(df)
    df["AlgoMsg"] = np.linspace(1000, 10000, n) + np.random.normal(0, 300, n)
    df["post2018"] = (df["date"].dt.year >= 2018).astype(int)
    df["interact"] = df["AlgoMsg"] * df["post2018"]

    # Test spread regression
    model_spread = smf.ols(formula="Spread ~ AlgoMsg + interact + Volatility", data=df).fit()

    # Check model properties
    assert hasattr(model_spread, "params")
    assert "AlgoMsg" in model_spread.params
    assert "interact" in model_spread.params
    assert "Volatility" in model_spread.params

    # Test volatility regression
    model_vol = smf.ols(formula="Volatility ~ AlgoMsg + interact", data=df).fit()

    assert hasattr(model_vol, "params")
    assert "AlgoMsg" in model_vol.params
    assert "interact" in model_vol.params


def test_date_filtering(mock_market_data) -> None:
    """Test the date filtering functionality"""

    df = mock_market_data.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Apply filtering
    start_date = "2018-01-01"
    end_date = "2019-03-01"
    filtered_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

    # Test filtering results
    assert filtered_df["date"].min() >= pd.Timestamp(start_date)
    assert filtered_df["date"].max() <= pd.Timestamp(end_date)
    assert len(filtered_df) < len(df)

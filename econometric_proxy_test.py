import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ------------------------------------------------------------------------
# The following code simulates the econometric proxy for market quality
#
# This analysis examines the relationship between algorithmic trading activity
# and market quality metrics (bid-ask spreads and volatility) over time.
#
# Key components:
# 1. Market quality metrics:
#    - Bid-ask spread: A measure of market liquidity
#    - Intraday volatility: A measure of price stability
#    - Flash crash events: Count of extreme price movements
#
# 2. Proxy construction:
#    - AlgoMsg: Synthetic proxy for algorithmic trading intensity
#    - post2018: Dummy variable for structural break analysis
#    - interact: Interaction term to capture post-2018 changes
#
# 3. Methodology:
#    - Two-stage regression approach:
#      a) First regression: Impact on bid-ask spreads
#      b) Second regression: Impact on volatility
#    - Controls for time period effects and market conditions
#
# 4. Hypotheses tested:
#    H1: Algorithmic trading affects market liquidity
#    H2: The relationship changed after 2018
#    H3: There's a link between algo trading and volatility
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# 1. Load the market data into a DataFrame and clean data
# ------------------------------------------------------------------------
file_path = "market_data.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

# Convert date strings to a proper datetime
df["date"] = pd.to_datetime(df["date"], format="%Y-%m")

# Rename columns
df.rename(
    columns={
        "bid_ask_spread": "Spread",
        "intraday_volatility": "Volatility",
        "flash_crash_events_count": "FlashCrashes",
    },
    inplace=True,
)

# ------------------------------------------------------------------------
# 2. Generate a synthetic "AlgoMsg" variable.
#    For demonstration, let's do something that trends upward over time.
#    Real work would replace this with actual message-traffic data.
# ------------------------------------------------------------------------
df = df.sort_values("date")

n = len(df)
# Synthetic: let's create a linear upward trend + random noise
# e.g. from ~ 1000 messages in 2005 to ~ 10000 in 2023
df["AlgoMsg"] = np.linspace(1000, 10000, n) + np.random.normal(0, 300, n)

# ------------------------------------------------------------------------
# 3. Create the post2018 dummy and interaction term
# ------------------------------------------------------------------------
df["post2018"] = (df["date"].dt.year >= 2018).astype(int)
df["interact"] = df["AlgoMsg"] * df["post2018"]

# ------------------------------------------------------------------------
# 4. Filter data from 2010-01 to 2022-12
# ------------------------------------------------------------------------
start_date = "2010-01-01"
end_date = "2022-12-01"
df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

# ------------------------------------------------------------------------
# 5. Run the OLS regression:
#    Spread_t = β0 + β1 AlgoMsg_t + β2 (AlgoMsg_t × post2018) + β3 Volatility_t
# ------------------------------------------------------------------------
model_spread = smf.ols(
    formula="Spread ~ AlgoMsg + interact + Volatility",
    data=df,
).fit()

print("=== Regression Results: Spread as Dependent Variable (2010-2022) ===")
print(model_spread.summary())

# ------------------------------------------------------------------------
# 6. Run the OLS regression with Volatility as the dependent variable
#    e.g., Volatility_t ~ γ0 + γ1 AlgoMsg_t + γ2 (AlgoMsg_t × post2018)
# ------------------------------------------------------------------------
model_vol = smf.ols(formula="Volatility ~ AlgoMsg + interact", data=df).fit()

print("\n=== Regression Results: Volatility as Dependent Variable (2010-2022) ===")
print(model_vol.summary())

# ------------------------------------------------------------------------
# 7. Interpretation of results:
#    - β1 (coefficient on AlgoMsg) = baseline relationship pre-2018
#    - β2 (coefficient on interact) = change in that relationship post-2018
#    - If β2 is small / not significant => no big shift after 2018
#    - Sign & significance of β3 = correlation of volatility with spread
#    - Similarly for the volatility regression
#
# ------------------------------------------------------------------------

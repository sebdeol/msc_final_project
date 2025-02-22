# MSc Computer Science Dissertation Project

This repository contains the code and analysis supporting my MSc MSc Computer Science dissertation at the University of Essex Online. The research investigates the impact of algorithmic trading on market quality through both empirical analysis and agent-based simulation.

## Research Overview

The project combines two complementary approaches to study algorithmic trading's market impact:

1. **Empirical Analysis**: Using historical market data to examine how algorithmic trading activity affects market quality metrics, with a particular focus on structural changes following 2018.

2. **Agent-Based Simulation**: Implementing a multi-agent market simulation to study the micro-level interactions between different trading strategies and their emergent effects on market quality.

The dual approach allows us to both validate empirical findings through simulation and explore market dynamics that might be difficult to isolate in historical data.

## Project Components

The project consists of two main components:
1. Econometric Analysis: Studies historical market quality metrics
2. Agent-Based Simulation: Simulates market dynamics with different types of trading agents

### Econometric Analysis

#### Market Quality Econometric Analysis

This project implements an econometric analysis to study the relationship between algorithmic trading activity and market quality metrics. It uses a proxy-based approach to examine how algorithmic trading affects market liquidity and volatility, with a particular focus on structural changes after 2018.

The analysis examines three key hypotheses:
1. The impact of algorithmic trading on market liquidity (measured by bid-ask spreads)
2. Whether this relationship experienced a structural break after 2018
3. The connection between algorithmic trading activity and market volatility

### Agent-Based Simulation
The simulation framework includes:
- Market simulator with realistic price dynamics and flash crashes
- Order book implementation with price-time priority
- Multiple agent types:
  - Fixed-rule agents using mean reversion strategies
  - AI agents trained using reinforcement learning
  - Portfolio management with position and risk limits

## Components

### Market Quality Metrics
- Bid-ask spread (measure of market liquidity)
- Intraday volatility (measure of price stability)
- Flash crash events (count of extreme price movements)

### Econometric Models

The analysis employs two main regression models to test our hypotheses:

1. **Spread Regression**:
   ```math
   Spread_t = β₀ + β₁ AlgoMsg_t + β₂ (AlgoMsg_t × post2018) + β₃ Volatility_t
   ```
   where:
   - β₀: Baseline spread level
   - β₁: Impact of algorithmic trading on spreads pre-2018
   - β₂: Change in algorithmic trading impact post-2018
   - β₃: Control for volatility effects

2. **Volatility Regression**:
   ```math
   Volatility_t = γ₀ + γ₁ AlgoMsg_t + γ₂ (AlgoMsg_t × post2018)
   ```
   where:
   - γ₀: Baseline volatility level
   - γ₁: Impact of algorithmic trading on volatility pre-2018
   - γ₂: Change in algorithmic trading impact post-2018

### Agent Simulation Components
- **OrderBook**: Implements a limit order book with:
  - Price-time priority matching
  - Buy/sell order queues
  - Order cancellation
  - Partial fills

- **Portfolio**: Manages trading positions with:
  - Cash and position tracking
  - Position limits
  - P&L calculation
  - Risk management

- **MarketSimulator**: Simulates market dynamics:
  - Geometric Brownian Motion for price evolution
  - Configurable volatility
  - Flash crash events
  - Bid-ask spread dynamics

- **FixedAgent**: Rule-based trading agent:
  - Mean reversion strategy
  - Moving average signals
  - Configurable thresholds
  - Position sizing

## Requirements

- Python 3.9+ (tested on Python 3.9, however, it should be compatible with newer versions)
- Required packages are listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone git@github.com:sebdeol/msc_final_project.git
cd msc_final_project
```

2. Create and activate a virtual environment:
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# OR
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

## Usage

### Running the Econometric Analysis

To run the main analysis:
```bash
python3 econometric_proxy_test.py
```

### Running the Agent Simulation

To run the agent simulation:
```bash
python3 agent_simulation/main.py
```

### Running Tests

The project includes comprehensive test suites for both components:

```bash
python3 -m pytest
```

## Data Format

### Econometric Analysis Data
The analysis expects a CSV file (`market_data.csv`).
The data isn't included in the repository but is available upon request (sp23223 [ @ ] essex.ac.uk)
- `date`: Month in YYYY-MM format
- `bid_ask_spread`: Average bid-ask spread
- `intraday_volatility`: Intraday price volatility measure
- `flash_crash_events_count`: Number of flash crash events

### Agent Simulation Configuration
The simulation can be configured with:
- Number of agents
- Trading parameters
- Market conditions
- Simulation length

## Project Structure

```
msc_final_project/
├── README.md
├── requirements.txt
├── econometric_proxy_test.py    # Main econometric analysis
├── market_data.csv              # Input data
├── agent_simulation/
│   ├── main.py                  # Simulation entry point
│   ├── order_book.py            # Order book implementation
│   ├── portfolio.py             # Portfolio management
│   ├── market_simulator.py      # Price/market simulation
│   ├── fixed_agent.py           # Rule-based agent
│   └── charts/                  # Simulation output charts
└── tests/
    ├── __init__.py
    ├── test_econometric_proxy.py
    ├── test_market_simulator.py
    ├── test_order_book.py
    ├── test_portfolio.py
    └── test_fixed_agent.py
```

## Results

### Econometric Analysis
The regression results provide insights into:
- The baseline relationship between algorithmic trading and market quality pre-2018
- Changes in this relationship post-2018
- The correlation between volatility and spreads
- Statistical significance of the relationships

### Agent Simulation
The simulation generates:
- Price and return distributions
- Order book dynamics
- Agent performance metrics
- Market quality indicators

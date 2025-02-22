import logging

import matplotlib.pyplot as plt
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agent_simulation.ai_agent_vs_fixed_rule_agent import AIAgentVsFixedRuleAgent
from agent_simulation.multi_agent_hft_environment import MultiAgentHFTEnv

TOTAL_TIMESTEPS: int = 1000000


def load_market_data(csv_file) -> pd.DataFrame:
    """
    Load monthly market data from a CSV file, clean the dates and return a DataFrame.
    """

    df = pd.read_csv(csv_file)
    df["date"] = pd.to_datetime(df["date"])
    return df


# Load the data
data = load_market_data("market_data.csv")


def train_ai_agent(env, total_timesteps=100000) -> RecurrentPPO:
    """
    Train the AI agent using RecurrentPPO with an LSTM policy.

    Args:
        env: The Gym environment.
        total_timesteps: Number of timesteps to train for (default: 100000).
    Returns:
        The trained RecurrentPPO model.
    """

    model = RecurrentPPO(
        "MlpLstmPolicy",  # Policy with LSTM for recurrent processing
        env,
        verbose=1,  # Print training progress
        policy_kwargs={  # Customise the LSTM and network architecture
            "lstm_hidden_size": 64,  # Size of the LSTM hidden state
            "n_lstm_layers": 1,  # Number of LSTM layers
            "net_arch": [64, 64],  # Architecture of the MLP after LSTM
        },
    )
    model.learn(total_timesteps=total_timesteps)

    return model


def evaluate_simulation(model, env, num_episodes=1) -> None:
    """
    Evaluate the simulation and log results.
    """

    logging.basicConfig(filename="simulation.log", level=logging.INFO)
    ai_pnls = []
    fixed_pnls = []
    prices = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_ai_pnl = []
        episode_fixed_pnl = []
        episode_prices = []

        while not done:
            action, _ = model.predict(obs)
            obs, _, dones, infos = env.step(action)
            done = dones[0]
            info = infos[0]
            episode_ai_pnl.append(info["ai_pnl"])
            episode_fixed_pnl.append(info["fixed_pnl"])
            episode_prices.append(info["current_price"])

        ai_pnls.append(episode_ai_pnl)
        fixed_pnls.append(episode_fixed_pnl)
        prices.append(episode_prices)

        logging.info(f"Episode {episode + 1}: AI PnL = {episode_ai_pnl[-1]}, Fixed PnL = {episode_fixed_pnl[-1]}")

    # Plot results
    plt.figure(figsize=(12, 6))

    for i in range(num_episodes):
        plt.plot(ai_pnls[i], label=f"AI PnL Episode {i + 1}")
        plt.plot(fixed_pnls[i], label=f"Fixed PnL Episode {i + 1}", linestyle="--")
    plt.title("AI vs Fixed Agent PnL")
    plt.xlabel("Time Step")
    plt.ylabel("PnL")
    plt.legend()
    plt.savefig("agent_simulation/charts/pnl_comparison_single_agent_vs_fixed_rule_agent.png")
    plt.close()

    plt.figure(figsize=(12, 6))

    for i in range(num_episodes):
        plt.plot(prices[i], label=f"Price Episode {i + 1}")
    plt.title("Price Trajectories")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("agent_simulation/charts/price_trajectories_single_agent_vs_fixed_rule_agent.png")
    plt.close()


def evaluate_two_ai_agents(model1, model2, env, num_episodes=1) -> None:
    ai_pnls1 = []
    ai_pnls2 = []
    prices = []
    actions1 = []
    actions2 = []

    for _ in range(num_episodes):
        obs_list = env.reset()
        done = False
        episode_ai_pnl1 = []
        episode_ai_pnl2 = []
        episode_prices = []
        episode_actions1 = []
        episode_actions2 = []

        while not done:
            action1, _ = model1.predict(obs_list[0])
            action2, _ = model2.predict(obs_list[1])
            obs_list, done, info = env.step([action1, action2])

            episode_ai_pnl1.append(info["ai_pnl1"])
            episode_ai_pnl2.append(info["ai_pnl2"])
            episode_prices.append(info["current_price"])
            episode_actions1.append(info["action1"])
            episode_actions2.append(info["action2"])

        ai_pnls1.append(episode_ai_pnl1)
        ai_pnls2.append(episode_ai_pnl2)
        prices.append(episode_prices)
        actions1.append(episode_actions1)
        actions2.append(episode_actions2)

    # Compare actions
    for ep in range(num_episodes):
        for t in range(len(actions1[ep])):
            if actions1[ep][t] == actions2[ep][t]:
                print(f"Episode {ep + 1}, Step {t}: Both agents took the same action: {actions1[ep][t]}")
            else:
                print(f"Episode {ep + 1}, Step {t}: Agent 1: {actions1[ep][t]}, Agent 2: {actions2[ep][t]}")

    # Plot PnL
    plt.figure(figsize=(12, 6))
    for i in range(num_episodes):
        plt.plot(ai_pnls1[i], label=f"AI1 PnL Episode {i + 1}")
        plt.plot(ai_pnls2[i], label=f"AI2 PnL Episode {i + 1}")

    plt.title("AI1 vs AI2 Agent PnL")
    plt.xlabel("Time Step")
    plt.ylabel("PnL")
    plt.legend()
    plt.savefig("agent_simulation/charts/pnl_comparison_two_ai_agents.png")
    plt.close()

    # Plot price trajectory
    plt.figure(figsize=(12, 6))
    for i in range(num_episodes):
        plt.plot(prices[i], label=f"Price Episode {i + 1}")
    plt.title("Price Trajectories")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.savefig("agent_simulation/charts/price_trajectories_two_ai_agents.png")
    plt.close()


if __name__ == "__main__":
    # Evaluate the simulation with an AI agent vs a fixed rule agent
    env = DummyVecEnv([lambda: AIAgentVsFixedRuleAgent(data)])
    model = train_ai_agent(env, total_timesteps=TOTAL_TIMESTEPS)
    evaluate_simulation(model, env)

    # Evaluate the simulation with two AI agents
    # Set up training environments
    training_environment_one = DummyVecEnv([lambda: AIAgentVsFixedRuleAgent(data)])
    training_environment_two = DummyVecEnv([lambda: AIAgentVsFixedRuleAgent(data)])

    # Train new models
    model_one = train_ai_agent(training_environment_one, total_timesteps=TOTAL_TIMESTEPS)
    model_two = train_ai_agent(training_environment_two, total_timesteps=TOTAL_TIMESTEPS)

    print("Trained new models.")

    # Evaluation environment (non-vectorised for two-agent evaluation)
    evaluation_environment = MultiAgentHFTEnv(data)

    # Evaluate the two AI agents
    evaluate_two_ai_agents(model_one, model_two, evaluation_environment)

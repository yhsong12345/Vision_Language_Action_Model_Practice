"""
Imitation Learning: Behavioral Cloning Example

This script demonstrates imitation learning using Behavioral Cloning (BC)
on the CartPole-v1 environment from Gymnasium.

Behavioral Cloning is the simplest form of imitation learning where we:
1. Collect expert demonstrations (state-action pairs)
2. Train a neural network via supervised learning to predict actions from states

Requirements:
    pip install gymnasium torch numpy matplotlib
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import deque
import random


# ============================================================================
# STEP 1: Define the Policy Network
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    A simple MLP policy network that maps states to action probabilities.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        """Select an action given a state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(state_tensor)
            if deterministic:
                action = torch.argmax(logits, dim=1).item()
            else:
                probs = torch.softmax(logits, dim=1)
                action = torch.multinomial(probs, 1).item()
        return action


# ============================================================================
# STEP 2: Create an Expert Policy (Rule-based for CartPole)
# ============================================================================

class ExpertPolicy:
    """
    A simple rule-based expert for CartPole.

    The expert uses the pole angle and angular velocity to decide actions:
    - If pole is falling right (positive angle), push right (action 1)
    - If pole is falling left (negative angle), push left (action 0)
    """
    def get_action(self, state: np.ndarray) -> int:
        # state = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        pole_angle = state[2]
        pole_velocity = state[3]

        # Simple control policy based on pole angle and angular velocity
        if pole_angle + 0.1 * pole_velocity > 0:
            return 1  # Push right
        else:
            return 0  # Push left


# ============================================================================
# STEP 3: Collect Expert Demonstrations
# ============================================================================

class DemonstrationDataset(Dataset):
    """Dataset for storing and loading expert demonstrations."""

    def __init__(self, states: np.ndarray, actions: np.ndarray):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return self.states[idx], self.actions[idx]


def collect_demonstrations(
    env: gym.Env,
    expert: ExpertPolicy,
    num_episodes: int = 50,
    max_steps: int = 500
) -> tuple:
    """
    Collect expert demonstrations by running the expert policy.

    Args:
        env: Gymnasium environment
        expert: Expert policy to collect demonstrations from
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode

    Returns:
        Tuple of (states, actions) numpy arrays
    """
    states_list = []
    actions_list = []
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = expert.get_action(state)

            # Store the state-action pair
            states_list.append(state)
            actions_list.append(action)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

            state = next_state

        episode_rewards.append(episode_reward)

    print(f"Expert Performance: Mean Reward = {np.mean(episode_rewards):.2f} "
          f"(+/- {np.std(episode_rewards):.2f})")
    print(f"Collected {len(states_list)} state-action pairs from {num_episodes} episodes")

    return np.array(states_list), np.array(actions_list)


# ============================================================================
# STEP 4: Train the Policy via Behavioral Cloning
# ============================================================================

def train_behavioral_cloning(
    policy: PolicyNetwork,
    dataset: DemonstrationDataset,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3
) -> list:
    """
    Train the policy network using behavioral cloning (supervised learning).

    Args:
        policy: Policy network to train
        dataset: Dataset of expert demonstrations
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer

    Returns:
        List of training losses
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for states, actions in dataloader:
            optimizer.zero_grad()

            # Forward pass
            logits = policy(states)
            loss = criterion(logits, actions)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return losses


# ============================================================================
# STEP 5: Evaluate the Learned Policy
# ============================================================================

def evaluate_policy(
    env: gym.Env,
    policy: PolicyNetwork,
    num_episodes: int = 20,
    max_steps: int = 500,
    render: bool = False
) -> tuple:
    """
    Evaluate the learned policy.

    Args:
        env: Gymnasium environment
        policy: Trained policy network
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment

    Returns:
        Tuple of (mean_reward, std_reward, episode_rewards)
    """
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = policy.get_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward, episode_rewards


# ============================================================================
# STEP 6: Visualization
# ============================================================================

def plot_results(losses: list, expert_rewards: list, learned_rewards: list):
    """Plot training loss and reward comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot training loss
    axes[0].plot(losses, color='blue', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Behavioral Cloning Training Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Plot reward comparison
    x_pos = [0, 1]
    means = [np.mean(expert_rewards), np.mean(learned_rewards)]
    stds = [np.std(expert_rewards), np.std(learned_rewards)]
    colors = ['green', 'blue']
    labels = ['Expert Policy', 'Learned Policy']

    bars = axes[1].bar(x_pos, means, yerr=stds, color=colors, alpha=0.7, capsize=10)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, fontsize=12)
    axes[1].set_ylabel('Episode Reward', fontsize=12)
    axes[1].set_title('Policy Performance Comparison', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                     f'{mean:.1f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('/purestorage/AILAB/AI_2/youhans/workspace/personal/VLA/IL/imitation_learning_results.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nResults saved to 'imitation_learning_results.png'")


# ============================================================================
# MAIN: Run the Complete Imitation Learning Pipeline
# ============================================================================

def main():
    """Main function to run the imitation learning pipeline."""

    print("=" * 60)
    print("IMITATION LEARNING: BEHAVIORAL CLONING ON CARTPOLE")
    print("=" * 60)

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # 4 for CartPole
    action_dim = env.action_space.n  # 2 for CartPole

    print(f"\nEnvironment: CartPole-v1")
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")

    # -------------------------------------------------------------------------
    # Step 1: Create expert and collect demonstrations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 1: Collecting Expert Demonstrations")
    print("-" * 60)

    expert = ExpertPolicy()
    states, actions = collect_demonstrations(env, expert, num_episodes=50)

    # Evaluate expert performance
    print("\nEvaluating Expert Policy...")
    expert_mean, expert_std, expert_rewards = evaluate_policy(
        env,
        type('', (), {'get_action': lambda self, s, **kwargs: expert.get_action(s)})(),
        num_episodes=20
    )
    print(f"Expert Policy: {expert_mean:.2f} (+/- {expert_std:.2f})")

    # -------------------------------------------------------------------------
    # Step 2: Create dataset and policy network
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 2: Creating Dataset and Policy Network")
    print("-" * 60)

    dataset = DemonstrationDataset(states, actions)
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim=128)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Policy Network:\n{policy}")

    # -------------------------------------------------------------------------
    # Step 3: Train via Behavioral Cloning
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 3: Training via Behavioral Cloning")
    print("-" * 60)

    losses = train_behavioral_cloning(
        policy,
        dataset,
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-3
    )

    # -------------------------------------------------------------------------
    # Step 4: Evaluate learned policy
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 4: Evaluating Learned Policy")
    print("-" * 60)

    learned_mean, learned_std, learned_rewards = evaluate_policy(
        env, policy, num_episodes=20
    )
    print(f"Learned Policy: {learned_mean:.2f} (+/- {learned_std:.2f})")

    # -------------------------------------------------------------------------
    # Step 5: Compare and visualize results
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 5: Results Summary")
    print("-" * 60)

    print(f"\n{'Policy':<20} {'Mean Reward':<15} {'Std Dev':<10}")
    print("-" * 45)
    print(f"{'Expert Policy':<20} {expert_mean:<15.2f} {expert_std:<10.2f}")
    print(f"{'Learned Policy':<20} {learned_mean:<15.2f} {learned_std:<10.2f}")

    improvement = (learned_mean / expert_mean) * 100 if expert_mean > 0 else 0
    print(f"\nLearned policy achieves {improvement:.1f}% of expert performance")

    # Plot results
    plot_results(losses, expert_rewards, learned_rewards)

    # Save the trained model
    model_path = '/purestorage/AILAB/AI_2/youhans/workspace/personal/VLA/IL/behavioral_cloning_model.pt'
    torch.save(policy.state_dict(), model_path)
    print(f"\nModel saved to '{model_path}'")

    env.close()

    print("\n" + "=" * 60)
    print("IMITATION LEARNING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

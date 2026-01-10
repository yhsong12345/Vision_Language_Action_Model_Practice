"""
Behavior Cloning Example - CartPole Environment

Behavior Cloning is a simple form of imitation learning where we:
1. Collect expert demonstrations (state-action pairs)
2. Train a neural network to predict actions given states using supervised learning

This example demonstrates:
- Generating expert data using a pre-trained/heuristic policy
- Training a policy network via supervised learning
- Evaluating the cloned behavior
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# 1. Expert Policy (Heuristic for CartPole)
# ============================================================================

def expert_policy(observation):
    """
    Simple heuristic expert for CartPole.
    The cart should move in the direction the pole is falling.

    Observation: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Action: 0 (push left) or 1 (push right)
    """
    # Use pole angle and angular velocity to decide action
    pole_angle = observation[2]
    pole_velocity = observation[3]

    # Push right if pole is falling right, push left if falling left
    if pole_angle + 0.1 * pole_velocity > 0:
        return 1  # Push right
    else:
        return 0  # Push left


# ============================================================================
# 2. Data Collection
# ============================================================================

def collect_expert_demonstrations(env, expert_fn, num_episodes=100):
    """
    Collect state-action pairs from expert demonstrations.

    Args:
        env: Gymnasium environment
        expert_fn: Expert policy function
        num_episodes: Number of episodes to collect

    Returns:
        states: numpy array of states
        actions: numpy array of actions
    """
    states = []
    actions = []
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = expert_fn(state)
            states.append(state)
            actions.append(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)

    print(f"Expert average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Collected {len(states)} state-action pairs from {num_episodes} episodes")

    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)


# ============================================================================
# 3. Dataset and DataLoader
# ============================================================================

class ExpertDataset(Dataset):
    """PyTorch Dataset for expert demonstrations."""

    def __init__(self, states, actions):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


# ============================================================================
# 4. Policy Network (Neural Network to Clone Behavior)
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Simple MLP policy network for behavior cloning.
    Maps states to action probabilities.
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        super(PolicyNetwork, self).__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Regularization
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """Returns action logits."""
        return self.network(state)

    def get_action(self, state):
        """Get action for a single state (for evaluation)."""
        self.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            logits = self.forward(state)
            action = torch.argmax(logits, dim=1)
        return action.item()


# ============================================================================
# 5. Training Loop
# ============================================================================

def train_behavior_cloning(
    policy: PolicyNetwork,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu"
):
    """
    Train the policy network using behavior cloning (supervised learning).

    Args:
        policy: Policy network to train
        train_loader: DataLoader with training data
        val_loader: Optional validation DataLoader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        Dictionary with training history
    """
    policy = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()  # Classification loss for discrete actions

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        # Training phase
        policy.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)

            # Forward pass
            logits = policy(states)
            loss = criterion(logits, actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * states.size(0)
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == actions).sum().item()
            train_total += states.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation phase
        if val_loader is not None:
            policy.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for states, actions in val_loader:
                    states = states.to(device)
                    actions = actions.to(device)

                    logits = policy(states)
                    loss = criterion(logits, actions)

                    val_loss += loss.item() * states.size(0)
                    predictions = torch.argmax(logits, dim=1)
                    val_correct += (predictions == actions).sum().item()
                    val_total += states.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    return history


# ============================================================================
# 6. Evaluation
# ============================================================================

def evaluate_policy(env, policy, num_episodes=20, render=False):
    """
    Evaluate the learned policy in the environment.

    Args:
        env: Gymnasium environment
        policy: Trained policy network
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment

    Returns:
        Average reward and standard deviation
    """
    policy.eval()
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = policy.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


# ============================================================================
# 7. Visualization
# ============================================================================

def plot_training_history(history, save_path=None):
    """Plot training metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss")
    if history["val_loss"]:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(history["train_acc"], label="Train Accuracy")
    if history["val_acc"]:
        axes[1].plot(history["val_acc"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()


# ============================================================================
# 8. Main Execution
# ============================================================================

def main():
    # Configuration
    ENV_NAME = "CartPole-v1"
    NUM_EXPERT_EPISODES = 100
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    TRAIN_VAL_SPLIT = 0.8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")
    print("=" * 60)

    # Step 1: Create environment and collect expert demonstrations
    print("\n[Step 1] Collecting expert demonstrations...")
    env = gym.make(ENV_NAME)
    states, actions = collect_expert_demonstrations(
        env, expert_policy, num_episodes=NUM_EXPERT_EPISODES
    )

    # Step 2: Create train/validation split
    print("\n[Step 2] Creating dataset...")
    num_samples = len(states)
    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * TRAIN_VAL_SPLIT)

    train_states = states[indices[:split_idx]]
    train_actions = actions[indices[:split_idx]]
    val_states = states[indices[split_idx:]]
    val_actions = actions[indices[split_idx:]]

    train_dataset = ExpertDataset(train_states, train_actions)
    val_dataset = ExpertDataset(val_states, val_actions)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Step 3: Create policy network
    print("\n[Step 3] Creating policy network...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64]
    )
    print(policy)

    # Step 4: Train the policy using behavior cloning
    print("\n[Step 4] Training policy via behavior cloning...")
    history = train_behavior_cloning(
        policy=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=DEVICE
    )

    # Step 5: Evaluate the learned policy
    print("\n[Step 5] Evaluating learned policy...")
    eval_env = gym.make(ENV_NAME)

    mean_reward, std_reward = evaluate_policy(eval_env, policy, num_episodes=50)
    print(f"Cloned Policy - Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Compare with random policy
    random_rewards = []
    for _ in range(50):
        state, _ = eval_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = eval_env.action_space.sample()
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        random_rewards.append(episode_reward)

    print(f"Random Policy - Average Reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")

    # Step 6: Plot results
    print("\n[Step 6] Plotting training history...")
    plot_training_history(history, save_path="training_history.png")

    # Cleanup
    env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Behavior Cloning Complete!")
    print("=" * 60)

    return policy, history


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    policy, history = main()
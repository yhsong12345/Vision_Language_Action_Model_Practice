"""
GAIL (Generative Adversarial Imitation Learning)

Learns reward function and policy simultaneously:
- Discriminator distinguishes expert vs. policy trajectories
- Policy optimizes against learned reward
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm

from .base_trainer import ILTrainer, ExpertDataset, PolicyNetwork

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.training_config import ILConfig


class Discriminator(nn.Module):
    """
    Discriminator network for GAIL.

    Classifies state-action pairs as expert or policy generated.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict probability of (state, action) being from expert.

        Args:
            state: (batch, state_dim)
            action: (batch, action_dim)

        Returns:
            prob: (batch, 1) probability of being expert
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

    def get_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get GAIL reward for (state, action).

        reward = -log(1 - D(s,a))

        This encourages policy to be classified as expert.
        """
        with torch.no_grad():
            d = self.forward(state, action)
            reward = -torch.log(1 - d + 1e-8)
        return reward


class ActorCriticGAIL(nn.Module):
    """Actor-Critic network for GAIL policy optimization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = True,
    ):
        super().__init__()

        self.continuous = continuous

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor (policy)
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        features = self.features(state)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp()
            value = self.critic(features)
            return mean, std, value
        else:
            logits = self.actor(features)
            value = self.critic(features)
            return logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            if self.continuous:
                mean, std, _ = self.forward(state)
                if deterministic:
                    return mean.squeeze(0)
                dist = torch.distributions.Normal(mean, std)
                return dist.sample().squeeze(0)
            else:
                logits, _ = self.forward(state)
                if deterministic:
                    return torch.argmax(logits, dim=-1).squeeze(0)
                probs = F.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1).squeeze(-1).squeeze(0)

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            logits, value = self.forward(state)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return value.squeeze(-1), log_prob, entropy


class GAIL(ILTrainer):
    """
    GAIL (Generative Adversarial Imitation Learning) trainer.

    Alternates between:
    1. Update discriminator to distinguish expert vs. policy
    2. Update policy using PPO with discriminator as reward

    Advantages over BC:
    - Can achieve better than expert performance
    - No need for action labels (can work with state-only demos)
    - Learns reward function that generalizes

    Disadvantages:
    - Requires environment interaction
    - More complex training
    - Can be unstable
    """

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        config: Optional[ILConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = ILConfig.gail()

        # Create actor-critic policy
        state_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, 'shape'):
            action_dim = env.action_space.shape[0]
            continuous = True
        else:
            action_dim = env.action_space.n
            continuous = False

        if policy is None:
            policy = ActorCriticGAIL(
                state_dim, action_dim, continuous=continuous
            )

        super().__init__(env, policy, config.output_dir, **kwargs)

        self.config = config

        # Create discriminator
        self.discriminator = Discriminator(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.gail_disc_hidden_dim,
        ).to(self.device)

        # Optimizers
        self.policy_optimizer = Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
        )
        self.disc_optimizer = Adam(
            self.discriminator.parameters(),
            lr=config.gail_disc_lr,
        )

        # PPO params
        self.ppo_epochs = 4
        self.clip_range = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.gamma = 0.99
        self.gae_lambda = 0.95

        # GAIL params
        self.disc_updates = config.gail_disc_updates
        self.reward_scale = config.gail_reward_scale

    def train(
        self,
        expert_states: Optional[np.ndarray] = None,
        expert_actions: Optional[np.ndarray] = None,
        expert_policy=None,
        num_expert_episodes: int = 50,
        total_timesteps: int = 100000,
        rollout_steps: int = 2048,
    ):
        """
        Run GAIL training.

        Args:
            expert_states: Expert states
            expert_actions: Expert actions
            expert_policy: Expert policy for collecting demos
            num_expert_episodes: Episodes to collect if using expert_policy
            total_timesteps: Total training timesteps
            rollout_steps: Steps per rollout
        """
        print("=" * 60)
        print("GAIL Training")
        print("=" * 60)

        # Collect expert demonstrations if not provided
        if expert_states is None or expert_actions is None:
            if expert_policy is None:
                raise ValueError("Must provide expert data or policy")

            expert_states, expert_actions = self.collect_expert_demonstrations(
                expert_policy, num_expert_episodes
            )

        # Create expert dataloader
        expert_dataset = ExpertDataset(expert_states, expert_actions)
        expert_loader = DataLoader(
            expert_dataset,
            batch_size=64,
            shuffle=True,
        )

        # Training loop
        timestep = 0
        best_reward = float("-inf")

        progress_bar = tqdm(total=total_timesteps, desc="Training")

        while timestep < total_timesteps:
            # Collect rollout
            rollout = self._collect_rollout(rollout_steps)
            timestep += len(rollout["states"])

            # Update discriminator
            disc_loss = self._update_discriminator(
                rollout, expert_loader
            )

            # Compute GAIL rewards
            gail_rewards = self._compute_gail_rewards(rollout)

            # Update policy with PPO
            policy_metrics = self._update_policy(rollout, gail_rewards)

            # Logging
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(list(self.episode_rewards))

                progress_bar.set_postfix({
                    "reward": f"{mean_reward:.1f}",
                    "disc_loss": f"{disc_loss:.3f}",
                })

                if mean_reward > best_reward:
                    best_reward = mean_reward
                    self.save(os.path.join(self.config.output_dir, "best_policy.pt"))

            progress_bar.update(len(rollout["states"]))

        progress_bar.close()

        # Final evaluation
        print("\nFinal Evaluation:")
        eval_results = self.evaluate()
        print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

        self.save()
        return eval_results

    def _collect_rollout(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """Collect rollout data."""
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        state, _ = self.env.reset()

        for _ in range(num_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                if self.continuous:
                    mean, std, value = self.policy(state_tensor.unsqueeze(0))
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(-1)
                else:
                    logits, value = self.policy(state_tensor.unsqueeze(0))
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

            action_np = action.squeeze(0).cpu().numpy()

            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            states.append(state)
            actions.append(action_np)
            log_probs.append(log_prob.cpu())
            values.append(value.squeeze().cpu())
            rewards.append(reward)
            dones.append(done)

            if done:
                self.episode_rewards.append(reward)
                state, _ = self.env.reset()
            else:
                state = next_state

        return {
            "states": torch.tensor(np.array(states), dtype=torch.float32),
            "actions": torch.tensor(np.array(actions), dtype=torch.float32),
            "log_probs": torch.stack(log_probs),
            "values": torch.stack(values),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "dones": torch.tensor(dones, dtype=torch.float32),
        }

    def _update_discriminator(
        self,
        rollout: Dict[str, torch.Tensor],
        expert_loader: DataLoader,
    ) -> float:
        """Update discriminator."""
        total_loss = 0

        for _ in range(self.disc_updates):
            # Get expert batch
            try:
                expert_states, expert_actions = next(iter(expert_loader))
            except StopIteration:
                expert_loader = DataLoader(expert_loader.dataset, batch_size=64, shuffle=True)
                expert_states, expert_actions = next(iter(expert_loader))

            expert_states = expert_states.to(self.device)
            expert_actions = expert_actions.to(self.device)

            # Sample policy batch
            batch_size = len(expert_states)
            indices = np.random.choice(len(rollout["states"]), batch_size)
            policy_states = rollout["states"][indices].to(self.device)
            policy_actions = rollout["actions"][indices].to(self.device)

            # Expert prediction (should be 1)
            expert_pred = self.discriminator(expert_states, expert_actions)
            expert_loss = F.binary_cross_entropy(
                expert_pred,
                torch.ones_like(expert_pred),
            )

            # Policy prediction (should be 0)
            policy_pred = self.discriminator(policy_states, policy_actions)
            policy_loss = F.binary_cross_entropy(
                policy_pred,
                torch.zeros_like(policy_pred),
            )

            loss = expert_loss + policy_loss

            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()

            total_loss += loss.item()

        return total_loss / self.disc_updates

    def _compute_gail_rewards(
        self,
        rollout: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute GAIL rewards using discriminator."""
        states = rollout["states"].to(self.device)
        actions = rollout["actions"].to(self.device)

        rewards = self.discriminator.get_reward(states, actions)
        return rewards.squeeze(-1) * self.reward_scale

    def _update_policy(
        self,
        rollout: Dict[str, torch.Tensor],
        gail_rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """Update policy using PPO with GAIL rewards."""
        states = rollout["states"].to(self.device)
        actions = rollout["actions"].to(self.device)
        old_log_probs = rollout["log_probs"].to(self.device)
        old_values = rollout["values"].to(self.device)
        dones = rollout["dones"].to(self.device)

        # Compute advantages using GAE
        advantages = torch.zeros_like(gail_rewards)
        returns = torch.zeros_like(gail_rewards)
        last_gae = 0

        for t in reversed(range(len(gail_rewards))):
            if t == len(gail_rewards) - 1:
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = old_values[t + 1]
                next_non_terminal = 1 - dones[t]

            delta = gail_rewards[t] + self.gamma * next_value * next_non_terminal - old_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + old_values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.ppo_epochs):
            values, log_probs, entropy = self.policy.evaluate_actions(states, actions)

            ratio = torch.exp(log_probs - old_log_probs.squeeze())

            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.policy_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }


class VLAGAIL:
    """
    GAIL for VLA models.

    Generative Adversarial Imitation Learning for vision-language-action models.
    Learns a reward function that distinguishes expert from policy trajectories.
    """

    def __init__(
        self,
        model,
        config: Optional[ILConfig] = None,
    ):
        if config is None:
            config = ILConfig.gail()

        self.model = model
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Get action dimension from model
        self.action_dim = 7  # Default robot action dimension

        # Create discriminator for VLA
        self.discriminator = nn.Sequential(
            nn.Linear(256 + self.action_dim, config.gail_disc_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gail_disc_hidden_dim, config.gail_disc_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gail_disc_hidden_dim, 1),
            nn.Sigmoid(),
        ).to(self.device)

        # Feature extractor for images
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(3 * 16, 256),
            nn.ReLU(),
        ).to(self.device)

        # Optimizers
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.policy_optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
        self.disc_optimizer = torch.optim.Adam(
            list(self.discriminator.parameters()) + list(self.feature_extractor.parameters()),
            lr=config.gail_disc_lr,
        )

        self.disc_updates = config.gail_disc_updates
        self.reward_scale = config.gail_reward_scale

        os.makedirs(config.output_dir, exist_ok=True)

    def get_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        return self.feature_extractor(pixel_values)

    def get_reward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get GAIL reward."""
        x = torch.cat([features, actions], dim=-1)
        d = self.discriminator(x)
        return -torch.log(1 - d + 1e-8) * self.reward_scale

    def update_discriminator(
        self,
        expert_batch: Dict[str, torch.Tensor],
        policy_actions: torch.Tensor,
    ) -> float:
        """Update discriminator."""
        total_loss = 0

        for _ in range(self.disc_updates):
            expert_features = self.get_features(expert_batch["pixel_values"])
            expert_actions = expert_batch["action"]
            policy_features = expert_features.detach()

            expert_x = torch.cat([expert_features, expert_actions], dim=-1)
            expert_pred = self.discriminator(expert_x)
            expert_loss = F.binary_cross_entropy(expert_pred, torch.ones_like(expert_pred))

            policy_x = torch.cat([policy_features, policy_actions.detach()], dim=-1)
            policy_pred = self.discriminator(policy_x)
            policy_loss = F.binary_cross_entropy(policy_pred, torch.zeros_like(policy_pred))

            loss = expert_loss + policy_loss

            self.disc_optimizer.zero_grad()
            loss.backward()
            self.disc_optimizer.step()

            total_loss += loss.item()

        return total_loss / self.disc_updates

    def train(self, expert_dataloader, total_steps: int = 10000):
        """Run GAIL training."""
        print("=" * 60)
        print("VLA GAIL Training")
        print("=" * 60)

        step = 0
        best_reward = float("-inf")
        progress_bar = tqdm(total=total_steps, desc="Training")

        while step < total_steps:
            for batch in expert_dataloader:
                if step >= total_steps:
                    break

                batch = {k: v.to(self.device) for k, v in batch.items()}

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(
                        pixel_values=batch["pixel_values"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                policy_actions = outputs["predicted_actions"]

                self.model.train()
                disc_loss = self.update_discriminator(batch, policy_actions)

                features = self.get_features(batch["pixel_values"])
                rewards = self.get_reward(features, policy_actions)

                outputs = self.model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    actions=batch["action"],
                )

                bc_loss = outputs["loss"]
                reward_bonus = -rewards.mean()
                loss = bc_loss + 0.1 * reward_bonus

                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.policy_optimizer.step()

                step += 1
                progress_bar.update(1)

                if step % 100 == 0:
                    mean_reward = rewards.mean().item()
                    progress_bar.set_postfix({"disc_loss": f"{disc_loss:.3f}", "reward": f"{mean_reward:.3f}"})
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        self.save(os.path.join(self.config.output_dir, "best_model.pt"))

        progress_bar.close()
        self.save(os.path.join(self.config.output_dir, "final_model.pt"))

    def save(self, path: str):
        """Save model."""
        torch.save({
            "model": self.model.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "feature_extractor": self.feature_extractor.state_dict(),
        }, path)
        print(f"Saved model to {path}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="GAIL Training")

    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment")
    parser.add_argument("--expert_data", type=str, default=None, help="Path to expert data (.npz)")
    parser.add_argument("--num_expert_episodes", type=int, default=50, help="Episodes to collect")
    parser.add_argument("--model_path", type=str, default=None, help="VLA model path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--gail_disc_hidden_dim", type=int, default=256)
    parser.add_argument("--gail_disc_updates", type=int, default=5)
    parser.add_argument("--gail_disc_lr", type=float, default=3e-4)
    parser.add_argument("--gail_reward_scale", type=float, default=1.0)
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="./output/gail")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def create_simple_expert(env_name: str):
    """Create simple expert policies for common environments."""
    if env_name == "CartPole-v1":
        def policy(state):
            return 1 if state[2] + 0.1 * state[3] > 0 else 0
        return policy
    raise ValueError(f"No simple expert for {env_name}. Provide --expert_data instead.")


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("GAIL Training")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.model_path is not None:
        print("VLA GAIL mode")

        from model.vla import VLAModel
        from model.vla.vla_model import VLAConfig
        from train.finetune.dataset import RobotDataset
        from torch.utils.data import DataLoader

        model_config = VLAConfig()
        model = VLAModel(model_config)

        if os.path.exists(args.model_path):
            state_dict = torch.load(args.model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded VLA model from {args.model_path}")

        config = ILConfig(
            gail_disc_hidden_dim=args.gail_disc_hidden_dim,
            gail_disc_updates=args.gail_disc_updates,
            gail_disc_lr=args.gail_disc_lr,
            gail_reward_scale=args.gail_reward_scale,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        trainer = VLAGAIL(model, config=config)

        try:
            dataset = RobotDataset(dataset_name="lerobot/pusht", max_samples=1000)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            trainer.train(dataloader, total_steps=args.total_timesteps)
        except Exception as e:
            print(f"Error loading dataset: {e}")

    else:
        print(f"Environment: {args.env}")

        import gymnasium as gym
        env = gym.make(args.env)

        config = ILConfig(
            gail_disc_hidden_dim=args.gail_disc_hidden_dim,
            gail_disc_updates=args.gail_disc_updates,
            gail_disc_lr=args.gail_disc_lr,
            gail_reward_scale=args.gail_reward_scale,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )

        trainer = GAIL(env, config=config)

        if args.resume:
            trainer.load(args.resume)

        if args.expert_data and os.path.exists(args.expert_data):
            data = np.load(args.expert_data)
            trainer.train(expert_states=data["states"], expert_actions=data["actions"], total_timesteps=args.total_timesteps)
        else:
            expert_policy = create_simple_expert(args.env)
            trainer.train(expert_policy=expert_policy, num_expert_episodes=args.num_expert_episodes, total_timesteps=args.total_timesteps)

    print("\nTraining complete!")

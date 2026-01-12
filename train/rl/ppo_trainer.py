"""
PPO (Proximal Policy Optimization) Trainer

Implements PPO algorithm for robot control:
- Clipped surrogate objective
- GAE for advantage estimation
- Multiple epochs per rollout
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm

from .base_trainer import RLTrainer, RolloutBuffer, ActorCritic

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.training_config import RLConfig


class PPOTrainer(RLTrainer):
    """
    PPO Trainer for robot control.

    Features:
    - Clipped surrogate objective
    - Value function clipping (optional)
    - Entropy bonus
    - GAE for advantage estimation
    """

    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        config: Optional[RLConfig] = None,
        **kwargs,
    ):
        # Default config
        if config is None:
            config = RLConfig.ppo_default()

        # Create default policy if not provided
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n

        if policy is None:
            policy = ActorCritic(obs_dim, action_dim)

        super().__init__(
            env=env,
            policy=policy,
            output_dir=config.output_dir,
            gamma=config.discount_gamma,
            total_timesteps=config.total_timesteps,
            **kwargs,
        )

        self.config = config

        # PPO specific params
        self.clip_range = config.ppo_clip_range
        self.clip_range_vf = config.ppo_clip_range_vf
        self.value_coef = config.ppo_value_coef
        self.entropy_coef = config.ppo_entropy_coef
        self.gae_lambda = config.ppo_gae_lambda
        self.ppo_epochs = config.ppo_epochs
        self.rollout_steps = config.rollout_steps
        self.batch_size = config.batch_size

        # Create buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_steps,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=str(self.device),
        )

        # Optimizer
        self.optimizer = Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )

    def train(self):
        """Run PPO training."""
        print("=" * 60)
        print("PPO Training")
        print("=" * 60)

        obs, _ = self.env.reset()
        timestep = 0
        best_reward = float("-inf")

        progress_bar = tqdm(total=self.total_timesteps, desc="Training")

        while timestep < self.total_timesteps:
            # Collect rollout
            self.buffer.clear()

            for _ in range(self.rollout_steps):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    action, value, log_prob = self.policy.get_action_value(obs_tensor)

                action_np = action.cpu().numpy()
                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
                done = terminated or truncated

                self.buffer.add(
                    obs=obs_tensor.cpu(),
                    action=action.cpu(),
                    reward=reward,
                    done=done,
                    value=value.cpu().item(),
                    log_prob=log_prob.cpu().item(),
                )

                if done:
                    obs, _ = self.env.reset()
                    self.episode_rewards.append(reward)
                else:
                    obs = next_obs

                timestep += 1

            # Compute returns and advantages
            with torch.no_grad():
                last_obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
                _, last_value = self.policy(last_obs)
                last_value = last_value.cpu().item()

            self.buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )

            # PPO update
            metrics = self.learn_step()

            # Logging
            if timestep % self.log_freq == 0:
                progress_bar.set_postfix({
                    "reward": f"{np.mean(list(self.episode_rewards)):.1f}" if self.episode_rewards else "N/A",
                    "loss": f"{metrics['loss']:.3f}",
                })

            # Evaluation
            if timestep % self.eval_freq == 0:
                eval_results = self.evaluate()
                print(f"\nStep {timestep}: Mean Reward = {eval_results['mean_reward']:.2f}")

                if eval_results["mean_reward"] > best_reward:
                    best_reward = eval_results["mean_reward"]
                    self.save(os.path.join(self.output_dir, "best_policy.pt"))

            # Save checkpoint
            if timestep % self.save_freq == 0:
                self.save(os.path.join(self.output_dir, f"policy_{timestep}.pt"))

            progress_bar.update(self.rollout_steps)

        progress_bar.close()
        self.save()

    def learn_step(self) -> Dict[str, float]:
        """Perform PPO update."""
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                observations = batch["observations"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]

                # Evaluate actions
                values, log_probs, entropy = self.policy.evaluate_actions(
                    observations, actions
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.clip_range_vf is not None:
                    # Clipped value loss
                    values_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                    value_loss1 = F.mse_loss(values, returns)
                    value_loss2 = F.mse_loss(values_clipped, returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def _store_transition(self, obs, action, reward, done, value, log_prob):
        """Store transition (handled in collect_rollout)."""
        pass


def train_ppo(
    env_name: str = "CartPole-v1",
    total_timesteps: int = 100000,
    **kwargs,
):
    """
    Convenience function for PPO training.

    Args:
        env_name: Gymnasium environment name
        total_timesteps: Total training timesteps
        **kwargs: Additional config arguments
    """
    import gymnasium as gym

    env = gym.make(env_name)
    config = RLConfig(
        total_timesteps=total_timesteps,
        **kwargs,
    )

    trainer = PPOTrainer(env, config=config)
    trainer.train()

    return trainer.policy


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="PPO Training")

    # Environment
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # PPO parameters
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--ppo_clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ppo_value_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--ppo_entropy_coef", type=float, default=0.01, help="Entropy coefficient")

    # Training
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--discount_gamma", type=float, default=0.99, help="Discount factor")

    # Logging
    parser.add_argument("--eval_freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--save_freq", type=int, default=10000, help="Checkpoint frequency")
    parser.add_argument("--log_freq", type=int, default=1000, help="Logging frequency")

    # Output
    parser.add_argument("--output_dir", type=str, default="./output/ppo", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("PPO Training")
    print("=" * 60)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import gymnasium as gym
    env = gym.make(args.env)

    print(f"Environment: {args.env}")
    print(f"Total timesteps: {args.total_timesteps}")

    config = RLConfig(
        algorithm="ppo",
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        ppo_clip_range=args.ppo_clip_range,
        ppo_gae_lambda=args.ppo_gae_lambda,
        ppo_value_coef=args.ppo_value_coef,
        ppo_entropy_coef=args.ppo_entropy_coef,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        discount_gamma=args.discount_gamma,
        output_dir=args.output_dir,
    )

    trainer = PPOTrainer(env, config=config)

    if args.resume:
        trainer.load(args.resume)

    trainer.train()

    print("\nTraining complete!")

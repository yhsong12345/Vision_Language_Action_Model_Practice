"""
Decision Transformer Trainer - Offline RL

Implements Decision Transformer for offline reinforcement learning:
- Sequence modeling approach to RL
- Conditions on desired return (returns-to-go)
- GPT-style causal transformer architecture
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import math

from .base_trainer import OfflineRLTrainer, OfflineRLConfig, OfflineReplayBuffer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class DecisionTransformer(nn.Module):
    """
    Decision Transformer model.

    Architecture:
    - Embeds (return, state, action) triples
    - GPT-style causal transformer
    - Predicts actions autoregressively
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 1,
        max_ep_len: int = 1000,
        max_seq_len: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)
        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(action_dim, hidden_dim)

        self.embed_ln = nn.LayerNorm(hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction heads
        self.predict_state = nn.Linear(hidden_dim, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.predict_return = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Decision Transformer.

        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
            returns_to_go: (batch, seq_len, 1)
            timesteps: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            state_preds, action_preds, return_preds
        """
        batch_size, seq_len = states.shape[:2]

        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Add time embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stack: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = torch.stack(
            [returns_embeddings, state_embeddings, action_embeddings], dim=2
        ).reshape(batch_size, 3 * seq_len, self.hidden_dim)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            3 * seq_len, device=states.device
        )

        # Attention mask
        if attention_mask is not None:
            # Expand mask to 3x for R, s, a
            stacked_mask = torch.stack(
                [attention_mask, attention_mask, attention_mask], dim=2
            ).reshape(batch_size, 3 * seq_len)
            padding_mask = ~stacked_mask.bool()
        else:
            padding_mask = None

        # Transformer
        transformer_outputs = self.transformer(
            stacked_inputs,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        # Reshape to separate R, s, a
        transformer_outputs = transformer_outputs.reshape(batch_size, seq_len, 3, self.hidden_dim)
        return_outputs = transformer_outputs[:, :, 0]
        state_outputs = transformer_outputs[:, :, 1]
        action_outputs = transformer_outputs[:, :, 2]

        # Predictions
        return_preds = self.predict_return(return_outputs)
        state_preds = self.predict_state(state_outputs)
        action_preds = self.predict_action(state_outputs)  # Predict action from state

        return state_preds, action_preds, return_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Get action prediction for the last timestep."""
        _, action_preds, _ = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[:, -1]


class DecisionTransformerTrainer(OfflineRLTrainer):
    """
    Decision Transformer Trainer.

    Trains the transformer using supervised learning on offline data,
    conditioning on returns-to-go for desired behavior.

    Reference: Chen et al., "Decision Transformer: Reinforcement
               Learning via Sequence Modeling"
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[OfflineRLConfig] = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 1,
        max_ep_len: int = 1000,
        context_len: int = 20,
    ):
        if config is None:
            config = OfflineRLConfig()

        # Create model
        model = DecisionTransformer(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_ep_len=max_ep_len,
            max_seq_len=context_len,
        )

        super().__init__(config, model)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_len = context_len
        self.max_ep_len = max_ep_len

        # Optimizer
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min((step + 1) / 1000, 1),  # Warmup
        )

        # For evaluation
        self.target_return = 3600  # Can be tuned per environment

    def train(self, buffer: OfflineReplayBuffer):
        """Run Decision Transformer offline training."""
        print("=" * 60)
        print("Decision Transformer Offline Training")
        print("=" * 60)
        print(f"Context length: {self.context_len}")
        print(f"Dataset size: {buffer.size}")

        num_batches = buffer.size // (self.config.batch_size * self.context_len)
        best_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            epoch_metrics = {"loss": [], "action_loss": []}

            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for _ in progress_bar:
                batch = buffer.sample_trajectories(self.config.batch_size, self.context_len)
                metrics = self.train_step(batch)

                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)

                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                })

            # Epoch summary
            avg_loss = np.mean(epoch_metrics["loss"])
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save(os.path.join(self.config.output_dir, "best_model.pt"))

            # Periodic save
            if (epoch + 1) % self.config.save_freq == 0:
                self.save(os.path.join(self.config.output_dir, f"model_epoch_{epoch+1}.pt"))

        self.save(os.path.join(self.config.output_dir, "final_model.pt"))

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one Decision Transformer training step."""
        states = batch["observations"]
        actions = batch["actions"]
        returns_to_go = batch["returns_to_go"].unsqueeze(-1)
        timesteps = batch["timesteps"]
        attention_mask = (batch["dones"].cumsum(dim=1) == 0).float()

        # Forward pass
        state_preds, action_preds, return_preds = self.policy(
            states, actions, returns_to_go, timesteps, attention_mask
        )

        # Action prediction loss (main objective)
        action_loss = F.mse_loss(action_preds, actions, reduction="none")
        action_loss = (action_loss * attention_mask.unsqueeze(-1)).mean()

        # Optional: state and return prediction losses
        state_loss = F.mse_loss(state_preds[:, :-1], states[:, 1:], reduction="none")
        state_loss = (state_loss * attention_mask[:, 1:].unsqueeze(-1)).mean()

        total_loss = action_loss + 0.1 * state_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.25)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "state_loss": state_loss.item(),
        }

    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Select action using Decision Transformer.

        Note: This requires maintaining context of past states/actions/returns.
        """
        # For simple evaluation, just use current observation
        # In practice, you'd maintain a history buffer

        batch_size = obs.shape[0]

        # Create dummy context
        states = obs.unsqueeze(1)  # (batch, 1, state_dim)
        actions = torch.zeros(batch_size, 1, self.action_dim, device=self.device)
        returns_to_go = torch.ones(batch_size, 1, 1, device=self.device) * self.target_return
        timesteps = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)

        return self.policy.get_action(states, actions, returns_to_go, timesteps)

    def evaluate_with_context(
        self,
        env,
        num_episodes: int = 10,
        target_return: float = None,
    ) -> Dict[str, float]:
        """
        Evaluate with proper context management.

        Maintains history of states, actions, returns-to-go during episode.
        """
        if target_return is not None:
            self.target_return = target_return

        self.policy.eval()
        episode_rewards = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False

            # Initialize context
            states = torch.zeros(1, self.context_len, self.obs_dim, device=self.device)
            actions = torch.zeros(1, self.context_len, self.action_dim, device=self.device)
            returns_to_go = torch.zeros(1, self.context_len, 1, device=self.device)
            timesteps = torch.arange(self.context_len, device=self.device).unsqueeze(0)

            # Set initial return target
            returns_to_go[0, 0, 0] = self.target_return

            t = 0
            while not done:
                # Add current state to context
                states[0, min(t, self.context_len - 1)] = torch.tensor(obs, device=self.device)

                # Get action
                with torch.no_grad():
                    if t < self.context_len:
                        context_states = states[:, :t+1]
                        context_actions = actions[:, :t+1]
                        context_rtg = returns_to_go[:, :t+1]
                        context_timesteps = timesteps[:, :t+1]
                    else:
                        context_states = states
                        context_actions = actions
                        context_rtg = returns_to_go
                        context_timesteps = timesteps

                    action = self.policy.get_action(
                        context_states, context_actions, context_rtg, context_timesteps
                    )

                action_np = action[0].cpu().numpy()
                obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated

                # Update context
                if t < self.context_len:
                    actions[0, t] = action[0]
                    if t + 1 < self.context_len:
                        returns_to_go[0, t + 1] = returns_to_go[0, t] - reward
                else:
                    # Shift context
                    states = torch.roll(states, -1, dims=1)
                    actions = torch.roll(actions, -1, dims=1)
                    returns_to_go = torch.roll(returns_to_go, -1, dims=1)

                    states[0, -1] = torch.tensor(obs, device=self.device)
                    actions[0, -1] = action[0]
                    returns_to_go[0, -1] = returns_to_go[0, -2] - reward

                episode_reward += reward
                t += 1

            episode_rewards.append(episode_reward)

        self.policy.train()

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
        }


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Decision Transformer Offline Training")

    parser.add_argument("--dataset", type=str, default="hopper-medium-v2", help="D4RL dataset")
    parser.add_argument("--num_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--context_len", type=int, default=20, help="Context length")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--output_dir", type=str, default="./output/dt", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Decision Transformer Offline Training")
    print("=" * 60)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = OfflineRLConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )

    trainer = DecisionTransformerTrainer(
        obs_dim=11,
        action_dim=3,
        config=config,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        context_len=args.context_len,
    )

    # Load dataset
    from .base_trainer import create_dummy_dataset
    dataset = create_dummy_dataset(obs_dim=11, action_dim=3)

    buffer = OfflineReplayBuffer(obs_dim=11, action_dim=3, device=str(trainer.device))
    buffer.load_dataset(dataset)

    trainer.train(buffer)

    print("\nTraining complete!")

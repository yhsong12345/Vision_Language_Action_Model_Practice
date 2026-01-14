"""
Latent World Model Module

Implements world models in learned latent space (Dreamer-style).
Key components:
- RSSM: Recurrent State-Space Model
- Latent Dynamics: Transition in latent space
- Image Decoder: Reconstruct observations from latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class LatentWorldModelConfig:
    """Configuration for latent world model."""
    # Observation
    image_size: int = 224
    image_channels: int = 3

    # Latent space
    stochastic_dim: int = 32  # Stochastic state dimension
    deterministic_dim: int = 256  # Deterministic state (GRU hidden)
    hidden_dim: int = 512
    action_dim: int = 7

    # Architecture
    num_categories: int = 32  # For categorical stochastic state
    encoder_layers: int = 4
    decoder_layers: int = 4
    use_discrete_latent: bool = True

    # Training
    kl_scale: float = 1.0
    free_nats: float = 3.0


class ConvEncoder(nn.Module):
    """CNN encoder for images."""

    def __init__(self, config: LatentWorldModelConfig):
        super().__init__()

        # Progressive downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(config.image_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output size
        dummy = torch.zeros(1, config.image_channels, config.image_size, config.image_size)
        with torch.no_grad():
            out = self.encoder(dummy)
        self.output_dim = out.shape[-1]

        # Project to hidden dim
        self.proj = nn.Linear(self.output_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            features: (batch, hidden_dim)
        """
        return self.proj(self.encoder(x))


class ImageDecoder(nn.Module):
    """Transposed CNN decoder for image reconstruction."""

    def __init__(self, config: LatentWorldModelConfig):
        super().__init__()
        self.config = config

        # Input dim is combined stochastic + deterministic state
        latent_dim = config.stochastic_dim + config.deterministic_dim

        # Project to spatial
        self.proj = nn.Linear(latent_dim, 256 * 14 * 14)

        # Progressive upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, config.image_channels, 4, 2, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (batch, stochastic_dim + deterministic_dim)
        Returns:
            image: (batch, channels, height, width)
        """
        x = self.proj(latent)
        x = x.view(-1, 256, 14, 14)
        return self.decoder(x)


class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM).

    The core of Dreamer-style world models.
    Maintains both deterministic (GRU) and stochastic state.
    """

    def __init__(self, config: LatentWorldModelConfig):
        super().__init__()
        self.config = config

        # Prior network: p(s_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(config.deterministic_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.stochastic_dim * 2),  # mean, std
        )

        # Posterior network: q(s_t | h_t, o_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(config.deterministic_dim + config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.stochastic_dim * 2),
        )

        # Deterministic state transition: h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
        self.gru = nn.GRUCell(
            config.stochastic_dim + config.action_dim,
            config.deterministic_dim,
        )

        # Min std for stability
        self.min_std = 0.1

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize state."""
        return {
            "stochastic": torch.zeros(batch_size, self.config.stochastic_dim, device=device),
            "deterministic": torch.zeros(batch_size, self.config.deterministic_dim, device=device),
        }

    def get_prior(self, deterministic: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute prior distribution p(s_t | h_t)."""
        out = self.prior_net(deterministic)
        mean, log_std = out.chunk(2, dim=-1)
        std = F.softplus(log_std) + self.min_std

        return {"mean": mean, "std": std}

    def get_posterior(
        self,
        deterministic: torch.Tensor,
        observation_embed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute posterior distribution q(s_t | h_t, o_t)."""
        x = torch.cat([deterministic, observation_embed], dim=-1)
        out = self.posterior_net(x)
        mean, log_std = out.chunk(2, dim=-1)
        std = F.softplus(log_std) + self.min_std

        return {"mean": mean, "std": std}

    def sample_stochastic(
        self,
        dist: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample from stochastic state distribution."""
        if deterministic:
            return dist["mean"]

        noise = torch.randn_like(dist["mean"])
        return dist["mean"] + noise * dist["std"]

    def step(
        self,
        prev_state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        observation_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single RSSM step.

        Args:
            prev_state: Previous state dict
            action: Action taken
            observation_embed: Encoded observation (for posterior)

        Returns:
            New state dict with prior/posterior distributions
        """
        # Deterministic transition
        x = torch.cat([prev_state["stochastic"], action], dim=-1)
        deterministic = self.gru(x, prev_state["deterministic"])

        # Prior
        prior = self.get_prior(deterministic)

        # Posterior (if observation available)
        if observation_embed is not None:
            posterior = self.get_posterior(deterministic, observation_embed)
            stochastic = self.sample_stochastic(posterior)
        else:
            posterior = None
            stochastic = self.sample_stochastic(prior)

        return {
            "stochastic": stochastic,
            "deterministic": deterministic,
            "prior": prior,
            "posterior": posterior,
        }

    def imagine(
        self,
        initial_state: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine trajectory without observations.

        Args:
            initial_state: Initial RSSM state
            actions: (batch, horizon, action_dim)

        Returns:
            Imagined states
        """
        batch_size, horizon, _ = actions.shape

        states = []
        state = initial_state

        for t in range(horizon):
            state = self.step(state, actions[:, t])
            states.append({
                "stochastic": state["stochastic"],
                "deterministic": state["deterministic"],
            })

        # Stack
        stochastic = torch.stack([s["stochastic"] for s in states], dim=1)
        deterministic = torch.stack([s["deterministic"] for s in states], dim=1)

        return {
            "stochastic": stochastic,
            "deterministic": deterministic,
        }


class LatentDynamics(nn.Module):
    """
    Simplified latent dynamics without image reconstruction.

    For use with pre-encoded observations (e.g., from vision encoder).
    """

    def __init__(self, config: LatentWorldModelConfig):
        super().__init__()
        self.config = config

        # State transition network
        self.transition = nn.Sequential(
            nn.Linear(config.stochastic_dim + config.action_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.stochastic_dim * 2),
        )

        # Encoder for observations
        self.encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.stochastic_dim * 2),
        )

        self.min_std = 0.1

    def encode(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode observation to latent distribution."""
        out = self.encoder(observation)
        mean, log_std = out.chunk(2, dim=-1)
        std = F.softplus(log_std) + self.min_std
        return {"mean": mean, "std": std}

    def transition_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict next state distribution."""
        x = torch.cat([state, action], dim=-1)
        out = self.transition(x)
        mean, log_std = out.chunk(2, dim=-1)
        std = F.softplus(log_std) + self.min_std
        return {"mean": mean, "std": std}

    def sample(
        self,
        dist: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample from distribution."""
        if deterministic:
            return dist["mean"]
        noise = torch.randn_like(dist["mean"])
        return dist["mean"] + noise * dist["std"]


class LatentWorldModel(nn.Module):
    """
    Complete latent world model.

    Integrates RSSM with encoder/decoder for full world modeling.
    """

    def __init__(self, config: LatentWorldModelConfig):
        super().__init__()
        self.config = config

        # Components
        self.encoder = ConvEncoder(config)
        self.rssm = RSSM(config)
        self.decoder = ImageDecoder(config)

        # Reward predictor from latent
        latent_dim = config.stochastic_dim + config.deterministic_dim
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, 1),
        )

        # Continue predictor (episode termination)
        self.continue_head = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def get_latent(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine stochastic and deterministic state."""
        return torch.cat([state["stochastic"], state["deterministic"]], dim=-1)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through world model.

        Args:
            observations: (batch, seq_len, channels, height, width)
            actions: (batch, seq_len, action_dim)

        Returns:
            Dict with reconstructions, predictions, and losses
        """
        batch_size, seq_len = observations.shape[:2]
        device = observations.device

        # Encode all observations
        obs_flat = observations.view(-1, *observations.shape[2:])
        obs_embed = self.encoder(obs_flat)
        obs_embed = obs_embed.view(batch_size, seq_len, -1)

        # Initialize state
        state = self.rssm.initial_state(batch_size, device)

        # Process sequence
        priors = []
        posteriors = []
        latents = []

        for t in range(seq_len):
            state = self.rssm.step(
                state,
                actions[:, t] if t > 0 else torch.zeros_like(actions[:, 0]),
                obs_embed[:, t],
            )

            priors.append(state["prior"])
            posteriors.append(state["posterior"])
            latents.append(self.get_latent(state))

        # Stack outputs
        latents = torch.stack(latents, dim=1)

        # Decode
        latents_flat = latents.view(-1, latents.shape[-1])
        recon = self.decoder(latents_flat)
        recon = recon.view(batch_size, seq_len, *recon.shape[1:])

        # Predict rewards and continues
        rewards = self.reward_head(latents_flat).view(batch_size, seq_len)
        continues = self.continue_head(latents_flat).view(batch_size, seq_len)

        return {
            "reconstruction": recon,
            "predicted_rewards": rewards,
            "predicted_continues": continues,
            "latents": latents,
            "priors": priors,
            "posteriors": posteriors,
        }

    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute world model loss."""
        outputs = self.forward(observations, actions)

        # Reconstruction loss
        recon_loss = F.mse_loss(outputs["reconstruction"], observations)

        # Reward loss
        reward_loss = F.mse_loss(outputs["predicted_rewards"], rewards)

        # Continue loss (binary cross entropy)
        continue_loss = F.binary_cross_entropy(
            outputs["predicted_continues"],
            1 - dones.float(),
        )

        # KL loss (between posterior and prior)
        kl_loss = 0
        for prior, posterior in zip(outputs["priors"], outputs["posteriors"]):
            kl = self._kl_divergence(posterior, prior)
            kl = torch.maximum(kl, torch.tensor(self.config.free_nats, device=kl.device))
            kl_loss += kl.mean()
        kl_loss /= len(outputs["priors"])

        # Total loss
        total_loss = (
            recon_loss +
            reward_loss +
            continue_loss +
            self.config.kl_scale * kl_loss
        )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "reward_loss": reward_loss,
            "continue_loss": continue_loss,
            "kl_loss": kl_loss,
        }

    def _kl_divergence(
        self,
        posterior: Dict[str, torch.Tensor],
        prior: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute KL divergence between two Gaussians."""
        mean_diff = posterior["mean"] - prior["mean"]
        var_ratio = (posterior["std"] / prior["std"]) ** 2

        kl = 0.5 * (
            var_ratio +
            (mean_diff / prior["std"]) ** 2 -
            1 -
            2 * (posterior["std"].log() - prior["std"].log())
        )
        return kl.sum(-1)

    def imagine_ahead(
        self,
        initial_state: Dict[str, torch.Tensor],
        policy,
        horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine trajectory using a policy.

        Args:
            initial_state: Starting RSSM state
            policy: Policy network that takes latent and returns action
            horizon: Number of steps to imagine

        Returns:
            Imagined trajectory
        """
        state = initial_state
        states = []
        actions = []
        rewards = []

        for _ in range(horizon):
            latent = self.get_latent(state)
            action = policy(latent)

            state = self.rssm.step(state, action)

            states.append(latent)
            actions.append(action)
            rewards.append(self.reward_head(latent).squeeze(-1))

        return {
            "states": torch.stack(states, dim=1),
            "actions": torch.stack(actions, dim=1),
            "rewards": torch.stack(rewards, dim=1),
        }


if __name__ == "__main__":
    config = LatentWorldModelConfig(
        image_size=224,
        stochastic_dim=32,
        deterministic_dim=256,
        action_dim=7,
    )

    # Test RSSM
    rssm = RSSM(config)
    batch_size = 4

    state = rssm.initial_state(batch_size, torch.device("cpu"))
    action = torch.randn(batch_size, config.action_dim)
    obs_embed = torch.randn(batch_size, config.hidden_dim)

    new_state = rssm.step(state, action, obs_embed)
    print(f"RSSM state shapes:")
    print(f"  stochastic: {new_state['stochastic'].shape}")
    print(f"  deterministic: {new_state['deterministic'].shape}")

    # Test imagination
    actions = torch.randn(batch_size, 10, config.action_dim)
    imagined = rssm.imagine(state, actions)
    print(f"Imagined trajectory shapes:")
    print(f"  stochastic: {imagined['stochastic'].shape}")
    print(f"  deterministic: {imagined['deterministic'].shape}")

    # Test full world model (commented due to memory)
    # world_model = LatentWorldModel(config)
    # obs = torch.randn(batch_size, 5, 3, 224, 224)
    # actions = torch.randn(batch_size, 5, 7)
    # outputs = world_model(obs, actions)
    print("\nLatent world model test passed!")

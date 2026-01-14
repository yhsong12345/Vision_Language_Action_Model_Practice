"""
History Encoder Module

Encodes observation and action history for informed decision making.
Provides context about past states and actions to improve current policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class HistoryEncoderConfig:
    """Configuration for history encoder."""
    obs_dim: int = 768
    action_dim: int = 7
    hidden_dim: int = 512
    output_dim: int = 768
    history_length: int = 16
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    use_action_embedding: bool = True
    use_timestep_embedding: bool = True


class ActionHistoryEncoder(nn.Module):
    """
    Encodes a sequence of past actions.

    Useful for action consistency and avoiding oscillation.
    """

    def __init__(self, config: HistoryEncoderConfig):
        super().__init__()
        self.config = config

        # Action embedding
        self.action_embed = nn.Linear(config.action_dim, config.hidden_dim)

        # Positional embedding for temporal order
        self.pos_embed = nn.Embedding(config.history_length, config.hidden_dim)

        # Transformer for action sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

        # Action history buffer
        self.history = deque(maxlen=config.history_length)

    def forward(
        self,
        actions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode action sequence.

        Args:
            actions: (batch_size, seq_len, action_dim)
            attention_mask: (batch_size, seq_len)

        Returns:
            encoded: (batch_size, output_dim) aggregated action history
        """
        batch_size, seq_len, _ = actions.shape

        # Embed actions
        x = self.action_embed(actions)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=actions.device)
        x = x + self.pos_embed(positions)

        # Create mask
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Encode
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Aggregate (use last or mean)
        if attention_mask is not None:
            # Masked mean
            x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        else:
            x = x.mean(1)

        return self.output_proj(x)

    def add_action(self, action: torch.Tensor):
        """Add an action to history buffer."""
        self.history.append(action.detach().cpu())

    def get_history_tensor(self, device: torch.device) -> torch.Tensor:
        """Get history as tensor for encoding."""
        if len(self.history) == 0:
            return None

        history = torch.stack(list(self.history), dim=0)
        return history.unsqueeze(0).to(device)

    def reset(self):
        """Clear action history."""
        self.history.clear()


class StateHistoryEncoder(nn.Module):
    """
    Encodes a sequence of past observations/states.

    Provides temporal context for current decision making.
    """

    def __init__(self, config: HistoryEncoderConfig):
        super().__init__()
        self.config = config

        # State projection
        self.state_embed = nn.Linear(config.obs_dim, config.hidden_dim)

        # Positional embedding
        self.pos_embed = nn.Embedding(config.history_length, config.hidden_dim)

        # Timestep embedding (optional, for variable timing)
        if config.use_timestep_embedding:
            self.time_embed = nn.Sequential(
                nn.Linear(1, config.hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(config.hidden_dim // 4, config.hidden_dim),
            )

        # Causal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

        # State history buffer
        self.history = deque(maxlen=config.history_length)

    def forward(
        self,
        states: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode state sequence.

        Args:
            states: (batch_size, seq_len, obs_dim)
            timesteps: (batch_size, seq_len) optional timestep indices
            attention_mask: (batch_size, seq_len)

        Returns:
            Dict with sequence and aggregated encodings
        """
        batch_size, seq_len, _ = states.shape

        # Embed states
        x = self.state_embed(states)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=states.device)
        x = x + self.pos_embed(positions)

        # Add timestep embeddings if provided
        if timesteps is not None and self.config.use_timestep_embedding:
            t_embed = self.time_embed(timesteps.unsqueeze(-1).float())
            x = x + t_embed

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=states.device
        )

        # Attention mask
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Encode
        sequence = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Get last hidden state
        last_hidden = sequence[:, -1]

        # Project outputs
        sequence = self.output_proj(sequence)
        aggregated = self.output_proj(last_hidden)

        return {
            "sequence": sequence,
            "last_hidden": aggregated,
        }

    def add_state(self, state: torch.Tensor):
        """Add a state to history buffer."""
        self.history.append(state.detach().cpu())

    def get_history_tensor(self, device: torch.device) -> torch.Tensor:
        """Get history as tensor for encoding."""
        if len(self.history) == 0:
            return None

        history = torch.stack(list(self.history), dim=0)
        return history.unsqueeze(0).to(device)

    def reset(self):
        """Clear state history."""
        self.history.clear()


class HistoryEncoder(nn.Module):
    """
    Combined history encoder for both states and actions.

    Provides a unified interface for encoding full interaction history.
    """

    def __init__(self, config: HistoryEncoderConfig):
        super().__init__()
        self.config = config

        # State encoder
        self.state_encoder = StateHistoryEncoder(config)

        # Action encoder
        self.action_encoder = ActionHistoryEncoder(config)

        # Fusion module
        self.fusion = nn.Sequential(
            nn.Linear(config.output_dim * 2, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

        # Cross-attention between states and actions
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.output_dim,
            num_heads=config.num_heads,
            batch_first=True,
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        state_mask: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode state-action history.

        Args:
            states: (batch_size, seq_len, obs_dim)
            actions: (batch_size, seq_len, action_dim) - actions taken after each state
            state_mask: (batch_size, seq_len)
            action_mask: (batch_size, seq_len)

        Returns:
            Dict with various history encodings
        """
        # Encode separately
        state_out = self.state_encoder(states, attention_mask=state_mask)
        action_out = self.action_encoder(actions, attention_mask=action_mask)

        state_features = state_out["last_hidden"]
        action_features = action_out

        # Simple fusion
        fused = self.fusion(torch.cat([state_features, action_features], dim=-1))

        # Cross-attention fusion
        state_seq = state_out["sequence"]
        action_seq_expanded = action_out.unsqueeze(1).expand(-1, state_seq.size(1), -1)

        cross_attended, _ = self.cross_attention(
            query=state_seq,
            key=action_seq_expanded,
            value=action_seq_expanded,
        )

        return {
            "state_encoding": state_features,
            "action_encoding": action_features,
            "fused": fused,
            "cross_attended": cross_attended,
            "state_sequence": state_seq,
        }

    def add_transition(self, state: torch.Tensor, action: torch.Tensor):
        """Add state-action pair to history."""
        self.state_encoder.add_state(state)
        self.action_encoder.add_action(action)

    def get_history(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get state and action history tensors."""
        states = self.state_encoder.get_history_tensor(device)
        actions = self.action_encoder.get_history_tensor(device)
        return states, actions

    def reset(self):
        """Clear all history."""
        self.state_encoder.reset()
        self.action_encoder.reset()


class HistoryAwarePolicy(nn.Module):
    """
    Wrapper that adds history awareness to any policy.

    Takes a base policy and augments its input with history context.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        obs_dim: int,
        action_dim: int,
        history_dim: int = 512,
        history_length: int = 16,
    ):
        super().__init__()
        self.base_policy = base_policy

        # History encoder
        config = HistoryEncoderConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=history_dim,
            output_dim=history_dim,
            history_length=history_length,
        )
        self.history_encoder = HistoryEncoder(config)

        # Projection to augment policy input
        self.context_proj = nn.Linear(history_dim, obs_dim)

        # Gate for combining current obs with history
        self.gate = nn.Sequential(
            nn.Linear(obs_dim * 2, obs_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        observation: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with history context.

        Args:
            observation: Current observation
            states: History of past states (optional, uses buffer if None)
            actions: History of past actions (optional, uses buffer if None)
        """
        # Get history context
        if states is not None and actions is not None:
            history_out = self.history_encoder(states, actions)
            context = self.context_proj(history_out["fused"])
        else:
            # Use internal buffers
            hist_states, hist_actions = self.history_encoder.get_history(
                observation.device
            )
            if hist_states is not None and hist_actions is not None:
                history_out = self.history_encoder(hist_states, hist_actions)
                context = self.context_proj(history_out["fused"])
            else:
                context = torch.zeros_like(observation)

        # Gate combination
        combined = torch.cat([observation, context], dim=-1)
        gate = self.gate(combined)
        augmented_obs = gate * observation + (1 - gate) * context

        # Run base policy
        return self.base_policy(augmented_obs)

    def update_history(self, state: torch.Tensor, action: torch.Tensor):
        """Update internal history buffers."""
        self.history_encoder.add_transition(state, action)

    def reset_history(self):
        """Clear history buffers."""
        self.history_encoder.reset()


if __name__ == "__main__":
    # Test history encoders
    config = HistoryEncoderConfig(
        obs_dim=768,
        action_dim=7,
        hidden_dim=512,
        output_dim=768,
        history_length=16,
    )

    # Test action history encoder
    action_encoder = ActionHistoryEncoder(config)
    actions = torch.randn(4, 16, 7)  # batch=4, seq_len=16, action_dim=7
    encoded = action_encoder(actions)
    print(f"Action history encoding shape: {encoded.shape}")

    # Test state history encoder
    state_encoder = StateHistoryEncoder(config)
    states = torch.randn(4, 16, 768)
    outputs = state_encoder(states)
    print(f"State history outputs:")
    print(f"  sequence: {outputs['sequence'].shape}")
    print(f"  last_hidden: {outputs['last_hidden'].shape}")

    # Test combined history encoder
    history_encoder = HistoryEncoder(config)
    outputs = history_encoder(states, actions)
    print(f"Combined history outputs:")
    print(f"  state_encoding: {outputs['state_encoding'].shape}")
    print(f"  action_encoding: {outputs['action_encoding'].shape}")
    print(f"  fused: {outputs['fused'].shape}")

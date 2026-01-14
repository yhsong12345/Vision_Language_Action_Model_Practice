"""
Temporal Encoder Module

Encodes sequences of observations over time for temporal reasoning.
Supports multiple architectures:
- Transformer-based temporal encoding
- LSTM-based temporal encoding
- Causal attention for autoregressive modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class TemporalEncoderConfig:
    """Configuration for temporal encoder."""
    input_dim: int = 768
    hidden_dim: int = 512
    output_dim: int = 768
    num_layers: int = 4
    num_heads: int = 8
    max_seq_len: int = 64
    dropout: float = 0.1
    use_causal_mask: bool = True
    encoder_type: str = "transformer"  # "transformer" or "lstm"
    use_positional_encoding: bool = True
    use_learned_pos_embed: bool = True


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer-based temporal encoder.

    Uses self-attention to capture temporal dependencies across observations.
    Supports causal masking for autoregressive generation.
    """

    def __init__(self, config: TemporalEncoderConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # Positional encoding
        if config.use_positional_encoding:
            if config.use_learned_pos_embed:
                self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_dim)
            else:
                self.pos_encoding = PositionalEncoding(
                    config.hidden_dim, config.max_seq_len, config.dropout
                )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.layer_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) - sequence of observations
            attention_mask: (batch_size, seq_len) - mask for padding

        Returns:
            output: (batch_size, seq_len, output_dim) - temporally encoded features
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

        # Add positional encoding
        if self.config.use_positional_encoding:
            if self.config.use_learned_pos_embed:
                positions = torch.arange(seq_len, device=x.device)
                x = x + self.pos_embed(positions)
            else:
                x = self.pos_encoding(x)

        # Create causal mask if needed
        if self.config.use_causal_mask:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=x.device
            )
        else:
            causal_mask = None

        # Create padding mask
        if attention_mask is not None:
            # Convert to transformer format (True = masked)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Transformer encoding
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Output projection
        x = self.output_proj(x)
        x = self.layer_norm(x)

        return x

    def get_last_hidden_state(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the last valid hidden state for each sequence."""
        output = self.forward(x, attention_mask)

        if attention_mask is not None:
            # Get index of last valid token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(output.size(0), device=output.device)
            return output[batch_indices, seq_lengths]
        else:
            return output[:, -1]


class TemporalLSTM(nn.Module):
    """
    LSTM-based temporal encoder.

    More efficient for very long sequences and provides
    explicit temporal state management.
    """

    def __init__(self, config: TemporalEncoderConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=False,
        )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.layer_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            hidden_state: Optional (h_n, c_n) from previous step

        Returns:
            output: (batch_size, seq_len, output_dim)
            hidden_state: (h_n, c_n) for next step
        """
        x = self.input_proj(x)
        x, hidden_state = self.lstm(x, hidden_state)
        x = self.output_proj(x)
        x = self.layer_norm(x)

        return x, hidden_state

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state for a new sequence."""
        h_0 = torch.zeros(
            self.config.num_layers, batch_size, self.config.hidden_dim,
            device=device
        )
        c_0 = torch.zeros(
            self.config.num_layers, batch_size, self.config.hidden_dim,
            device=device
        )
        return (h_0, c_0)


class TemporalEncoder(nn.Module):
    """
    Unified temporal encoder interface.

    Wraps different temporal encoding architectures (Transformer, LSTM)
    with a consistent interface for VLA integration.
    """

    def __init__(self, config: TemporalEncoderConfig):
        super().__init__()
        self.config = config

        if config.encoder_type == "transformer":
            self.encoder = TemporalTransformer(config)
        elif config.encoder_type == "lstm":
            self.encoder = TemporalLSTM(config)
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")

        # State aggregation for producing single vector
        self.aggregation = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

    def forward(
        self,
        observations: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a sequence of observations.

        Args:
            observations: (batch_size, seq_len, input_dim)
            attention_mask: (batch_size, seq_len)
            return_sequence: If True, return full sequence; else return aggregated

        Returns:
            Dict containing:
                - hidden_states: Full sequence or aggregated representation
                - last_hidden_state: Last timestep hidden state
        """
        if self.config.encoder_type == "transformer":
            hidden_states = self.encoder(observations, attention_mask)
            last_hidden_state = self.encoder.get_last_hidden_state(
                observations, attention_mask
            )
        else:
            hidden_states, _ = self.encoder(observations)
            last_hidden_state = hidden_states[:, -1]

        # Aggregate if needed
        aggregated = self.aggregation(last_hidden_state)

        return {
            "hidden_states": hidden_states if return_sequence else aggregated,
            "last_hidden_state": last_hidden_state,
            "aggregated": aggregated,
        }

    @classmethod
    def from_pretrained(cls, path: str) -> "TemporalEncoder":
        """Load a pretrained temporal encoder."""
        checkpoint = torch.load(path, map_location="cpu")
        config = TemporalEncoderConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def save_pretrained(self, path: str):
        """Save the temporal encoder."""
        torch.save({
            "config": vars(self.config),
            "state_dict": self.state_dict(),
        }, path)


if __name__ == "__main__":
    # Test temporal encoders
    config = TemporalEncoderConfig(
        input_dim=768,
        hidden_dim=512,
        output_dim=768,
        num_layers=4,
        max_seq_len=64,
    )

    # Test Transformer encoder
    transformer_encoder = TemporalTransformer(config)
    x = torch.randn(4, 32, 768)  # batch=4, seq_len=32
    mask = torch.ones(4, 32)

    output = transformer_encoder(x, mask)
    print(f"Transformer output shape: {output.shape}")

    # Test LSTM encoder
    config.encoder_type = "lstm"
    lstm_encoder = TemporalLSTM(config)
    output, hidden = lstm_encoder(x)
    print(f"LSTM output shape: {output.shape}")

    # Test unified encoder
    config.encoder_type = "transformer"
    encoder = TemporalEncoder(config)
    outputs = encoder(x, mask)
    print(f"Unified encoder output shapes:")
    print(f"  hidden_states: {outputs['hidden_states'].shape}")
    print(f"  last_hidden_state: {outputs['last_hidden_state'].shape}")
    print(f"  aggregated: {outputs['aggregated'].shape}")

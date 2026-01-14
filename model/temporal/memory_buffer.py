"""
Memory Buffer Module

Provides memory structures for long-horizon reasoning:
- EpisodicMemory: Stores complete episodes for experience replay
- WorkingMemory: Short-term memory for current task context
- HierarchicalMemory: Multi-level memory with different time scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import deque


@dataclass
class MemoryConfig:
    """Configuration for memory modules."""
    hidden_dim: int = 512
    memory_size: int = 1000  # Number of memory slots
    key_dim: int = 64
    value_dim: int = 512
    num_heads: int = 8
    num_write_heads: int = 1
    num_read_heads: int = 4
    use_temporal_linking: bool = True


class MemoryBuffer(nn.Module):
    """
    Base memory buffer with key-value storage.

    Implements a differentiable memory with content-based addressing.
    Similar to Neural Turing Machine memory.
    """

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config

        # Memory matrix: (memory_size, value_dim)
        self.register_buffer(
            "memory",
            torch.zeros(config.memory_size, config.value_dim)
        )

        # Key matrix for content addressing: (memory_size, key_dim)
        self.register_buffer(
            "keys",
            torch.zeros(config.memory_size, config.key_dim)
        )

        # Usage vector for memory allocation
        self.register_buffer(
            "usage",
            torch.zeros(config.memory_size)
        )

        # Write pointer
        self.write_ptr = 0

        # Query/Key/Value projections
        self.query_proj = nn.Linear(config.hidden_dim, config.key_dim * config.num_read_heads)
        self.key_proj = nn.Linear(config.hidden_dim, config.key_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.value_dim)

        # Output projection
        self.output_proj = nn.Linear(
            config.value_dim * config.num_read_heads, config.hidden_dim
        )

    def write(self, content: torch.Tensor, key: Optional[torch.Tensor] = None):
        """
        Write content to memory.

        Args:
            content: (hidden_dim,) content to write
            key: Optional key for the content
        """
        if key is None:
            key = self.key_proj(content)
        value = self.value_proj(content)

        # Write to current position
        self.memory[self.write_ptr] = value.detach()
        self.keys[self.write_ptr] = key.detach()
        self.usage[self.write_ptr] = 1.0

        # Update pointer
        self.write_ptr = (self.write_ptr + 1) % self.config.memory_size

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using content-based addressing.

        Args:
            query: (batch_size, hidden_dim) query vector

        Returns:
            retrieved: (batch_size, hidden_dim) retrieved content
        """
        batch_size = query.size(0)

        # Project query
        queries = self.query_proj(query)  # (batch_size, key_dim * num_heads)
        queries = queries.view(batch_size, self.config.num_read_heads, self.config.key_dim)

        # Compute attention weights
        # (batch_size, num_heads, memory_size)
        attention = torch.einsum("bhk,mk->bhm", queries, self.keys)
        attention = attention / (self.config.key_dim ** 0.5)

        # Mask unused memory slots
        usage_mask = (self.usage > 0).float()
        attention = attention.masked_fill(usage_mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)
        attention = F.softmax(attention, dim=-1)

        # Read from memory
        # (batch_size, num_heads, value_dim)
        read_values = torch.einsum("bhm,mv->bhv", attention, self.memory)

        # Concatenate heads and project
        read_values = read_values.view(batch_size, -1)
        output = self.output_proj(read_values)

        return output

    def reset(self):
        """Reset memory to initial state."""
        self.memory.zero_()
        self.keys.zero_()
        self.usage.zero_()
        self.write_ptr = 0


class EpisodicMemory(nn.Module):
    """
    Episodic memory for storing and retrieving complete experiences.

    Stores (observation, action, reward, next_observation) tuples
    with temporal linking for episode-level retrieval.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_episodes: int = 100,
        max_episode_length: int = 1000,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_episodes = max_episodes
        self.max_episode_length = max_episode_length
        self.hidden_dim = hidden_dim

        # Episode storage (CPU for large storage)
        self.episodes: List[Dict[str, torch.Tensor]] = []
        self.current_episode: Dict[str, List[torch.Tensor]] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        # Episode encoder for retrieval
        self.episode_encoder = nn.GRU(
            input_size=obs_dim + action_dim + 1,  # obs + action + reward
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Query network for episode retrieval
        self.query_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def add_transition(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
    ):
        """Add a transition to the current episode."""
        self.current_episode["observations"].append(obs.cpu())
        self.current_episode["actions"].append(action.cpu())
        self.current_episode["rewards"].append(torch.tensor([reward]))
        self.current_episode["dones"].append(torch.tensor([done]))

        if done:
            self._store_episode()

    def _store_episode(self):
        """Store completed episode and reset current."""
        if len(self.current_episode["observations"]) > 0:
            episode = {
                "observations": torch.stack(self.current_episode["observations"]),
                "actions": torch.stack(self.current_episode["actions"]),
                "rewards": torch.stack(self.current_episode["rewards"]),
                "dones": torch.stack(self.current_episode["dones"]),
            }

            # Encode episode
            with torch.no_grad():
                episode["embedding"] = self._encode_episode(episode)

            self.episodes.append(episode)

            # Remove oldest if at capacity
            if len(self.episodes) > self.max_episodes:
                self.episodes.pop(0)

        # Reset current episode
        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

    def _encode_episode(self, episode: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode an episode into a fixed-size vector."""
        obs = episode["observations"]
        actions = episode["actions"]
        rewards = episode["rewards"]

        # Concatenate features
        features = torch.cat([obs, actions, rewards], dim=-1)
        features = features.unsqueeze(0)  # Add batch dim

        # Encode with GRU
        _, hidden = self.episode_encoder(features)
        return hidden[-1].squeeze(0)  # Last layer hidden state

    def retrieve_similar(
        self,
        query_obs: torch.Tensor,
        k: int = 5,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Retrieve k most similar episodes based on query observation.

        Args:
            query_obs: Current observation to query with
            k: Number of episodes to retrieve

        Returns:
            List of similar episodes
        """
        if len(self.episodes) == 0:
            return []

        # Encode query
        query = self.query_net(query_obs)

        # Compute similarities
        similarities = []
        for episode in self.episodes:
            sim = F.cosine_similarity(
                query.unsqueeze(0),
                episode["embedding"].unsqueeze(0),
            )
            similarities.append(sim.item())

        # Get top-k indices
        indices = np.argsort(similarities)[-k:][::-1]

        return [self.episodes[i] for i in indices]

    def get_all_transitions(self) -> Dict[str, torch.Tensor]:
        """Get all stored transitions for training."""
        if len(self.episodes) == 0:
            return None

        all_obs = torch.cat([ep["observations"] for ep in self.episodes])
        all_actions = torch.cat([ep["actions"] for ep in self.episodes])
        all_rewards = torch.cat([ep["rewards"] for ep in self.episodes])
        all_dones = torch.cat([ep["dones"] for ep in self.episodes])

        return {
            "observations": all_obs,
            "actions": all_actions,
            "rewards": all_rewards,
            "dones": all_dones,
        }


class WorkingMemory(nn.Module):
    """
    Working memory for current task context.

    Maintains a fixed-size buffer of recent observations/states
    with attention-based retrieval for decision making.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        memory_slots: int = 32,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots

        # Memory buffer
        self.memory = deque(maxlen=memory_slots)

        # Attention for reading
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Memory update gate
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def update(self, new_state: torch.Tensor):
        """Add new state to working memory."""
        self.memory.append(new_state.detach().cpu())

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Read from working memory using attention.

        Args:
            query: (batch_size, hidden_dim) query vector

        Returns:
            context: (batch_size, hidden_dim) attended memory content
        """
        if len(self.memory) == 0:
            return torch.zeros_like(query)

        # Stack memory into tensor
        memory_tensor = torch.stack(list(self.memory), dim=0)
        memory_tensor = memory_tensor.to(query.device)

        # Expand for batch
        batch_size = query.size(0)
        memory_tensor = memory_tensor.unsqueeze(0).expand(batch_size, -1, -1)

        # Query attention
        query = query.unsqueeze(1)  # (batch, 1, hidden)
        context, _ = self.attention(query, memory_tensor, memory_tensor)
        context = context.squeeze(1)

        return context

    def get_gated_update(
        self,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get gated combination of current state and memory context.

        Args:
            current_state: (batch_size, hidden_dim)

        Returns:
            updated_state: (batch_size, hidden_dim)
        """
        context = self.read(current_state)

        combined = torch.cat([current_state, context], dim=-1)
        gate = self.update_gate(combined)

        updated = gate * current_state + (1 - gate) * context
        return updated

    def reset(self):
        """Clear working memory."""
        self.memory.clear()


class HierarchicalMemory(nn.Module):
    """
    Hierarchical memory with multiple time scales.

    Combines short-term working memory with long-term episodic memory
    for handling tasks at different temporal scales.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        short_term_slots: int = 32,
        long_term_slots: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Short-term memory (recent observations)
        self.short_term = WorkingMemory(
            hidden_dim=hidden_dim,
            memory_slots=short_term_slots,
            num_heads=num_heads,
        )

        # Long-term memory (compressed/important states)
        self.long_term = MemoryBuffer(
            MemoryConfig(
                hidden_dim=hidden_dim,
                memory_size=long_term_slots,
                key_dim=64,
                value_dim=hidden_dim,
                num_heads=num_heads,
            )
        )

        # Importance scorer for memory consolidation
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Consolidation threshold
        self.consolidation_threshold = 0.7

    def update(self, state: torch.Tensor):
        """
        Update memory with new state.

        Important states are consolidated to long-term memory.
        """
        # Always add to short-term
        self.short_term.update(state)

        # Check importance for long-term storage
        importance = self.importance_scorer(state)
        if importance.mean() > self.consolidation_threshold:
            self.long_term.write(state)

    def read(self, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Read from both memory levels.

        Args:
            query: (batch_size, hidden_dim)

        Returns:
            Dict with short_term, long_term, and fused memories
        """
        short_term_context = self.short_term.read(query)
        long_term_context = self.long_term.read(query)

        # Fuse memories
        combined = torch.cat([short_term_context, long_term_context], dim=-1)
        fused = self.fusion(combined)

        return {
            "short_term": short_term_context,
            "long_term": long_term_context,
            "fused": fused,
        }

    def reset(self, reset_long_term: bool = False):
        """Reset memories."""
        self.short_term.reset()
        if reset_long_term:
            self.long_term.reset()


if __name__ == "__main__":
    # Test memory modules
    config = MemoryConfig(hidden_dim=512, memory_size=100)

    # Test basic memory buffer
    memory = MemoryBuffer(config)

    # Write some content
    for i in range(10):
        content = torch.randn(512)
        memory.write(content)

    # Read
    query = torch.randn(4, 512)
    retrieved = memory.read(query)
    print(f"Memory buffer read shape: {retrieved.shape}")

    # Test working memory
    working_mem = WorkingMemory(hidden_dim=512, memory_slots=32)
    for i in range(20):
        working_mem.update(torch.randn(512))

    context = working_mem.read(torch.randn(4, 512))
    print(f"Working memory read shape: {context.shape}")

    # Test hierarchical memory
    hier_mem = HierarchicalMemory(hidden_dim=512)
    for i in range(50):
        hier_mem.update(torch.randn(512))

    outputs = hier_mem.read(torch.randn(4, 512))
    print(f"Hierarchical memory output shapes:")
    print(f"  short_term: {outputs['short_term'].shape}")
    print(f"  long_term: {outputs['long_term'].shape}")
    print(f"  fused: {outputs['fused'].shape}")

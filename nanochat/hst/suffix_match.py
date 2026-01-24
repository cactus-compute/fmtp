"""
Context Suffix Matching for HST speculation.

A zero-parameter lookup mechanism for capturing in-context repetition patterns
(variable names, repeated phrases, code structures).

Key components:
- ContextBuffer: Rolling buffer of recent tokens
- SuffixMatcher: Efficient suffix matching with frequency-weighted results

Mechanism:
1. Maintain a rolling buffer of the last N tokens (512-2048)
2. Build suffix index over this buffer for efficient lookups
3. At inference, find all positions where current suffix appeared earlier
4. Return tokens that followed those positions, weighted by recency

Use cases:
- Code generation: variable names, function signatures, repeated patterns
- Repetitive text: boilerplate, templates, structured content
- Complements learned retrieval which captures general patterns
"""

import torch
from typing import Optional
from collections import defaultdict


class ContextBuffer:
    """
    Rolling buffer of recent token IDs for suffix matching.

    Maintains a fixed-size circular buffer with efficient append and
    provides access to recent context for suffix matching.
    """

    def __init__(
        self,
        max_size: int = 1024,
        device: str = "cpu",
    ):
        """
        Args:
            max_size: Maximum number of tokens to store
            device: Device for tensor storage
        """
        self.max_size = max_size
        self.device = device

        # Circular buffer storage
        self.buffer = torch.zeros(max_size, dtype=torch.long, device=device)
        self.size = 0  # Current number of valid tokens
        self.write_pos = 0  # Next write position

    def append(self, token_ids: torch.Tensor | list[int]) -> None:
        """
        Append token(s) to the buffer.

        Args:
            token_ids: Token ID(s) to append. Can be:
                - Single int
                - 1D tensor of token IDs
                - List of ints
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        if token_ids.dim() == 0:
            token_ids = token_ids.unsqueeze(0)

        n_tokens = len(token_ids)

        if n_tokens >= self.max_size:
            # If appending more than buffer size, just keep last max_size
            self.buffer.copy_(token_ids[-self.max_size:])
            self.size = self.max_size
            self.write_pos = 0
        else:
            # Handle wrap-around
            for i, token in enumerate(token_ids):
                self.buffer[self.write_pos] = token
                self.write_pos = (self.write_pos + 1) % self.max_size
                if self.size < self.max_size:
                    self.size += 1

    def get_recent(self, n: int) -> torch.Tensor:
        """
        Get the most recent n tokens.

        Args:
            n: Number of tokens to retrieve

        Returns:
            Tensor of shape [min(n, size)] containing recent tokens
        """
        n = min(n, self.size)
        if n == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)

        # Calculate start position (going backwards from write_pos)
        indices = torch.arange(n, device=self.device)
        read_positions = (self.write_pos - n + indices) % self.max_size

        return self.buffer[read_positions]

    def get_all(self) -> torch.Tensor:
        """
        Get all tokens in chronological order.

        Returns:
            Tensor of shape [size] with all buffered tokens
        """
        return self.get_recent(self.size)

    def __len__(self) -> int:
        return self.size

    def reset(self) -> None:
        """Clear the buffer."""
        self.size = 0
        self.write_pos = 0
        self.buffer.zero_()


class SuffixMatcher:
    """
    Efficient suffix matching over a context buffer.

    Finds all positions in the context where a given suffix has appeared,
    and returns the tokens that followed those positions, weighted by recency.

    This implementation maintains a simple list of all tokens and uses
    direct scanning for suffix matching. For typical buffer sizes (512-2048),
    this is fast enough and avoids hash collision issues.
    """

    def __init__(
        self,
        buffer_size: int = 1024,
        min_suffix_len: int = 1,
        max_suffix_len: int = 4,
        recency_decay: float = 0.95,
        device: str = "cpu",
    ):
        """
        Args:
            buffer_size: Size of the underlying context buffer
            min_suffix_len: Minimum suffix length to match (default 1)
            max_suffix_len: Maximum suffix length to index (default 4)
            recency_decay: Exponential decay factor for older matches (0-1)
            device: Device for tensor operations
        """
        self.buffer = ContextBuffer(max_size=buffer_size, device=device)
        self.min_suffix_len = min_suffix_len
        self.max_suffix_len = max_suffix_len
        self.recency_decay = recency_decay
        self.device = device

        # Keep a simple list for direct suffix matching
        self.token_list: list[int] = []

    def append(self, token_ids: torch.Tensor | list[int] | int) -> None:
        """
        Append tokens to context.

        Args:
            token_ids: Token ID(s) to append
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Append to buffer
        self.buffer.append(token_ids)

        # Append to list (maintain max size)
        self.token_list.extend(token_ids)
        if len(self.token_list) > self.buffer.max_size:
            self.token_list = self.token_list[-self.buffer.max_size:]

    def find_continuations(
        self,
        suffix: torch.Tensor,
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find tokens that followed the given suffix in context.

        Args:
            suffix: [n] tensor of token IDs representing the suffix to match
            top_k: Maximum number of continuation candidates to return
            temperature: Temperature for normalizing scores (higher = more uniform)

        Returns:
            tokens: [k] tensor of continuation token IDs
            scores: [k] tensor of recency-weighted scores (sum to 1.0)
        """
        if suffix.dim() == 0:
            suffix = suffix.unsqueeze(0)

        suffix_list = suffix.tolist()
        n = len(suffix_list)

        if n < self.min_suffix_len:
            return (
                torch.tensor([], dtype=torch.long, device=self.device),
                torch.tensor([], dtype=torch.float, device=self.device),
            )

        # Clamp to max suffix length
        if n > self.max_suffix_len:
            suffix_list = suffix_list[-self.max_suffix_len:]
            n = self.max_suffix_len

        tokens = self.token_list
        num_tokens = len(tokens)

        if num_tokens <= n:
            return (
                torch.tensor([], dtype=torch.long, device=self.device),
                torch.tensor([], dtype=torch.float, device=self.device),
            )

        # Find all positions where suffix matches
        # Position i means tokens[i:i+n] matches suffix, and tokens[i+n] is the continuation
        continuations: dict[int, float] = defaultdict(float)

        for i in range(num_tokens - n):
            # Check if suffix matches at position i
            if tokens[i:i+n] == suffix_list:
                # Get the continuation token
                continuation_token = tokens[i + n]

                # Compute recency weight (positions closer to end get higher weight)
                # Distance from end of buffer
                distance = num_tokens - (i + n + 1)
                weight = self.recency_decay ** distance

                continuations[continuation_token] += weight

        if not continuations:
            return (
                torch.tensor([], dtype=torch.long, device=self.device),
                torch.tensor([], dtype=torch.float, device=self.device),
            )

        # Sort by score and take top-k
        sorted_items = sorted(continuations.items(), key=lambda x: -x[1])[:top_k]
        result_tokens = torch.tensor([t for t, _ in sorted_items], dtype=torch.long, device=self.device)
        scores = torch.tensor([s for _, s in sorted_items], dtype=torch.float, device=self.device)

        # Apply temperature and normalize
        scores = scores / temperature
        scores = torch.softmax(scores, dim=0)

        return result_tokens, scores

    def get_suffix_probabilities(
        self,
        suffix: torch.Tensor,
        vocab_size: int,
        smoothing: float = 1e-6,
    ) -> torch.Tensor:
        """
        Get probability distribution over vocabulary for continuations.

        Returns a sparse distribution based on observed continuations,
        with smoothing for unseen tokens.

        Args:
            suffix: [n] tensor representing current suffix
            vocab_size: Size of vocabulary
            smoothing: Smoothing factor for unseen tokens

        Returns:
            probs: [vocab_size] probability distribution
        """
        tokens, scores = self.find_continuations(suffix, top_k=100)

        # Start with uniform smoothing
        probs = torch.full(
            (vocab_size,),
            smoothing,
            dtype=torch.float,
            device=self.device
        )

        if len(tokens) > 0:
            # Add observed continuation probabilities
            # Scale scores to be meaningful relative to smoothing
            probs[tokens] += scores * (1.0 - smoothing * vocab_size)

        # Renormalize
        probs = probs / probs.sum()

        return probs

    def reset(self) -> None:
        """Clear buffer and token list."""
        self.buffer.reset()
        self.token_list = []

    def __len__(self) -> int:
        return len(self.buffer)

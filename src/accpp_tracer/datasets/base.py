"""Base protocol for tracing datasets."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class TracingDataset(Protocol):
    """Minimal interface that all tracing datasets must satisfy.

    Attributes:
        toks: Tokenized input tensor, shape (N, seq_len).
        word_idx: Dict mapping semantic roles to token position tensors.
    """

    toks: torch.Tensor
    word_idx: dict[str, torch.Tensor]

    def __len__(self) -> int: ...

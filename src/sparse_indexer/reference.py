"""Correctness-first reference operations used by the initial test suite.

These functions intentionally use Python lists. They define the semantics that
future Torch/CUDA implementations must match; they are not performance code.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class PageAddress:
    """Physical address of one logical token in a paged KV cache."""

    page_id: int
    offset: int

    def slot_for(self, block_size: int) -> int:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        return self.page_id * block_size + self.offset


def stable_topk(
    scores: Sequence[Sequence[float]],
    k: int,
    valid_lengths: Sequence[int] | None = None,
) -> list[list[int]]:
    """Return row-wise Top-K logical positions with deterministic tie handling.

    Scores beyond ``valid_lengths[row]`` are not candidates. Ties are resolved
    by the smaller logical position, matching the reference contract used by
    the tests and making comparisons reproducible across implementations.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if valid_lengths is not None and len(valid_lengths) != len(scores):
        raise ValueError("valid_lengths must have one entry per score row")

    result: list[list[int]] = []
    for row_id, row in enumerate(scores):
        valid_len = len(row) if valid_lengths is None else valid_lengths[row_id]
        if not 0 <= valid_len <= len(row):
            raise ValueError("valid length must be within the score row")
        for score in row[:valid_len]:
            if not isfinite(float(score)):
                raise ValueError("scores must be finite in the reference path")
        positions = list(range(valid_len))
        positions.sort(key=lambda position: (-float(row[position]), position))
        result.append(positions[:k])
    return result


def map_logical_to_paged(
    logical_positions: Sequence[Sequence[int]],
    page_tables: Sequence[Sequence[int]],
    block_size: int,
    valid_lengths: Sequence[int] | None = None,
) -> list[list[PageAddress]]:
    """Map logical token positions to per-request paged-KV addresses."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if len(logical_positions) != len(page_tables):
        raise ValueError("one page table is required per request")
    if valid_lengths is not None and len(valid_lengths) != len(logical_positions):
        raise ValueError("valid_lengths must have one entry per request")

    mapped: list[list[PageAddress]] = []
    for row_id, positions in enumerate(logical_positions):
        valid_len = None if valid_lengths is None else valid_lengths[row_id]
        row_table = page_tables[row_id]
        row_addresses: list[PageAddress] = []
        for position in positions:
            if position < 0 or (valid_len is not None and position >= valid_len):
                raise ValueError("logical position is outside the valid sequence")
            page_index, offset = divmod(position, block_size)
            if page_index >= len(row_table):
                raise ValueError("page table does not cover the logical position")
            row_addresses.append(PageAddress(row_table[page_index], offset))
        mapped.append(row_addresses)
    return mapped

"""Small, framework-independent reference helpers for sparse-indexer tests."""

from .reference import PageAddress, map_logical_to_paged, stable_topk

__all__ = ["PageAddress", "map_logical_to_paged", "stable_topk"]

#!/usr/bin/env python3
"""
Simple Federated Learning utilities (FedAvg) for MARL agents.

This provides a minimal parameter averaging implementation that can average
numpy arrays or PyTorch state_dicts (if torch is available).
"""
from __future__ import annotations

from typing import Any, List
import numpy as np

try:  # Optional torch support
    import torch
except Exception:
    torch = None


class FedAvgAggregator:
    def __init__(self) -> None:
        self.client_updates: List[Any] = []

    def add_update(self, weights: Any) -> None:
        self.client_updates.append(weights)

    def aggregate(self) -> Any:
        if not self.client_updates:
            return None

        first = self.client_updates[0]
        if (
            torch is not None
            and isinstance(first, dict)
            and all(hasattr(v, "size") for v in first.values())
        ):
            # Assume PyTorch state_dict
            avg_state = {}
            for k in first.keys():
                tensors = [u[k] for u in self.client_updates]
                avg_state[k] = sum(tensors) / float(len(tensors))
            self.client_updates.clear()
            return avg_state

        # Fallback: average list/np arrays
        try:
            stacked = np.stack(self.client_updates)
            avg = np.mean(stacked, axis=0)
            self.client_updates.clear()
            return avg
        except Exception:
            # Return first if cannot aggregate
            result = self.client_updates[0]
            self.client_updates.clear()
            return result

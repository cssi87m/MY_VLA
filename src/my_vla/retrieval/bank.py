"""A dependency-light retrieval bank for demonstration records."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class RetrievalRecord:
    key: np.ndarray
    state: np.ndarray
    instruction: str
    base_action: np.ndarray
    expert_action: np.ndarray
    residual_target: np.ndarray
    future_state: np.ndarray
    return_to_go: float

    @classmethod
    def from_sample(cls, sample: dict[str, Any], key: np.ndarray, base_action: np.ndarray) -> RetrievalRecord:
        expert_action = np.asarray(sample["expert_action"], dtype=np.float32)
        base_action = np.asarray(base_action, dtype=np.float32)
        return cls(
            key=np.asarray(key, dtype=np.float32),
            state=np.asarray(sample["state"], dtype=np.float32),
            instruction=str(sample["instruction"]),
            base_action=base_action,
            expert_action=expert_action,
            residual_target=expert_action - base_action,
            future_state=np.asarray(sample["future_state"], dtype=np.float32),
            return_to_go=float(sample["return_to_go"]),
        )


def hash_text_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Create a deterministic, dependency-free language embedding for indexing.

    GROOT's language encoder remains the preferred embedding when available;
    this fallback makes indexing reproducible for offline preprocessing and is
    intentionally not presented as a semantic language model.
    """

    if dim < 1:
        raise ValueError("dim must be positive")
    vector = np.zeros(dim, dtype=np.float32)
    encoded = str(text).lower().encode("utf-8")
    for index, byte in enumerate(encoded):
        vector[(byte + 31 * index) % dim] += 1.0 if index % 2 == 0 else -1.0
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-6)


class RetrievalBank:
    """Exact cosine retrieval with a fixed-size weighted context aggregate."""

    def __init__(self, keys: np.ndarray | None = None, records: Iterable[RetrievalRecord] = ()) -> None:
        self.keys = np.empty((0, 0), dtype=np.float32) if keys is None else np.asarray(keys, dtype=np.float32)
        self.records = list(records)
        if self.records and keys is None:
            self.keys = np.stack([record.key for record in self.records])
        if len(self.records) != len(self.keys):
            raise ValueError("keys and records must have the same number of rows")
        if self.keys.ndim == 2 and len(self.keys) and np.any(~np.isfinite(self.keys)):
            raise ValueError("retrieval keys must be finite")
        self._contexts = self._make_contexts()

    @property
    def key_dim(self) -> int:
        return int(self.keys.shape[-1]) if self.keys.ndim == 2 else 0

    @property
    def context_dim(self) -> int:
        return int(self._contexts.shape[-1]) if self._contexts.ndim == 2 else 0

    def _make_contexts(self) -> np.ndarray:
        if not self.records:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(
            [
                np.concatenate(
                    [
                        record.state,
                        record.base_action,
                        record.expert_action,
                        record.residual_target,
                        record.future_state,
                        np.asarray([record.return_to_go], dtype=np.float32),
                    ]
                )
                for record in self.records
            ]
        ).astype(np.float32)

    def add(self, record: RetrievalRecord) -> None:
        key = np.asarray(record.key, dtype=np.float32).reshape(1, -1)
        if self.key_dim and key.shape[1] != self.key_dim:
            raise ValueError(f"record key dim {key.shape[1]} != bank key dim {self.key_dim}")
        self.keys = key if not len(self.keys) else np.concatenate([self.keys, key], axis=0)
        self.records.append(record)
        self._contexts = self._make_contexts()

    def query(self, query_key: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """Return record indices and cosine scores in descending score order."""

        if k < 1:
            raise ValueError("k must be positive")
        query = np.asarray(query_key, dtype=np.float32)
        if query.ndim != 1 or (self.key_dim and query.shape[0] != self.key_dim):
            raise ValueError(f"query key must have shape ({self.key_dim},), got {query.shape}")
        if not len(self.records):
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        keys = self.keys / np.maximum(np.linalg.norm(self.keys, axis=1, keepdims=True), 1e-6)
        query = query / max(float(np.linalg.norm(query)), 1e-6)
        scores = keys @ query
        count = min(k, len(scores))
        indices = np.argsort(-scores, kind="stable")[:count]
        return indices.astype(np.int32), scores[indices].astype(np.float32)

    def aggregate(self, query_key: np.ndarray, k: int = 8) -> np.ndarray:
        """Return a fixed-size weighted context, zero-filled for an empty bank."""

        if not self.records:
            return np.zeros(self.context_dim, dtype=np.float32)
        indices, scores = self.query(query_key, k)
        weights = np.exp(scores - np.max(scores))
        weights /= np.maximum(np.sum(weights), 1e-6)
        return np.sum(self._contexts[indices] * weights[:, None], axis=0).astype(np.float32)

    def save(self, path: Path) -> None:
        """Save arrays and JSON metadata without pickling arbitrary objects."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), keys=self.keys, contexts=self._contexts)
        metadata = {
            "instructions": [record.instruction for record in self.records],
            "states": [record.state.tolist() for record in self.records],
            "base_actions": [record.base_action.tolist() for record in self.records],
            "expert_actions": [record.expert_action.tolist() for record in self.records],
            "residual_targets": [record.residual_target.tolist() for record in self.records],
            "future_states": [record.future_state.tolist() for record in self.records],
            "returns_to_go": [record.return_to_go for record in self.records],
        }
        path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RetrievalBank:
        path = Path(path)
        arrays = np.load(path.with_suffix(".npz"))
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        records = []
        for index, instruction in enumerate(metadata["instructions"]):
            records.append(
                RetrievalRecord(
                    key=arrays["keys"][index],
                    state=np.asarray(metadata["states"][index], dtype=np.float32),
                    instruction=instruction,
                    base_action=np.asarray(metadata["base_actions"][index], dtype=np.float32),
                    expert_action=np.asarray(metadata["expert_actions"][index], dtype=np.float32),
                    residual_target=np.asarray(metadata["residual_targets"][index], dtype=np.float32),
                    future_state=np.asarray(metadata["future_states"][index], dtype=np.float32),
                    return_to_go=float(metadata["returns_to_go"][index]),
                )
            )
        bank = cls(keys=np.asarray(arrays["keys"], dtype=np.float32), records=records)
        if not np.allclose(bank._contexts, arrays["contexts"]):
            raise ValueError("retrieval metadata and stored contexts disagree")
        return bank

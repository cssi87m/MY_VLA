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
    episode_id: str = ""
    timestep: int = -1
    source: str = "expert"
    tail_reward: float = 0.0
    tail_discount: float = 1.0

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
            episode_id=str(sample.get("episode_id", "")),
            timestep=int(sample.get("timestep", -1)),
            source=str(sample.get("source", "expert")),
            tail_reward=float(sample.get("tail_reward", 0.0)),
            tail_discount=float(sample.get("tail_discount", 1.0)),
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


def checkpoint_retrieval_key(state: Any, instruction: str) -> np.ndarray:
    """Build the standard state-and-language key used at a repair checkpoint.

    The state must be the actual state observed after the base prefix has run.
    This lightweight first version does not require another VLM forward pass.
    """

    checkpoint_state = np.asarray(state, dtype=np.float32)
    if checkpoint_state.ndim != 1:
        raise ValueError(f"checkpoint state must be one-dimensional, got {checkpoint_state.shape}")
    if not np.all(np.isfinite(checkpoint_state)):
        raise ValueError("checkpoint state must be finite")
    return np.concatenate([checkpoint_state, hash_text_embedding(instruction)]).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class RetrievedTails:
    """Fixed-size action-tail candidates returned by checkpoint retrieval."""

    expert_actions: np.ndarray
    scores: np.ndarray
    returns_to_go: np.ndarray
    mask: np.ndarray


def tail_candidate_features(result: RetrievedTails) -> np.ndarray:
    """Flatten retrieved tails and retrieval metadata for the residual actor."""

    return np.concatenate(
        [result.expert_actions.reshape(-1), result.scores, result.returns_to_go, result.mask]
    ).astype(np.float32)


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
        self._normalized_keys = self._normalize_keys()
        self._contexts = self._make_contexts()

    @property
    def key_dim(self) -> int:
        return int(self.keys.shape[-1]) if self.keys.ndim == 2 else 0

    @property
    def context_dim(self) -> int:
        return int(self._contexts.shape[-1]) if self._contexts.ndim == 2 else 0

    @property
    def expert_action_dim(self) -> int:
        return int(self.records[0].expert_action.size) if self.records else 0

    def _normalize_keys(self) -> np.ndarray:
        if not len(self.keys):
            return self.keys
        return self.keys / np.maximum(np.linalg.norm(self.keys, axis=1, keepdims=True), 1e-6)

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
        self._normalized_keys = self._normalize_keys()
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
        query = query / max(float(np.linalg.norm(query)), 1e-6)
        scores = self._normalized_keys @ query
        count = min(k, len(scores))
        indices = np.argsort(-scores, kind="stable")[:count]
        return indices.astype(np.int32), scores[indices].astype(np.float32)

    def retrieve_tails(
        self,
        query_key: np.ndarray,
        *,
        retrieval_k: int,
        exclude_episode_id: str | None = None,
    ) -> RetrievedTails:
        """Retrieve padded expert action tails, optionally excluding an episode."""

        if retrieval_k < 1:
            raise ValueError("retrieval_k must be positive")
        query = np.asarray(query_key, dtype=np.float32)
        if query.ndim != 1 or (self.key_dim and query.shape[0] != self.key_dim):
            raise ValueError(f"query key must have shape ({self.key_dim},), got {query.shape}")
        action_dim = self.expert_action_dim
        expert_actions = np.zeros((retrieval_k, action_dim), dtype=np.float32)
        scores = np.zeros(retrieval_k, dtype=np.float32)
        returns = np.zeros(retrieval_k, dtype=np.float32)
        mask = np.zeros(retrieval_k, dtype=np.float32)
        if not self.records:
            return RetrievedTails(expert_actions, scores, returns, mask)

        eligible = np.ones(len(self.records), dtype=bool)
        if exclude_episode_id is not None:
            eligible &= np.asarray([record.episode_id != exclude_episode_id for record in self.records])
        eligible_indices = np.flatnonzero(eligible)
        if not len(eligible_indices):
            return RetrievedTails(expert_actions, scores, returns, mask)

        normalized_query = query / max(float(np.linalg.norm(query)), 1e-6)
        all_scores = self._normalized_keys[eligible_indices] @ normalized_query
        count = min(retrieval_k, len(eligible_indices))
        order = np.argsort(-all_scores, kind="stable")[:count]
        selected = eligible_indices[order]
        for output_index, record_index in enumerate(selected):
            record = self.records[int(record_index)]
            if record.expert_action.size != action_dim:
                raise ValueError("all retrieval records must have the same expert action shape")
            expert_actions[output_index] = record.expert_action
            scores[output_index] = all_scores[order[output_index]]
            returns[output_index] = record.return_to_go
            mask[output_index] = 1.0
        return RetrievedTails(expert_actions, scores, returns, mask)

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
            "episode_ids": [record.episode_id for record in self.records],
            "timesteps": [record.timestep for record in self.records],
            "sources": [record.source for record in self.records],
            "tail_rewards": [record.tail_reward for record in self.records],
            "tail_discounts": [record.tail_discount for record in self.records],
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
                    episode_id=str(metadata.get("episode_ids", [""] * len(metadata["instructions"]))[index]),
                    timestep=int(metadata.get("timesteps", [-1] * len(metadata["instructions"]))[index]),
                    source=str(metadata.get("sources", ["expert"] * len(metadata["instructions"]))[index]),
                    tail_reward=float(metadata.get("tail_rewards", [0.0] * len(metadata["instructions"]))[index]),
                    tail_discount=float(metadata.get("tail_discounts", [1.0] * len(metadata["instructions"]))[index]),
                )
            )
        bank = cls(keys=np.asarray(arrays["keys"], dtype=np.float32), records=records)
        if not np.allclose(bank._contexts, arrays["contexts"]):
            raise ValueError("retrieval metadata and stored contexts disagree")
        return bank

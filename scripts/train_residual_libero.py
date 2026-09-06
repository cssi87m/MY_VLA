"""Train the LIBERO residual heads from a GROOT N1.5 base policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import torch

from src.my_vla.data.libero import DEFAULT_LIBERO_DATA_ROOT
from src.my_vla.data.libero import LiberoConfig
from src.my_vla.data.libero import iter_libero_transitions
from src.my_vla.models.base_vlm import GrootN15Adapter
from src.my_vla.retrieval.bank import RetrievalBank
from src.my_vla.retrieval.bank import checkpoint_retrieval_key
from src.my_vla.retrieval.bank import tail_candidate_features
from src.my_vla.training.pretrain import PretrainConfig
from src.my_vla.training.pretrain import initialize_pretraining
from src.my_vla.training.pretrain import pretrain_step


@dataclass(frozen=True)
class TrainConfig:
    """Inputs and hyperparameters for residual pretraining."""

    groot_model_path: str
    groot_data_config: str = "src.my_vla.models.groot_libero_config:LiberoDataConfig"
    groot_embodiment_tag: str = "new_embodiment"
    device: str = "cuda"
    output: Path = Path("checkpoints/residual_libero")
    memory_bank: Path = Path("checkpoints/expert_memory_bank")
    data_root: Path = DEFAULT_LIBERO_DATA_ROOT
    horizon: int = 8
    replan_steps: int = 4
    retrieval_k: int = 8
    max_samples: int = 256
    epochs: int = 1
    batch_size: int = 16
    seed: int = 0


def _parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-model-path", required=True)
    parser.add_argument("--groot-data-config", default="src.my_vla.models.groot_libero_config:LiberoDataConfig")
    parser.add_argument("--groot-embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/residual_libero"))
    parser.add_argument("--memory-bank", type=Path, default=Path("checkpoints/expert_memory_bank"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_LIBERO_DATA_ROOT)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=4,
        help="Execute this many base actions before the residual corrects the remaining tail.",
    )
    parser.add_argument("--retrieval-k", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return TrainConfig(**vars(parser.parse_args(argv)))


def _make_adapter(args: TrainConfig) -> GrootN15Adapter:
    try:
        from gr00t.experiment.data_config import load_data_config  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Install NVIDIA Isaac-GR00T N1.5 to run this script") from exc
    data_config = load_data_config(args.groot_data_config)
    return GrootN15Adapter(
        model_path=args.groot_model_path,
        embodiment_tag=args.groot_embodiment_tag,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        device=args.device,
    )


def _pad_action_chunk(actions: np.ndarray, horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if len(actions) >= horizon:
        return actions[:horizon]
    if not len(actions):
        return np.zeros((horizon, 7), dtype=np.float32)
    return np.concatenate([actions, np.repeat(actions[-1:], horizon - len(actions), axis=0)], axis=0)


class LiberoSampleCollector:
    """Collect GR00T features while the expert memory bank stays fixed."""

    def __init__(self, config: TrainConfig, adapter: GrootN15Adapter) -> None:
        self.config = config
        self.adapter = adapter

    def collect_features(self) -> list[dict]:
        args = self.config
        adapter = self.adapter
        samples = []
        if not 1 <= args.replan_steps < args.horizon:
            raise ValueError("replan_steps must be at least 1 and smaller than horizon")
        for sample in iter_libero_transitions(
            LiberoConfig(args.data_root, horizon=args.horizon, replan_steps=args.replan_steps)
        ):
            base = adapter(sample["observation"], sample["instruction"])
            base_chunk = _pad_action_chunk(base.base_action, args.horizon)
            hidden = np.asarray(base.hidden, dtype=np.float32)
            pooled_hidden = hidden.mean(axis=tuple(range(hidden.ndim - 1))) if hidden.ndim > 1 else hidden
            checkpoint_state = np.asarray(sample["state_chunk"][args.replan_steps], dtype=np.float32)
            base_tail = base_chunk[args.replan_steps :]
            samples.append(
                {
                    **sample,
                    "hidden": pooled_hidden,
                    "base_action_prefix": base_chunk[: args.replan_steps],
                    "base_action_tail": base_tail,
                    "checkpoint_state": checkpoint_state,
                }
            )
            if args.max_samples > 0 and len(samples) >= args.max_samples:
                break
        if not samples:
            raise RuntimeError("No valid LIBERO transitions were found at the required dataset path")
        return samples

    def collect(self) -> tuple[list, list[dict]]:
        """Compatibility entry point; retrieval records now come from disk."""
        return [], self.collect_features()


def collect_samples(args: TrainConfig, adapter: GrootN15Adapter):
    """Compatibility entry point for callers collecting features directly."""
    return LiberoSampleCollector(args, adapter).collect_features()


class ResidualLiberoTrainer:
    """Coordinate feature collection, pretraining, and checkpoint storage."""

    def __init__(self, config: TrainConfig) -> None:
        if not 1 <= config.replan_steps < config.horizon:
            raise ValueError("replan_steps must be at least 1 and smaller than horizon")
        if config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if config.retrieval_k < 1:
            raise ValueError("retrieval_k must be positive")
        self.config = config

    def run(self) -> None:
        bank = self._load_memory_bank()
        adapter = _make_adapter(self.config)
        try:
            samples = LiberoSampleCollector(self.config, adapter).collect_features()
        finally:
            # Release GROOT before JAX allocates memory for training.
            del adapter
            gc.collect()
            torch.cuda.empty_cache()

        self.config.output.mkdir(parents=True, exist_ok=True)
        config, params = self.train(samples, bank)
        self.save_checkpoint(config, params)

    def _load_memory_bank(self) -> RetrievalBank:
        root = self.config.memory_bank
        bank_path = root / "expert_memory_bank" if root.is_dir() else root
        manifest_path = bank_path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"memory-bank manifest was not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "residual_expert_memory_bank_v1":
            raise ValueError("memory bank is not a residual expert memory-bank artifact")
        bank_config = manifest.get("config", {})
        for name, expected in (("horizon", self.config.horizon), ("replan_steps", self.config.replan_steps)):
            if int(bank_config.get(name, -1)) != expected:
                raise ValueError(f"memory bank {name}={bank_config.get(name)} != training {name}={expected}")
        bank = RetrievalBank.load(bank_path)
        if bank.key_dim != 7 + 64:
            raise ValueError(f"memory bank key dim {bank.key_dim} is not a checkpoint-state key")
        if bank.expert_action_dim != (self.config.horizon - self.config.replan_steps) * 7:
            raise ValueError("memory bank expert-tail shape does not match the requested horizon")
        return bank

    def train(self, samples: list[dict], bank: RetrievalBank):
        """Train residual heads from collected features."""
        args = self.config
        hidden_dim = int(samples[0]["hidden"].shape[-1])
        config = PretrainConfig(
            action_horizon=args.horizon,
            replan_steps=args.replan_steps,
            vlm_hidden_dim=hidden_dim,
            retrieval_k=args.retrieval_k,
        )
        bundle, params, opt_state = initialize_pretraining(jax.random.key(args.seed), config)
        for _ in range(args.epochs):
            for start in range(0, len(samples), args.batch_size):
                batch_samples = samples[start : start + args.batch_size]
                batch = self._make_batch(batch_samples, bank)
                params, opt_state, metrics = pretrain_step(bundle, params, opt_state, batch)
                print({key: float(value) for key, value in metrics.items()})

        return config, params

    def _make_batch(self, batch_samples: list[dict], bank: RetrievalBank) -> dict:
        args = self.config
        retrieval_context = []
        for sample in batch_samples:
            key = checkpoint_retrieval_key(sample["checkpoint_state"], sample["instruction"])
            result = bank.retrieve_tails(
                key,
                retrieval_k=args.retrieval_k,
                exclude_episode_id=sample.get("episode_id"),
            )
            retrieval_context.append(tail_candidate_features(result))
        return {
            "hidden": jnp.asarray(np.stack([x["hidden"] for x in batch_samples])),
            "state": jnp.asarray(np.stack([x["state"] for x in batch_samples])),
            "base_action_prefix": jnp.asarray(np.stack([x["base_action_prefix"] for x in batch_samples])),
            "base_action_tail": jnp.asarray(np.stack([x["base_action_tail"] for x in batch_samples])),
            "future_state": jnp.asarray(np.stack([x["checkpoint_state"] for x in batch_samples])),
            "residual_target": jnp.asarray(
                np.stack([x["expert_action_chunk"][args.replan_steps :] - x["base_action_tail"] for x in batch_samples])
            ),
            "retrieval_context": jnp.asarray(np.stack(retrieval_context)),
        }

    def save_checkpoint(self, config: PretrainConfig, params: dict) -> None:
        (self.config.output / "params.msgpack").write_bytes(flax.serialization.to_bytes(params))
        (self.config.output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def main() -> None:
    ResidualLiberoTrainer(_parse_args()).run()


if __name__ == "__main__":
    main()

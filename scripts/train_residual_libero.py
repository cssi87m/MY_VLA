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
from src.my_vla.retrieval.bank import RetrievalRecord
from src.my_vla.retrieval.bank import hash_text_embedding
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
    data_root: Path = DEFAULT_LIBERO_DATA_ROOT
    horizon: int = 8
    replan_steps: int = 4
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
    parser.add_argument("--data-root", type=Path, default=DEFAULT_LIBERO_DATA_ROOT)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=4,
        help="Execute this many base actions before the residual corrects the remaining tail.",
    )
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
    """Build retrieval records and training features with one base adapter."""

    def __init__(self, config: TrainConfig, adapter: GrootN15Adapter) -> None:
        self.config = config
        self.adapter = adapter

    def collect(self) -> tuple[list[RetrievalRecord], list[dict]]:
        args = self.config
        adapter = self.adapter
        records = []
        samples = []
        if not 1 <= args.replan_steps < args.horizon:
            raise ValueError("replan_steps must be at least 1 and smaller than horizon")
        for sample in iter_libero_transitions(LiberoConfig(args.data_root, horizon=args.horizon)):
            base = adapter(sample["observation"], sample["instruction"])
            base_chunk = _pad_action_chunk(base.base_action, args.horizon)
            hidden = np.asarray(base.hidden, dtype=np.float32)
            pooled_hidden = hidden.mean(axis=tuple(range(hidden.ndim - 1))) if hidden.ndim > 1 else hidden
            key = np.concatenate(
                [pooled_hidden, np.asarray(sample["state"], dtype=np.float32), hash_text_embedding(sample["instruction"])]
            )
            checkpoint_state = np.asarray(sample["state_chunk"][args.replan_steps], dtype=np.float32)
            base_tail = base_chunk[args.replan_steps :]
            expert_tail = np.asarray(sample["expert_action_chunk"][args.replan_steps :], dtype=np.float32)
            record_sample = {
                **sample,
                "state": checkpoint_state,
                "expert_action": expert_tail.reshape(-1),
                "future_state": checkpoint_state,
            }
            records.append(RetrievalRecord.from_sample(record_sample, key, base_tail.reshape(-1)))
            samples.append(
                {
                    **sample,
                    "hidden": pooled_hidden,
                    "base_action_prefix": base_chunk[: args.replan_steps],
                    "base_action_tail": base_tail,
                    "checkpoint_state": checkpoint_state,
                    "key": key,
                }
            )
            if args.max_samples > 0 and len(samples) >= args.max_samples:
                break
        if not samples:
            raise RuntimeError("No valid LIBERO transitions were found at the required dataset path")
        return records, samples


def collect_samples(args: TrainConfig, adapter: GrootN15Adapter):
    """Compatibility entry point for callers collecting features directly."""
    return LiberoSampleCollector(args, adapter).collect()


class ResidualLiberoTrainer:
    """Coordinate feature collection, pretraining, and checkpoint storage."""

    def __init__(self, config: TrainConfig) -> None:
        if not 1 <= config.replan_steps < config.horizon:
            raise ValueError("replan_steps must be at least 1 and smaller than horizon")
        if config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.config = config

    def run(self) -> None:
        adapter = _make_adapter(self.config)
        try:
            records, samples = LiberoSampleCollector(self.config, adapter).collect()
        finally:
            # Release GROOT before JAX allocates memory for training.
            del adapter
            gc.collect()
            torch.cuda.empty_cache()

        bank = RetrievalBank(records=records)
        self.config.output.mkdir(parents=True, exist_ok=True)
        bank.save(self.config.output / "retrieval_bank")
        config, params = self.train(records, samples, bank)
        self.save_checkpoint(config, params)

    def train(self, records: list[RetrievalRecord], samples: list[dict], bank: RetrievalBank):
        """Train residual heads from collected features."""
        args = self.config
        hidden_dim = int(samples[0]["hidden"].shape[-1])
        config = PretrainConfig(
            action_horizon=args.horizon,
            replan_steps=args.replan_steps,
            vlm_hidden_dim=hidden_dim,
            context_dim=bank.context_dim,
        )
        bundle, params, opt_state = initialize_pretraining(jax.random.key(args.seed), config)
        for _ in range(args.epochs):
            for start in range(0, len(samples), args.batch_size):
                batch_samples = samples[start : start + args.batch_size]
                batch_records = records[start : start + args.batch_size]
                batch = self._make_batch(batch_samples, batch_records, bank)
                params, opt_state, metrics = pretrain_step(bundle, params, opt_state, batch)
                print({key: float(value) for key, value in metrics.items()})

        return config, params

    def _make_batch(self, batch_samples: list[dict], batch_records: list[RetrievalRecord], bank: RetrievalBank) -> dict:
        args = self.config
        return {
            "hidden": jnp.asarray(np.stack([x["hidden"] for x in batch_samples])),
            "state": jnp.asarray(np.stack([x["state"] for x in batch_samples])),
            "base_action_prefix": jnp.asarray(np.stack([x["base_action_prefix"] for x in batch_samples])),
            "base_action_tail": jnp.asarray(np.stack([x["base_action_tail"] for x in batch_samples])),
            "future_state": jnp.asarray(np.stack([x["checkpoint_state"] for x in batch_samples])),
            "residual_target": jnp.asarray(
                np.stack([x["expert_action_chunk"][args.replan_steps :] - x["base_action_tail"] for x in batch_samples])
            ),
            "retrieval_context": jnp.asarray(
                np.stack([bank.aggregate(record.key, k=8) for record in batch_records])
            ),
        }

    def save_checkpoint(self, config: PretrainConfig, params: dict) -> None:
        (self.config.output / "params.msgpack").write_bytes(flax.serialization.to_bytes(params))
        (self.config.output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def main() -> None:
    ResidualLiberoTrainer(_parse_args()).run()


if __name__ == "__main__":
    main()

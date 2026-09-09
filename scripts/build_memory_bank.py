"""Build a checkpoint-aligned expert memory bank from LIBERO demonstrations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.my_vla.data.libero import DEFAULT_LIBERO_DATA_ROOT
from src.my_vla.data.libero import LiberoConfig
from src.my_vla.data.libero import iter_libero_transitions
from src.my_vla.models.base_vlm import GrootN15Adapter
from src.my_vla.retrieval.bank import RetrievalBank
from src.my_vla.retrieval.bank import RetrievalRecord
from src.my_vla.retrieval.bank import checkpoint_retrieval_key


@dataclass(frozen=True)
class MemoryBankBuildConfig:
    """Inputs for constructing an expert-only checkpoint memory bank."""

    groot_model_path: str
    groot_data_config: str = "src.my_vla.models.groot_libero_config:LiberoDataConfig"
    groot_embodiment_tag: str = "new_embodiment"
    device: str = "cuda"
    output: Path = Path("checkpoints/expert_memory_bank")
    data_root: Path = DEFAULT_LIBERO_DATA_ROOT
    horizon: int = 8
    replan_steps: int = 4
    max_samples: int = 0


def _parse_args(argv: list[str] | None = None) -> MemoryBankBuildConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-model-path", required=True)
    parser.add_argument("--groot-data-config", default="src.my_vla.models.groot_libero_config:LiberoDataConfig")
    parser.add_argument("--groot-embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/expert_memory_bank"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_LIBERO_DATA_ROOT)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use every valid transition.")
    return MemoryBankBuildConfig(**vars(parser.parse_args(argv)))


def _make_adapter(config: MemoryBankBuildConfig) -> GrootN15Adapter:
    try:
        from gr00t.experiment.data_config import load_data_config  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("Install NVIDIA Isaac-GR00T N1.5 to build a memory bank") from exc
    data_config = load_data_config(config.groot_data_config)
    adapter = GrootN15Adapter(
        model_path=config.groot_model_path,
        embodiment_tag=config.groot_embodiment_tag,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        device=config.device,
    )
    adapter.policy.model.eval()
    adapter.policy.model.requires_grad_(False)
    return adapter


def _pad_action_chunk(actions: np.ndarray, horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if len(actions) >= horizon:
        return actions[:horizon]
    if not len(actions):
        raise ValueError("GR00T returned an empty action chunk")
    return np.concatenate([actions, np.repeat(actions[-1:], horizon - len(actions), axis=0)])


class ExpertMemoryBankBuilder:
    """Create immutable expert records keyed by the correction checkpoint."""

    def __init__(self, config: MemoryBankBuildConfig) -> None:
        if not 1 <= config.replan_steps < config.horizon:
            raise ValueError("replan_steps must be at least 1 and smaller than horizon")
        self.config = config

    def build_records(self, adapter: GrootN15Adapter) -> list[RetrievalRecord]:
        records: list[RetrievalRecord] = []
        config = self.config
        dataset = LiberoConfig(config.data_root, horizon=config.horizon, replan_steps=config.replan_steps)
        for sample in tqdm(iter_libero_transitions(dataset), desc="Building expert memory bank"):
            base = adapter(sample["observation"], sample["instruction"])
            base_chunk = _pad_action_chunk(base.base_action, config.horizon)
            checkpoint_state = np.asarray(sample["state_chunk"][config.replan_steps], dtype=np.float32)
            base_tail = base_chunk[config.replan_steps :]
            expert_tail = np.asarray(sample["expert_action_chunk"][config.replan_steps :], dtype=np.float32)
            record_sample = {
                "state": checkpoint_state,
                "instruction": sample["instruction"],
                "expert_action": expert_tail.reshape(-1),
                "future_state": sample["future_state"],
                "return_to_go": sample["checkpoint_return_to_go"],
                "tail_reward": sample["tail_reward"],
                "tail_discount": sample["tail_discount"],
                "episode_id": sample["episode_id"],
                "timestep": sample["timestep"],
                "source": "expert",
            }
            records.append(
                RetrievalRecord.from_sample(
                    record_sample,
                    checkpoint_retrieval_key(checkpoint_state, sample["instruction"]),
                    base_tail.reshape(-1),
                )
            )
            if config.max_samples > 0 and len(records) >= config.max_samples:
                break
        if not records:
            raise RuntimeError("No valid LIBERO transitions were found at the required dataset path")
        return records

    def run(self) -> dict[str, int | str]:
        adapter = _make_adapter(self.config)
        records = self.build_records(adapter)
        bank = RetrievalBank(records=records)
        self.config.output.mkdir(parents=True, exist_ok=True)
        bank_path = self.config.output / "expert_memory_bank"
        bank.save(bank_path)
        manifest = {
            "format": "residual_expert_memory_bank_v1",
            "records": len(records),
            "key": "checkpoint_state_plus_instruction_hash",
            "payload": "base_tail_expert_tail_residual_target_future_state_return",
            "source": "expert",
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self.config).items()},
        }
        (self.config.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"records": len(records), "bank": str(bank_path)}


def main() -> None:
    print(json.dumps(ExpertMemoryBankBuilder(_parse_args()).run(), indent=2))


if __name__ == "__main__":
    main()

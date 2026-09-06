"""Evaluate residual repair of the unexecuted tail after a base-plan checkpoint."""
# ruff: noqa: E402, PLC0415

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import flax.serialization
import jax.numpy as jnp
import numpy as np

from src.my_vla.data.libero import DEFAULT_LIBERO_DATA_ROOT
from src.my_vla.data.libero import LiberoConfig
from src.my_vla.data.libero import iter_libero_transitions
from src.my_vla.models.base_vlm import GrootN15Adapter
from src.my_vla.models.future_state import ActionConditionedTransition
from src.my_vla.models.future_state import future_consistency
from src.my_vla.models.projector import LatentProjector
from src.my_vla.retrieval.bank import RetrievalBank
from src.my_vla.retrieval.bank import hash_text_embedding
from src.my_vla.rl.residual_ac import ResidualActor
from src.my_vla.rl.residual_ac import build_actor_input
from src.my_vla.rl.residual_ac import clip_libero_action
from src.my_vla.training.pretrain import PretrainConfig


@dataclass(frozen=True)
class EvalConfig:
    """Dataset and checkpoint inputs for offline evaluation."""

    groot_model_path: str
    checkpoint: Path
    groot_data_config: str = "my_vla.models.groot_libero_config:LiberoDataConfig"
    groot_embodiment_tag: str = "new_embodiment"
    device: str = "cuda"
    data_root: Path = DEFAULT_LIBERO_DATA_ROOT
    max_samples: int = 256
    replan_steps: int | None = None


def _parse_args(argv: list[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-model-path", required=True)
    parser.add_argument("--groot-data-config", default="my_vla.models.groot_libero_config:LiberoDataConfig")
    parser.add_argument("--groot-embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_LIBERO_DATA_ROOT)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="Checkpoint after this many base actions; defaults to the value saved with the checkpoint.",
    )
    return EvalConfig(**vars(parser.parse_args(argv)))


class ResidualLiberoEvaluator:
    """Load residual assets once and measure tail correction over a dataset."""

    def __init__(self, config: EvalConfig) -> None:
        self.args = config

    def load_checkpoint(self) -> None:
        args = self.args
        from gr00t.experiment.data_config import load_data_config

        data_config = load_data_config(args.groot_data_config)
        adapter = GrootN15Adapter(
            model_path=args.groot_model_path,
            embodiment_tag=args.groot_embodiment_tag,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            device=args.device,
        )
        # Set to eval mode
        adapter.policy.model.eval()
        adapter.policy.model.requires_grad_(False)

        config_data = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
        self.config = config = PretrainConfig(
            **{key: value for key, value in config_data.items() if key in PretrainConfig.__dataclass_fields__}
        )
        replan_steps = config.replan_steps if args.replan_steps is None else args.replan_steps
        if replan_steps != config.replan_steps:
            raise ValueError(
                f"checkpoint was trained with replan_steps={config.replan_steps}; "
                "choose that value or train a checkpoint for the requested checkpoint."
            )
        self.params = flax.serialization.msgpack_restore((args.checkpoint / "params.msgpack").read_bytes())
        self.bank = RetrievalBank.load(args.checkpoint / "retrieval_bank")
        self.projector = LatentProjector(output_dim=config.latent_dim)
        self.transition = ActionConditionedTransition(
            state_dim=config.state_dim, action_dim=config.action_dim, action_horizon=config.action_horizon
        )
        self.actor = ResidualActor(action_dim=config.action_dim, action_horizon=config.correction_horizon)
        self.adapter = adapter

    def evaluate_sample(self, sample: dict) -> tuple[float, float]:
        """Return base and corrected tail MSE using the loaded checkpoint."""
        config = self.config
        replan_steps = config.replan_steps
        output = self.adapter(sample["observation"], sample["instruction"])
        hidden = jnp.asarray(output.hidden[None])
        base_chunk = jnp.asarray(output.base_action[None, : config.action_horizon])
        if base_chunk.shape[1] < config.action_horizon:
            base_chunk = jnp.concatenate(
                [base_chunk, jnp.repeat(base_chunk[:, -1:], config.action_horizon - base_chunk.shape[1], axis=1)],
                axis=1,
            )
        state = jnp.asarray(sample["state"][None])
        # The base plan runs through the checkpoint; the residual only repairs
        # the tail that has not yet been executed.

        #  Prefix: first chunk to "replan_steps" chunk
        base_prefix = base_chunk[:, :replan_steps]
        # Tail: from "replan_steps" chunk to the end of the base plan. This tail is 
        # where the residual repair is applied.
        base_tail = base_chunk[:, replan_steps:]
        checkpoint_state = jnp.asarray(sample["state_chunk"][replan_steps][None])

        # Calculate the policy's predicted future state based on the current state and the base plan prefix.
        policy_future = self.transition.apply({"params": self.params["transition"]}, state, base_prefix)
        # Latent representation of the hidden state. This value represent the underlying features 
        # extracted from the observation and instruction.
        latent = self.projector.apply({"params": self.params["projector"]}, hidden)
        consistency = future_consistency(checkpoint_state, policy_future)
        hidden_np = np.asarray(output.hidden)
        pooled_hidden = hidden_np.mean(axis=tuple(range(hidden_np.ndim - 1))) if hidden_np.ndim > 1 else hidden_np
        key = np.concatenate([pooled_hidden, sample["state"], hash_text_embedding(sample["instruction"])])
        context = jnp.asarray(self.bank.aggregate(key)[None])
        actor_input = build_actor_input(latent, base_tail.reshape(1, -1), consistency, context)
        delta = self.actor.apply({"params": self.params["actor"]}, actor_input)[0].reshape(config.correction_horizon, config.action_dim)
        final = clip_libero_action(base_tail[0] + delta)
        expert = jnp.asarray(sample["expert_action_chunk"][replan_steps:])
        return (
            float(jnp.mean(jnp.square(base_tail[0] - expert))),
            float(jnp.mean(jnp.square(final - expert))),
        )

    def run(self) -> dict:
        self.load_checkpoint()
        total_base = total_final = 0.0
        count = 0
        dataset = LiberoConfig(self.args.data_root, horizon=self.config.action_horizon)
        for sample in iter_libero_transitions(dataset):
            base_mse, final_mse = self.evaluate_sample(sample)
            total_base += base_mse
            total_final += final_mse
            count += 1
            if self.args.max_samples > 0 and count >= self.args.max_samples:
                break
        if not count:
            raise RuntimeError("No valid LIBERO transitions were found at the required dataset path")
        return {
            "samples": count,
            "replan_steps": self.config.replan_steps,
            "correction_horizon": self.config.correction_horizon,
            "base_tail_mse": total_base / count,
            "residual_tail_mse": total_final / count,
        }


def main() -> None:
    print(json.dumps(ResidualLiberoEvaluator(_parse_args()).run(), indent=2))


if __name__ == "__main__":
    main()

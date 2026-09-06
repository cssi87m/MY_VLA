"""Online LIBERO policy that adds a learned residual to GR00T actions.

The residual checkpoint is deliberately not an OpenPI checkpoint: it contains
only the small Flax heads trained in ``train_residual_libero.py``.  This module
is the bridge for simulators and serving code.  It accepts the request schema
used by ``lap/scripts/libero/main.py`` and returns a one-action chunk.  Use a
replan interval of one simulator step, since the residual actor is trained
only for the first action in each GR00T action horizon.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from my_vla.models.base_vlm import BaseVLM
from my_vla.models.base_vlm import GrootN15Adapter
from my_vla.models.future_state import ActionConditionedTransition
from my_vla.models.future_state import FutureStateHead
from my_vla.models.future_state import future_consistency
from my_vla.models.projector import LatentProjector
from my_vla.retrieval.bank import RetrievalBank
from my_vla.retrieval.bank import hash_text_embedding
from my_vla.rl.residual_ac import ResidualActor
from my_vla.rl.residual_ac import build_actor_input
from my_vla.rl.residual_ac import clip_libero_action
from my_vla.training.pretrain import PretrainConfig


def lap_rot6d_state_to_libero_state(state: Any) -> np.ndarray:
    """Convert LAP's 10-D ``xyz + rot6d + gripper`` state to 7-D LIBERO state.

    The two 3-D rotation columns are orthonormalized before extracting the
    extrinsic XYZ Euler angles used by the residual checkpoint's training data.
    A 7-D input is returned unchanged, which is handy for native callers.
    """

    value = np.asarray(state, dtype=np.float32)
    if value.shape == (7,):
        return value
    if value.shape != (10,):
        raise ValueError(f"state must be a 7-D LIBERO or 10-D LAP state, got {value.shape}")
    first = value[3:6]
    second = value[6:9]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-6:
        raise ValueError("LAP rotation's first 6-D column has zero length")
    column_0 = first / first_norm
    second = second - column_0 * np.dot(column_0, second)
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-6:
        raise ValueError("LAP rotation's two 6-D columns are collinear")
    column_1 = second / second_norm
    column_2 = np.cross(column_0, column_1)
    rotation = np.stack([column_0, column_1, column_2], axis=1)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.concatenate([value[:3], np.asarray([roll, pitch, yaw, value[9]], dtype=np.float32)])


class ResidualLiberoRolloutPolicy:
    """Execute a base prefix, then repair its unexecuted tail without GR00T."""

    def __init__(
        self,
        *,
        base_vlm: BaseVLM,
        config: PretrainConfig,
        params: Mapping[str, Any],
        retrieval_bank: RetrievalBank,
    ) -> None:
        if config.state_dim != 7 or config.action_dim != 7:
            raise ValueError("LIBERO rollout requires 7-D state and action configurations")
        required = {"projector", "future_head", "transition", "actor"}
        missing = required.difference(params)
        if missing:
            raise ValueError(f"residual checkpoint is missing parameter groups: {sorted(missing)}")
        if retrieval_bank.context_dim != config.context_dim:
            raise ValueError(
                f"retrieval context dim {retrieval_bank.context_dim} != checkpoint context dim {config.context_dim}"
            )
        self.base_vlm = base_vlm
        self.config = config
        self.params = params
        self.retrieval_bank = retrieval_bank
        self.projector = LatentProjector(output_dim=config.latent_dim)
        self.future_head = FutureStateHead(state_dim=config.state_dim)
        self.transition = ActionConditionedTransition(
            state_dim=config.state_dim, action_dim=config.action_dim, action_horizon=config.action_horizon
        )
        self.actor = ResidualActor(action_dim=config.action_dim, action_horizon=config.correction_horizon)
        self._cached_plan: _CachedPlan | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        groot_model_path: str,
        groot_data_config: str = "my_vla.models.groot_libero_config:LiberoDataConfig",
        groot_embodiment_tag: str = "new_embodiment",
        device: str = "cuda",
        is_infer: bool=False
    ) -> ResidualLiberoRolloutPolicy:
        """Load the GR00T base model and residual assets from a checkpoint directory."""

        import flax.serialization  # noqa: PLC0415
        from gr00t.experiment.data_config import load_data_config  # noqa: PLC0415

        checkpoint = Path(checkpoint)
        config_data = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        config = PretrainConfig(**{key: value for key, value in config_data.items() if key in PretrainConfig.__dataclass_fields__})
        params = flax.serialization.msgpack_restore((checkpoint / "params.msgpack").read_bytes())
        data_config = load_data_config(groot_data_config)
        base_vlm = GrootN15Adapter(
            model_path=groot_model_path,
            embodiment_tag=groot_embodiment_tag,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            device=device,
        )
        if is_infer:
            base_vlm.policy.model.eval()
            base_vlm.policy.model.requires_grad_(False)

        return cls(base_vlm=base_vlm, config=config, params=params, retrieval_bank=RetrievalBank.load(checkpoint / "retrieval_bank"))

    def infer(self, request: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Return a base prefix, then a GR00T-free corrected tail on the next call."""

        raw_observation = request.get("observation")
        if not isinstance(raw_observation, Mapping):
            raise ValueError("request must contain an observation mapping")
        try:
            image = raw_observation["base_0_rgb"]
            wrist_image = raw_observation["left_wrist_0_rgb"]
            state = lap_rot6d_state_to_libero_state(raw_observation["state"])
            instruction = str(request["prompt"])
        except KeyError as exc:
            raise ValueError(f"request is missing required field: {exc.args[0]}") from exc
        if request.get("reset_residual_plan"):
            self._cached_plan = None

        if self._cached_plan is None:
            output = self.base_vlm({"image": image, "wrist_image": wrist_image, "state": state}, instruction)
            base_chunk = self._base_chunk(output.base_action)
            hidden = jnp.asarray(output.hidden)[None]
            latent = self.projector.apply({"params": self.params["projector"]}, hidden)
            pooled_hidden = self._pool_hidden(output.hidden)
            key = np.concatenate([pooled_hidden, state, hash_text_embedding(instruction)]).astype(np.float32)
            context = jnp.asarray(self.retrieval_bank.aggregate(key)[None])
            prefix = base_chunk[:, : self.config.replan_steps]
            tail = base_chunk[:, self.config.replan_steps :]
            expected_checkpoint = self.transition.apply({"params": self.params["transition"]}, jnp.asarray(state)[None], prefix)
            self._cached_plan = _CachedPlan(
                state=jnp.asarray(state)[None],
                prefix=prefix,
                tail=tail,
                expected_checkpoint=expected_checkpoint,
                latent=latent,
                context=context,
            )
            return {
                "actions": np.asarray(clip_libero_action(prefix[0]), dtype=np.float32),
                "replan_steps": self.config.replan_steps,
                "residual_phase": "base_prefix",
            }

        cached = self._cached_plan
        self._cached_plan = None
        actual_checkpoint = jnp.asarray(state)[None]
        consistency = future_consistency(actual_checkpoint, cached.expected_checkpoint)
        actor_input = build_actor_input(cached.latent, cached.tail.reshape(1, -1), consistency, cached.context)
        residual = self.actor.apply({"params": self.params["actor"]}, actor_input)[0]
        corrected_tail = clip_libero_action(cached.tail[0] + residual.reshape(self.config.correction_horizon, self.config.action_dim))
        return {
            "actions": np.asarray(corrected_tail, dtype=np.float32),
            "replan_steps": self.config.replan_steps,
            "residual_phase": "corrected_tail",
        }

    def _base_chunk(self, base_action: Any) -> jnp.ndarray:
        chunk = np.asarray(base_action, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != self.config.action_dim:
            raise ValueError(f"GR00T must return (H,{self.config.action_dim}) actions, got {chunk.shape}")
        if len(chunk) == 0:
            raise ValueError("GR00T returned an empty action chunk")
        chunk = chunk[: self.config.action_horizon]
        if len(chunk) < self.config.action_horizon:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], self.config.action_horizon - len(chunk), axis=0)])
        return jnp.asarray(chunk[None])

    @staticmethod
    def _pool_hidden(hidden: Any) -> np.ndarray:
        value = np.asarray(hidden, dtype=np.float32)
        if value.ndim < 1:
            raise ValueError(f"GR00T hidden features must have at least one dimension, got {value.shape}")
        return value.mean(axis=tuple(range(value.ndim - 1))) if value.ndim > 1 else value


@dataclasses.dataclass(frozen=True)
class _CachedPlan:
    """Information retained between prefix execution and tail correction."""

    state: jnp.ndarray
    prefix: jnp.ndarray
    tail: jnp.ndarray
    expected_checkpoint: jnp.ndarray
    latent: jnp.ndarray
    context: jnp.ndarray

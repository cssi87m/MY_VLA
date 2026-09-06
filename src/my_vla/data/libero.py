"""Read the modified LIBERO RLDS data used by OpenVLA.

The raw dataset stores an 8-D state as ``xyz + axis-angle + two finger
positions`` and a 7-D action as ``xyz + axis-angle + gripper``.  This module
uses the same convention as LAP's ``libero_dataset_transform``: state
orientation is converted to extrinsic XYZ Euler angles and the first finger
position is scaled by 0.04 m.  The resulting state is 7-D and gripper values
are ``1=open, 0=closed``.

TensorFlow is imported only when an RLDS builder is opened.  Conversion
helpers themselves are JAX functions so they can be used in a jitted train
step without bringing TF tensors into the model.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

DEFAULT_LIBERO_DATA_ROOT = Path("/home/vrh3/workspace/lap/data/openvla-modified_libero_rlds")
DEFAULT_DATASET_NAMES = (
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
)


@dataclasses.dataclass(frozen=True)
class LiberoConfig:
    """Configuration for extracting fixed-horizon RLDS training samples."""

    data_root: Path = DEFAULT_LIBERO_DATA_ROOT
    dataset_names: tuple[str, ...] = DEFAULT_DATASET_NAMES
    horizon: int = 8
    replan_steps: int = 0

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0 <= self.replan_steps < self.horizon:
            raise ValueError("replan_steps must be non-negative and smaller than horizon")


def _axis_angle_to_matrix(axis_angle: jnp.ndarray) -> jnp.ndarray:
    angle = jnp.linalg.norm(axis_angle, axis=-1, keepdims=True)
    safe_angle = jnp.maximum(angle, 1e-8)
    axis = axis_angle / safe_angle
    default_axis = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0], dtype=axis.dtype), axis.shape)
    axis = jnp.where(angle < 1e-8, default_axis, axis)
    x, y, z = jnp.moveaxis(axis, -1, 0)
    c = jnp.cos(angle[..., 0])
    s = jnp.sin(angle[..., 0])
    one_c = 1.0 - c
    return jnp.stack(
        [
            jnp.stack([c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s], axis=-1),
            jnp.stack([y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s], axis=-1),
            jnp.stack([z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c], axis=-1),
        ],
        axis=-2,
    )


def axis_angle_to_extrinsic_xyz(axis_angle: Any) -> jnp.ndarray:
    """Convert axis-angle to LAP's extrinsic XYZ/RPY representation."""

    rotation = _axis_angle_to_matrix(jnp.asarray(axis_angle, dtype=jnp.float32))
    r20, r21, r22 = rotation[..., 2, 0], rotation[..., 2, 1], rotation[..., 2, 2]
    r10, r00 = rotation[..., 1, 0], rotation[..., 0, 0]
    pitch = jnp.arcsin(jnp.clip(-r20, -1.0, 1.0))
    roll = jnp.arctan2(r21, r22)
    yaw = jnp.arctan2(r10, r00)
    return jnp.stack([roll, pitch, yaw], axis=-1)


def libero_state(raw_state: Any) -> jnp.ndarray:
    """Return a canonical 7-D ``xyz + extrinsic XYZ + open gripper`` state."""

    state = jnp.asarray(raw_state, dtype=jnp.float32)
    if state.shape[-1] != 8:
        raise ValueError(f"LIBERO raw state must have 8 values, got {state.shape}")
    orientation = axis_angle_to_extrinsic_xyz(state[..., 3:6])
    gripper = jnp.clip(state[..., -2:-1] / 0.04, 0.0, 1.0)
    return jnp.concatenate([state[..., :3], orientation, gripper], axis=-1)


def libero_action(raw_action: Any) -> jnp.ndarray:
    """Return a canonical 7-D action with LAP's open-gripper convention."""

    action = jnp.asarray(raw_action, dtype=jnp.float32)
    if action.shape[-1] != 7:
        raise ValueError(f"LIBERO action must have 7 values, got {action.shape}")
    gripper = 1.0 - jnp.clip(action[..., 6:7], 0.0, 1.0)
    return jnp.concatenate([action[..., :6], gripper], axis=-1)


def _as_numpy(value: Any) -> Any:
    if hasattr(value, "numpy"):
        return value.numpy()
    if isinstance(value, dict):
        return {key: _as_numpy(item) for key, item in value.items()}
    return value


def _decode_instruction(value: Any) -> str:
    value = _as_numpy(value)
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _check_root(data_root: Path) -> Path:
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"LIBERO RLDS data was not found at the required path: {data_root}. "
            "Pass the exact dataset root explicitly if it is mounted elsewhere."
        )
    return data_root


def load_libero_episodes(
    data_root: Path = DEFAULT_LIBERO_DATA_ROOT,
    dataset_names: Sequence[str] = DEFAULT_DATASET_NAMES,
    *,
    split: str = "train",
) -> Iterator[dict[str, Any]]:
    """Yield decoded episodes from the local TFDS RLDS builders.

    The function deliberately does not download or substitute a different
    path.  This keeps training reproducible and enforces the path specified by
    the architecture plan.
    """

    import tensorflow_datasets as tfds  # noqa: PLC0415

    root = _check_root(data_root)
    for dataset_name in dataset_names:
        dataset_dir = root / dataset_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"LIBERO dataset builder is missing: {dataset_dir}")
        builder = tfds.builder(dataset_name, data_dir=str(root), try_gcs=False)
        for episode in builder.as_dataset(split=split, shuffle_files=False):
            steps = [_as_numpy(step) for step in episode["steps"]]
            if not steps:
                continue
            yield {
                "dataset_name": dataset_name,
                "episode_metadata": _as_numpy(episode["episode_metadata"]),
                "steps": steps,
            }


def _episode_arrays(episode: dict[str, Any]) -> dict[str, Any]:
    steps = episode["steps"]
    raw_states = np.stack([np.asarray(step["observation"]["state"], dtype=np.float32) for step in steps])
    raw_actions = np.stack([np.asarray(step["action"], dtype=np.float32) for step in steps])
    states = np.asarray(libero_state(raw_states))
    actions = np.asarray(libero_action(raw_actions))
    instructions = [_decode_instruction(step["language_instruction"]) for step in steps]
    rewards = np.asarray([step.get("reward", 0.0) for step in steps], dtype=np.float32)
    discounts = np.asarray([step.get("discount", 1.0) for step in steps], dtype=np.float32)
    rtg = np.zeros_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + float(discounts[index]) * running
        rtg[index] = running
    return {
        "states": states,
        "actions": actions,
        "instructions": instructions,
        "rewards": rewards,
        "discounts": discounts,
        "returns": rtg,
    }


def iter_libero_transitions(
    config: LiberoConfig | None = None,
    *,
    split: str = "train",
) -> Iterator[dict[str, Any]]:
    """Yield one valid ``t -> t+H`` sample, including its state sequence."""

    config = LiberoConfig() if config is None else config
    for episode_index, episode in enumerate(load_libero_episodes(config.data_root, config.dataset_names, split=split)):
        arrays = _episode_arrays(episode)
        length = len(arrays["states"])
        for timestep in range(max(0, length - config.horizon)):
            step = episode["steps"][timestep]
            observation = step["observation"]
            yield {
                "episode_id": f"{episode['dataset_name']}:{episode_index}",
                "timestep": timestep,
                "observation": {
                    "image": np.asarray(observation["image"], dtype=np.uint8),
                    "wrist_image": np.asarray(observation["wrist_image"], dtype=np.uint8),
                    "state": arrays["states"][timestep],
                },
                "instruction": arrays["instructions"][timestep],
                "state": arrays["states"][timestep],
                "expert_action": arrays["actions"][timestep],
                "expert_action_chunk": arrays["actions"][timestep : timestep + config.horizon],
                # Retaining intermediate states lets a residual policy use an
                # actual checkpoint observation after a base-plan prefix.
                "state_chunk": arrays["states"][timestep : timestep + config.horizon + 1],
                "future_state": arrays["states"][timestep + config.horizon],
                "return_to_go": arrays["returns"][timestep],
                "checkpoint_return_to_go": arrays["returns"][timestep + config.replan_steps],
                "tail_reward": arrays["rewards"][timestep + config.replan_steps],
                "tail_discount": arrays["discounts"][timestep + config.replan_steps],
            }

# `my_vla.data`

LIBERO RLDS loading and conversion to the convention used by the residual
policy. The package deliberately opens TensorFlow/TFDS only while reading a
dataset; the conversion helpers are JAX-compatible and can run in compiled
training code.

## Canonical convention

Canonical states and actions have seven values:

```text
[x, y, z, roll, pitch, yaw, gripper]
```

- A raw LIBERO state is eight values: position, axis-angle orientation, and
  two finger positions. `libero_state()` converts the orientation to
  extrinsic XYZ Euler angles and maps the first finger value from metres to a
  clipped `0`–`1` open-gripper value.
- A raw LIBERO action is seven values. `libero_action()` preserves the first
  six dimensions and flips the dataset gripper convention so `1` means open
  and `0` means closed.

Keep this convention at every package boundary. In particular, do not feed a
raw eight-dimensional dataset state to model or RL components.

## Public API

| Symbol | Purpose |
| --- | --- |
| `LiberoConfig` | Immutable dataset root, selected dataset names, and fixed action horizon. The horizon must be positive. |
| `load_libero_episodes()` | Lazily yields decoded episodes from local TFDS RLDS builders. It neither downloads data nor substitutes a path. |
| `iter_libero_transitions()` | Lazily yields valid fixed-horizon samples for training or evaluation. |
| `libero_state()` / `libero_action()` | Convert raw arrays to the canonical convention. |
| `axis_angle_to_extrinsic_xyz()` | JAX conversion helper for raw orientation values. |
| `DEFAULT_LIBERO_DATA_ROOT` | Default local dataset path; override it with `LiberoConfig(data_root=...)` for portable runs. |

## Transition samples

`iter_libero_transitions(LiberoConfig(...))` yields dictionaries containing:

| Key | Shape / meaning |
| --- | --- |
| `observation` | `image`, `wrist_image`, and canonical `state` at timestep `t`. |
| `instruction` | Language instruction for the timestep. |
| `state`, `expert_action` | Canonical `(7,)` current state and action. |
| `expert_action_chunk` | Canonical expert actions with shape `(H, 7)`. |
| `state_chunk` | Current through horizon-end states with shape `(H + 1, 7)`. |
| `future_state` | State at `t + H`, shape `(7,)`. |
| `return_to_go` | Discounted return from `t`, computed from the episode rewards and discounts. |

Only timesteps that have a complete horizon are emitted. Dataset roots must
contain the selected TFDS builders (by default, the four `*_no_noops` LIBERO
sets); a missing root or builder raises `FileNotFoundError`.

## Example

```python
from pathlib import Path
from my_vla.data import LiberoConfig, iter_libero_transitions

config = LiberoConfig(data_root=Path("/datasets/openvla-modified_libero_rlds"), horizon=8)
sample = next(iter_libero_transitions(config))
assert sample["expert_action_chunk"].shape == (8, 7)
```

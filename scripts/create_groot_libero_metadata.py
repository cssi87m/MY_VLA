"""Create GR00T ``new_embodiment`` metadata from canonical LIBERO training data.

The output can be merged with a base checkpoint's ``experiment_cfg/metadata.json``
and placed in a copied fine-tuning checkpoint directory. It never modifies the
Hugging Face cache.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.my_vla.data.libero import DEFAULT_LIBERO_DATA_ROOT
from src.my_vla.data.libero import libero_action
from src.my_vla.data.libero import libero_state
from src.my_vla.data.libero import load_libero_episodes

STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def _statistics(values: list[np.ndarray]) -> dict[str, list[float]]:
    data = np.concatenate(values, axis=0)
    return {
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.maximum(np.std(data, axis=0), 1e-6).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


def _image_metadata(image: np.ndarray, fps: float) -> dict[str, Any]:
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected an HWC image, got shape {image.shape}")
    height, width, channels = image.shape
    return {"resolution": [width, height], "channels": channels, "fps": fps}


def build_new_embodiment_metadata(episodes: Iterable[dict[str, Any]], *, fps: float) -> dict[str, Any]:
    """Calculate a schema-valid entry using every training timestep.

    This uses the same conversion functions as the residual pipeline. In
    particular, ``state.gripper`` becomes two finger positions, while
    ``action.gripper`` remains a scalar command.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")

    values: dict[str, dict[str, list[np.ndarray]]] = {
        "state": defaultdict(list),
        "action": defaultdict(list),
    }
    video_metadata: dict[str, dict[str, Any]] | None = None
    step_count = 0

    for episode in episodes:
        steps = episode["steps"]
        if not steps:
            continue
        raw_states = np.stack([np.asarray(step["observation"]["state"], dtype=np.float32) for step in steps])
        raw_actions = np.stack([np.asarray(step["action"], dtype=np.float32) for step in steps])
        states = np.asarray(libero_state(raw_states), dtype=np.float32)
        actions = np.asarray(libero_action(raw_actions), dtype=np.float32)
        fingers = np.stack([states[:, 6] * 0.04, -states[:, 6] * 0.04], axis=-1)

        for index, key in enumerate(STATE_KEYS[:-1]):
            values["state"][key].append(states[:, index : index + 1])
        values["state"]["gripper"].append(fingers)
        for index, key in enumerate(ACTION_KEYS):
            values["action"][key].append(actions[:, index : index + 1])

        observed_video = {
            "image": _image_metadata(np.asarray(steps[0]["observation"]["image"]), fps),
            "wrist_image": _image_metadata(np.asarray(steps[0]["observation"]["wrist_image"]), fps),
        }
        if video_metadata is None:
            video_metadata = observed_video
        elif video_metadata != observed_video:
            raise ValueError("LIBERO camera shape changed between episodes")
        step_count += len(steps)

    if not step_count or video_metadata is None:
        raise ValueError("no LIBERO training steps were found")

    state_modalities = {
        key: {"absolute": True, "rotation_type": None, "shape": [1], "continuous": True}
        for key in STATE_KEYS[:-1]
    }
    state_modalities["gripper"] = {
        "absolute": True,
        "rotation_type": None,
        "shape": [2],
        "continuous": True,
    }
    action_modalities = {
        key: {"absolute": False, "rotation_type": None, "shape": [1], "continuous": True}
        for key in ACTION_KEYS
    }
    return {
        "embodiment_tag": "new_embodiment",
        "modalities": {
            "video": video_metadata,
            "state": state_modalities,
            "action": action_modalities,
        },
        "statistics": {
            modality: {key: _statistics(series) for key, series in by_key.items()}
            for modality, by_key in values.items()
        },
    }


def write_metadata(
    entry: dict[str, Any], output: Path, *, base_metadata: Path | None = None, overwrite: bool = False
) -> None:
    """Write ``entry`` as ``new_embodiment``, optionally retaining base tags."""
    metadata: dict[str, Any] = {}
    if base_metadata is not None:
        metadata = json.loads(base_metadata.read_text())
    if output.exists() and base_metadata != output:
        metadata = json.loads(output.read_text())
    if "new_embodiment" in metadata and not overwrite:
        raise FileExistsError(f"{output} already has a new_embodiment entry; pass --overwrite to replace it")
    metadata["new_embodiment"] = entry
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_LIBERO_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True, help="New metadata.json path; never use the HF cache.")
    parser.add_argument("--base-metadata", type=Path, help="Checkpoint metadata.json whose pretrained entries to retain.")
    parser.add_argument("--fps", type=float, required=True, help="LIBERO observation rate in frames per second.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    entry = build_new_embodiment_metadata(load_libero_episodes(args.data_root), fps=args.fps)
    write_metadata(entry, args.output, base_metadata=args.base_metadata, overwrite=args.overwrite)
    print(f"Wrote new_embodiment metadata for {len(entry['statistics']['state'])} state keys to {args.output}")


if __name__ == "__main__":
    main()

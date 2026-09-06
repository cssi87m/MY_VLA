"""GROOT N1.5 adapter with a small, testable model protocol."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any, Protocol

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class BaseVLMOutput:
    """The base-policy values consumed by the residual pipeline."""

    base_action: np.ndarray
    hidden: np.ndarray


class BaseVLM(Protocol):
    """Minimal interface allowing GROOT or a deterministic test double."""

    def __call__(self, observation: dict[str, Any], instruction: str) -> BaseVLMOutput: ...


def default_libero_groot_observation(observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Map canonical samples to the flattened modality keys used by GROOT N1.5."""

    state = np.asarray(observation["state"], dtype=np.float32)
    image = np.array(observation["image"], dtype=np.uint8, copy=True)
    wrist_image = np.array(observation["wrist_image"], dtype=np.uint8, copy=True)
    if state.shape[-1] != 7:
        raise ValueError(f"canonical LIBERO state must have 7 values, got {state.shape}")
    gripper = np.concatenate(
      [state[6:7] * 0.04, -state[6:7] * 0.04]
  )
    # GROOT expects (batch, time, ...) and the modality config supplies the
    # actual names. These defaults match the N1.5 LIBERO/simulation config.
    return {
        "video.image": image[None, None],
        "video.wrist_image": wrist_image[None, None],
        "state.x": state[None, None, 0:1],
        "state.y": state[None, None, 1:2],
        "state.z": state[None, None, 2:3],
        "state.roll": state[None, None, 3:4],
        "state.pitch": state[None, None, 4:5],
        "state.yaw": state[None, None, 5:6],
        "state.gripper": gripper[None, None],
        "annotation.human.action.task_description": np.asarray([[instruction]], dtype=object),
    }


class GrootN15Adapter:
    """Expose GROOT N1.5 action chunks and backbone features.

    GROOT is optional because the JAX residual components are useful for unit
    tests and offline preprocessing without a GPU. Pass an already-created
    ``gr00t.policy.policy.Gr00tPolicy`` when embedding this in a larger app,
    or pass its constructor arguments to load one lazily.
    """

    def __init__(
        self,
        policy: Any | None = None,
        *,
        model_path: str | None = None,
        embodiment_tag: str = "new_embodiment",
        modality_config: dict[str, Any] | None = None,
        modality_transform: Any | None = None,
        device: str = "cuda",
        observation_builder: Callable[[dict[str, Any], str], dict[str, Any]] = default_libero_groot_observation,
    ) -> None:
        if policy is None:
            if model_path is None:
                raise ValueError("provide policy or model_path")
            if modality_config is None or modality_transform is None:
                raise ValueError("GROOT N1.5 requires modality_config and modality_transform")
            try:
                from gr00t.model.policy import Gr00tPolicy  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError("Install NVIDIA Isaac-GR00T N1.5 to use GrootN15Adapter") from exc
            policy = Gr00tPolicy(
                model_path=model_path,
                embodiment_tag=embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                device=device,
            )
        self.policy = policy
        self.observation_builder = observation_builder
        self._eagle_dtype_hook = self._install_eagle_dtype_hook()

    def _install_eagle_dtype_hook(self) -> Any | None:
        """Make Eagle vision features compatible with its LLM embeddings."""
        model = getattr(self.policy, "model", None)
        backbone = getattr(model, "backbone", None)
        eagle = getattr(backbone, "eagle_model", None)
        projector = getattr(eagle, "mlp1", None)
        language_model = getattr(eagle, "language_model", None)
        if projector is None or language_model is None:
            return None

        embedding = language_model.get_input_embeddings()
        target_dtype = next(embedding.parameters()).dtype

        def cast_projector_output(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, torch.Tensor) and output.dtype != target_dtype:
                return output.to(dtype=target_dtype)
            return output

        return projector.register_forward_hook(cast_projector_output)

    def __call__(self, observation: dict[str, Any], instruction: str) -> BaseVLMOutput:
        raw_observation = self.observation_builder(observation, instruction)
        # The N1.5 policy's public path performs modality transforms. Reusing
        # it here preserves checkpoint normalization and temporal indexing.
        with torch.inference_mode():
            with torch.autocast("cuda"):
                normalized = self.policy.apply_transforms(raw_observation)
                model = self.policy.model
                backbone_inputs, action_inputs = model.prepare_input(normalized)
                backbone_outputs = model.backbone(backbone_inputs)
                action_head_outputs = model.action_head.get_action(backbone_outputs, action_inputs)
        actions = action_head_outputs["action_pred"]

        actions = self.policy.unapply_transforms({"action": actions.float().cpu()})
        # Construct actions to be tensor, not dict
        action_tensor = np.concatenate(
            [actions[key] for key in actions.keys()],
            axis=-1,
        )

        hidden = backbone_outputs["backbone_features"]
        hidden = hidden.detach().float().cpu().numpy() if hasattr(hidden, "detach") else np.asarray(hidden)
        if hasattr(action_tensor, "detach"):
            action_tensor = action_tensor.detach().cpu().numpy()
        action_tensor = np.asarray(action_tensor)
        if action_tensor.ndim != 3:
            raise ValueError(f"GROOT action prediction must be (B,H,A), got {action_tensor.shape}")
        return BaseVLMOutput(
            base_action=action_tensor[0],
            hidden=hidden[0],
        )

# ruff: noqa: PLC0415, RUF012, I001
"""GROOT N1.5 modality configuration for canonical LIBERO samples.
# ruff: noqa: PLC0415, RUF012

This module is imported by GROOT's ``load_data_config`` only when the optional
GROOT dependency is installed.  It keeps the model-specific transform outside
the JAX package while giving the adapter a reproducible LIBERO configuration.
"""

from __future__ import annotations


class LiberoDataConfig:
    """N1.5 config for two RGB views and canonical 7-D EEF state/actions."""

    video_keys = ["video.image", "video.wrist_image"]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        from gr00t.data.dataset import ModalityConfig

        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        from gr00t.data.transform.base import ComposedModalityTransform
        from gr00t.data.transform.concat import ConcatTransform
        from gr00t.data.transform.state_action import StateActionToTensor
        from gr00t.data.transform.video import VideoResize, VideoToNumpy, VideoToTensor
        from gr00t.model.transforms import GR00TTransform

        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

"""Serve the GR00T + residual LIBERO rollout policy over OpenPI websocket RPC.

Example:
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \\
      python -m scripts.serve_residual_libero \\
      --groot-model-path "$GROOT_CHECKPOINT" \\
      --checkpoint checkpoints/residual_libero \\
      --memory-bank checkpoints/expert_memory_bank

The corresponding LIBERO evaluator must use the ``--replan-steps`` value
saved in the residual checkpoint.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from openpi.serving import websocket_policy_server
import tyro

from src.my_vla.models.residual_libero_rollout import ResidualLiberoRolloutPolicy


@dataclasses.dataclass(frozen=True)
class Args:
    """Configuration for the residual LIBERO websocket server."""

    groot_model_path: str
    checkpoint: Path
    memory_bank: Path
    groot_data_config: str = "my_vla.models.groot_libero_config:LiberoDataConfig"
    groot_embodiment_tag: str = "new_embodiment"
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 11004


class ResidualLiberoServer:
    """Own the rollout policy and its websocket serving lifecycle."""

    def __init__(self, args: Args) -> None:
        self.args = args

    def load_policy(self) -> None:
        args = self.args
        self.policy = ResidualLiberoRolloutPolicy.from_checkpoint(
            args.checkpoint,
            memory_bank=args.memory_bank,
            groot_model_path=args.groot_model_path,
            groot_data_config=args.groot_data_config,
            groot_embodiment_tag=args.groot_embodiment_tag,
            device=args.device,
            is_infer=True,
        )

    def run(self) -> None:
        self.load_policy()
        args = self.args
        policy = self.policy
        metadata = {
            "policy_type": "groot_residual_libero",
            "action_dim": policy.config.action_dim,
            "action_horizon": policy.config.action_horizon,
            "recommended_replan_steps": policy.config.replan_steps,
            "checkpoint": str(args.checkpoint),
            "memory_bank": str(args.memory_bank),
        }
        logging.info("Serving residual LIBERO policy on %s:%d", args.host, args.port)
        websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=metadata,
        ).serve_forever()


def main(args: Args) -> None:
    """Load the policy once, then serve it until the process is stopped."""
    ResidualLiberoServer(args).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

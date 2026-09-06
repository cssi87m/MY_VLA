"""Stage 1--3 initialization: future-state/dynamics heads and residual BC."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import optax

from src.my_vla.models.future_state import ActionConditionedTransition
from src.my_vla.models.future_state import future_consistency
from src.my_vla.models.projector import LatentProjector
from src.my_vla.rl.residual_ac import ResidualActor
from src.my_vla.rl.residual_ac import build_actor_input


@dataclasses.dataclass(frozen=True)
class PretrainConfig:
    state_dim: int = 7
    action_dim: int = 7
    action_horizon: int = 8
    replan_steps: int = 4
    latent_dim: int = 256
    vlm_hidden_dim: int = 1024
    retrieval_k: int = 8
    consistency_dim: int = 18
    learning_rate: float = 3e-4
    residual_loss_weight: float = 1.0

    @property
    def correction_horizon(self) -> int:
        return self.action_horizon - self.replan_steps

    @property
    def retrieval_context_dim(self) -> int:
        tail_dim = self.correction_horizon * self.action_dim
        return self.retrieval_k * (tail_dim + 3)


@dataclasses.dataclass
class PretrainBundle:
    config: PretrainConfig
    projector: LatentProjector
    transition: ActionConditionedTransition
    actor: ResidualActor
    optimizer: optax.GradientTransformation


def initialize_pretraining(
    rng: jax.Array, config: PretrainConfig | None = None
) -> tuple[PretrainBundle, dict, optax.OptState]:
    """Initialize all trainable heads using representative shapes."""

    config = PretrainConfig() if config is None else config
    if not 1 <= config.replan_steps < config.action_horizon:
        raise ValueError("replan_steps must be at least 1 and smaller than action_horizon")
    if config.consistency_dim != 2 * config.state_dim + 4:
        raise ValueError("consistency_dim must equal 2 * state_dim + 4")
    bundle = PretrainBundle(
        config=config,
        projector=LatentProjector(output_dim=config.latent_dim),
        transition=ActionConditionedTransition(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
        ),
        actor=ResidualActor(action_dim=config.action_dim, action_horizon=config.correction_horizon),
        optimizer=optax.adam(config.learning_rate),
    )
    keys = jax.random.split(rng, 4)
    hidden = jnp.zeros((2, 16, config.vlm_hidden_dim), dtype=jnp.float32)
    state = jnp.zeros((2, config.state_dim), dtype=jnp.float32)
    prefix_actions = jnp.zeros((2, config.replan_steps, config.action_dim), dtype=jnp.float32)
    tail_actions = jnp.zeros((2, config.correction_horizon, config.action_dim), dtype=jnp.float32)
    context = jnp.zeros((2, config.retrieval_context_dim), dtype=jnp.float32)
    consistency = jnp.zeros((2, config.consistency_dim), dtype=jnp.float32)
    params = {
        "projector": bundle.projector.init(keys[0], hidden)["params"],
        "transition": bundle.transition.init(keys[1], state, prefix_actions)["params"],
        "actor": bundle.actor.init(
            keys[2], jnp.concatenate([jnp.zeros((2, config.latent_dim)), tail_actions.reshape(2, -1), consistency, context], -1)
        )["params"],
    }
    return bundle, params, bundle.optimizer.init(params)


def _loss(
    bundle: PretrainBundle, params: dict, batch: dict[str, jnp.ndarray]
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    latent = bundle.projector.apply({"params": params["projector"]}, batch["hidden"])
    policy_future = bundle.transition.apply(
        {"params": params["transition"]}, batch["state"], batch["base_action_prefix"]
    )
    future_target = batch["future_state"]
    weights = batch["sample_weight"]

    def weighted_mean(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(value * weights) / jnp.maximum(jnp.sum(weights), 1.0)

    future_loss = weighted_mean(jnp.mean(jnp.square(policy_future - future_target), axis=-1))
    # At the checkpoint the residual sees the actual state reached after the
    # base prefix, not a second model prediction.
    consistency = future_consistency(future_target, policy_future)
    actor_input = build_actor_input(
        latent, batch["base_action_tail"].reshape(batch["base_action_tail"].shape[0], -1), consistency, batch["retrieval_context"]
    )
    predicted_residual = bundle.actor.apply({"params": params["actor"]}, actor_input)
    residual_target = batch["residual_target"].reshape(predicted_residual.shape)
    residual_loss = weighted_mean(jnp.mean(jnp.square(predicted_residual - residual_target), axis=-1))
    total = future_loss + bundle.config.residual_loss_weight * residual_loss
    return total, {"loss": total, "future_loss": future_loss, "residual_loss": residual_loss}


def pretrain_step(bundle: PretrainBundle, params: dict, opt_state: optax.OptState, batch: dict[str, jnp.ndarray]):
    """Run one JAX update for future prediction and residual behavior cloning."""

    def loss_fn(current_params):
        return _loss(bundle, current_params, batch)

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = bundle.optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, metrics


def make_pretrain_step(bundle: PretrainBundle):
    """Return a compiled fixed-shape residual update for the training loop."""

    def step(params: dict, opt_state: optax.OptState, batch: dict[str, jnp.ndarray]):
        return pretrain_step(bundle, params, opt_state, batch)

    return jax.jit(step)

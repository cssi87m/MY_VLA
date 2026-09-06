"""Flax residual actor and conservative twin critic."""

from __future__ import annotations

from collections.abc import Sequence

from flax import linen as nn
import jax.numpy as jnp

DEFAULT_MAX_RESIDUAL = (0.02, 0.02, 0.02, 0.10, 0.10, 0.10, 1.0)


def build_actor_input(
    latent: jnp.ndarray,
    base_action: jnp.ndarray,
    consistency: jnp.ndarray,
    retrieval_context: jnp.ndarray,
) -> jnp.ndarray:
    """Concatenate the four explicitly specified residual-policy inputs."""

    arrays = [jnp.asarray(value) for value in (latent, base_action, consistency, retrieval_context)]
    if len({array.ndim for array in arrays}) != 1:
        raise ValueError(f"actor inputs must have equal rank, got {[array.shape for array in arrays]}")
    return jnp.concatenate(arrays, axis=-1)


class ResidualActor(nn.Module):
    """Bounded residual plan-repair policy initialized close to zero."""

    action_dim: int = 7
    action_horizon: int = 1
    max_residual: Sequence[float] = DEFAULT_MAX_RESIDUAL
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, actor_input: jnp.ndarray) -> jnp.ndarray:
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if len(self.max_residual) != self.action_dim:
            raise ValueError("max_residual must contain one bound per action dimension")
        x = nn.relu(nn.Dense(self.hidden_dim)(actor_input))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        residual = nn.tanh(nn.Dense(self.action_horizon * self.action_dim, kernel_init=nn.initializers.zeros)(x))
        return residual * jnp.tile(jnp.asarray(self.max_residual, dtype=jnp.float32), self.action_horizon)


class _Critic(nn.Module):
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, critic_input: jnp.ndarray) -> jnp.ndarray:
        x = nn.relu(nn.Dense(self.hidden_dim)(critic_input))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(1)(x)[..., 0]


class TwinCritic(nn.Module):
    """Two independent Q functions used for conservative offline estimates."""

    hidden_dim: int = 512

    @nn.compact
    def __call__(self, critic_input: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _Critic(self.hidden_dim, name="critic_1")(critic_input), _Critic(self.hidden_dim, name="critic_2")(
            critic_input
        )


def clip_libero_action(action: jnp.ndarray) -> jnp.ndarray:
    """Clip a canonical LIBERO action to normalized pose and gripper ranges."""

    action = jnp.asarray(action)
    if action.shape[-1] != 7:
        raise ValueError(f"LIBERO action must have 7 values, got {action.shape}")
    low = jnp.asarray([-1.0] * 6 + [0.0], dtype=action.dtype)
    high = jnp.asarray([1.0] * 6 + [1.0], dtype=action.dtype)
    return jnp.clip(action, low, high)


def critic_loss(q1: jnp.ndarray, q2: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Twin-critic regression loss with the lower estimate used at inference."""

    target = jax_stop_gradient(target)
    return 0.5 * (jnp.mean(jnp.square(q1 - target)) + jnp.mean(jnp.square(q2 - target)))


def advantage_weighted_residual_loss(
    predicted_residual: jnp.ndarray,
    target_residual: jnp.ndarray,
    advantage: jnp.ndarray,
    *,
    temperature: float = 1.0,
    max_weight: float = 20.0,
    residual_penalty: float = 1e-3,
) -> jnp.ndarray:
    """IQL/AWR-style weighted BC objective plus a large-residual penalty."""

    weights = jnp.exp(jnp.clip(advantage / temperature, -10.0, 10.0))
    weights = jnp.minimum(weights, max_weight)
    error = jnp.mean(jnp.square(predicted_residual - target_residual), axis=-1)
    penalty = jnp.mean(jnp.square(predicted_residual))
    return jnp.mean(weights * error) + residual_penalty * penalty


def jax_stop_gradient(value: jnp.ndarray) -> jnp.ndarray:
    """Local wrapper keeps this module's public dependency surface tiny."""

    from jax import lax  # noqa: PLC0415

    return lax.stop_gradient(value)

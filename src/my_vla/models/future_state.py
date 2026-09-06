"""Future-state prediction and consistency features for LIBERO."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp


class FutureStateHead(nn.Module):
    """Predict the canonical 7-D LIBERO state at the end of the horizon."""

    state_dim: int = 7
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, hidden: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.asarray(hidden)
        if hidden.ndim > 2:
            hidden = jnp.mean(hidden, axis=tuple(range(1, hidden.ndim - 1)))
        x = nn.gelu(nn.Dense(self.hidden_dim)(hidden))
        x = nn.gelu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.state_dim)(x)


class ActionConditionedTransition(nn.Module):
    """Roll out a learned state transition conditioned on an action chunk."""

    state_dim: int = 7
    action_dim: int = 7
    action_horizon: int = 8
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, state: jnp.ndarray, action_chunk: jnp.ndarray) -> jnp.ndarray:
        state = jnp.asarray(state)
        action_chunk = jnp.asarray(action_chunk)
        if action_chunk.ndim == 2:
            action_chunk = action_chunk[:, None, :]
        if action_chunk.ndim != 3 or action_chunk.shape[-1] != self.action_dim:
            raise ValueError(f"action_chunk must have shape (B,H,{self.action_dim}), got {action_chunk.shape}")
        if not 1 <= action_chunk.shape[1] <= self.action_horizon:
            raise ValueError(f"expected 1 to {self.action_horizon} actions, got {action_chunk.shape[1]}")

        predicted = state
        for index in range(action_chunk.shape[1]):
            x = jnp.concatenate([predicted, action_chunk[:, index]], axis=-1)
            x = nn.gelu(nn.Dense(self.hidden_dim, name=f"transition_dense_{index}_0")(x))
            x = nn.gelu(nn.Dense(self.hidden_dim, name=f"transition_dense_{index}_1")(x))
            delta = nn.Dense(self.state_dim, name=f"transition_delta_{index}")(x)
            predicted = predicted + delta
        return predicted


def _safe_cosine_similarity(x: jnp.ndarray, y: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    numerator = jnp.sum(x * y, axis=-1)
    denominator = jnp.maximum(jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(y, axis=-1), eps)
    return numerator / denominator


def _wrapped_angle_difference(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    difference = x - y
    return (difference + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def rotation_distance(predicted: jnp.ndarray, policy: jnp.ndarray) -> jnp.ndarray:
    """Return an Euler-angle distance that is stable across ±pi wraparound."""

    return jnp.linalg.norm(_wrapped_angle_difference(predicted, policy), axis=-1)


def future_consistency(predicted_future_state: jnp.ndarray, policy_future_state: jnp.ndarray) -> jnp.ndarray:
    """Build state-wise and summary agreement features for actor/critic input."""

    predicted = jnp.asarray(predicted_future_state)
    policy = jnp.asarray(policy_future_state)
    if predicted.shape != policy.shape or predicted.shape[-1] < 7:
        raise ValueError(f"future states must match and have at least 7 values: {predicted.shape}, {policy.shape}")
    diff = predicted - policy
    position_error = jnp.linalg.norm(diff[..., :3], axis=-1)
    orientation_error = rotation_distance(predicted[..., 3:6], policy[..., 3:6])
    gripper_error = jnp.abs(diff[..., -1])
    summary = jnp.stack(
        [_safe_cosine_similarity(predicted, policy), position_error, orientation_error, gripper_error], axis=-1
    )
    return jnp.concatenate([diff, jnp.abs(diff), summary], axis=-1)


def mse(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error helper used by both pretraining and tests."""

    return jnp.mean(jnp.square(jnp.asarray(prediction) - jnp.asarray(target)))

"""Conservative offline critic and IQL/AWR residual objectives."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class OfflineRLConfig:
    discount: float = 0.99
    expectile: float = 0.7
    temperature: float = 1.0
    cql_alpha: float = 0.1
    max_advantage_weight: float = 20.0
    residual_penalty: float = 1e-3


def expectile_loss(value: jnp.ndarray, target_q: jnp.ndarray, expectile: float = 0.7) -> jnp.ndarray:
    difference = target_q - value
    weight = jnp.where(difference > 0, expectile, 1.0 - expectile)
    return jnp.mean(weight * jnp.square(difference))


def iql_advantage(q: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    return q - value


def conservative_critic_loss(
    data_q1: jnp.ndarray,
    data_q2: jnp.ndarray,
    target_q: jnp.ndarray,
    sampled_q1: jnp.ndarray,
    sampled_q2: jnp.ndarray,
    *,
    cql_alpha: float = 0.1,
) -> jnp.ndarray:
    """Twin regression plus a CQL log-sum-exp penalty on sampled actions."""

    regression = 0.5 * (jnp.mean(jnp.square(data_q1 - target_q)) + jnp.mean(jnp.square(data_q2 - target_q)))
    conservative = 0.5 * (
        jnp.mean(jnp.logsumexp(sampled_q1, axis=-1) - data_q1) + jnp.mean(jnp.logsumexp(sampled_q2, axis=-1) - data_q2)
    )
    return regression + cql_alpha * conservative


def iql_target(
    reward: jnp.ndarray, discount: jnp.ndarray, next_value: jnp.ndarray, config: OfflineRLConfig
) -> jnp.ndarray:
    return reward + config.discount * discount * next_value

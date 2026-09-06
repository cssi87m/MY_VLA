"""Project GROOT backbone features into the residual-policy latent space."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp


class LatentProjector(nn.Module):
    """Pool token features and project them to a stable residual-policy width."""

    output_dim: int = 256
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, hidden: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.asarray(hidden)
        if hidden.ndim < 2:
            raise ValueError(f"hidden must have at least two dimensions, got {hidden.shape}")
        if hidden.ndim > 2:
            hidden = jnp.mean(hidden, axis=tuple(range(1, hidden.ndim - 1)))
        x = nn.LayerNorm()(hidden)
        x = nn.gelu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.output_dim)(x)

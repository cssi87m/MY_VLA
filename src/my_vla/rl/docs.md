# `my_vla.rl`

Flax residual actor-critic building blocks and loss helpers for bounded,
offline repair of a base VLM action plan.

## Policy and critic

| Symbol | Contract |
| --- | --- |
| `build_actor_input()` | Concatenates equally ranked latent, flattened base-action tail, future-consistency, and retrieval-context arrays on the last axis. |
| `ResidualActor` | Two-layer ReLU MLP that predicts a bounded action-tail residual. Its final kernel is zero-initialized, so the initial policy preserves the base plan. |
| `TwinCritic` | Two independent two-layer Q functions returning `(q1, q2)` for conservative offline estimates. |
| `clip_libero_action()` | Clips seven-value canonical actions: first six values to `[-1, 1]`, gripper to `[0, 1]`. |

The actor emits `action_horizon * action_dim` values. It applies `tanh` and
scales each action step with `max_residual`; the default bounds are
`(0.02, 0.02, 0.02, 0.10, 0.10, 0.10, 1.0)` for translation, rotation, and
gripper respectively.

## Objectives

`advantage_weighted_residual_loss()` is an IQL/AWR-style weighted behavior
cloning objective. It exponentiates a temperature-scaled advantage (clipped
before exponentiation and capped by `max_weight`), then adds an L2 penalty on
the predicted residual magnitude.

`critic_loss(q1, q2, target)` regresses both critics to a stop-gradient target.
The training package provides the accompanying expectile, target, and CQL
helpers; see [`../training/docs.md`](../training/docs.md).

## Typical use

```python
actor_input = build_actor_input(latent, base_tail.reshape(batch_size, -1), consistency, context)
residual = actor.apply({"params": actor_params}, actor_input)
corrected = clip_libero_action(base_tail + residual.reshape(batch_size, tail_horizon, 7))
```

`base_tail` and its residual target must refer only to actions not yet
executed. The rollout policy executes the base prefix first, observes the
checkpoint state, and repairs this tail on the subsequent request.

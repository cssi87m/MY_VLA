# `my_vla.training`

Initialization and objectives for training the small residual heads on top of
a frozen base VLM. Script-level orchestration lives in
[`scripts/train_residual_libero.py`](../../../scripts/train_residual_libero.py).

## Pretraining

`PretrainConfig` defines the model and optimization contract. Defaults use an
eight-action base horizon, execute four base actions before replanning, and
therefore train a four-action correction tail:

```text
correction_horizon = action_horizon - replan_steps
```

`initialize_pretraining(rng, config)` constructs a `PretrainBundle`, Flax
parameter tree, and Optax optimizer state. It validates that
`1 <= replan_steps < action_horizon` and that `consistency_dim` equals
`2 * state_dim + 4` (18 for the canonical seven-dimensional state).

`pretrain_step(bundle, params, opt_state, batch)` performs one JAX/Optax
update. `make_pretrain_step(bundle)` returns its JIT-compiled form for the
fixed-size script training loop. Its combined loss has two parts:

1. Future-state loss: the action-conditioned transition model regresses to
   `batch["future_state"]`.
2. Residual behavior-cloning loss: the actor predicts
   `expert_action_chunk[replan_steps:] - base_action_tail` from projected VLM
   features, the base tail, checkpoint consistency, and retrieval context.

The checkpoint consistency is calculated against the actual state reached at
the replan checkpoint, rather than against another predicted state.

### Batch contract

| Key | Expected shape |
| --- | --- |
| `hidden` | `(B, ..., vlm_hidden_dim)`; extra axes are pooled by the heads. |
| `state`, `future_state` | `(B, state_dim)`; default `(B, 7)`. |
| `base_action_prefix` | `(B, replan_steps, action_dim)`. |
| `base_action_tail`, `residual_target` | `(B, correction_horizon, action_dim)`. |
| `retrieval_context` | `(B, retrieval_context_dim)`, containing padded retrieved expert tails, scores, returns, and a mask. |
| `sample_weight` | `(B,)`; masks duplicate padding used to keep the last batch at a fixed compiled shape. |

The training script loads an immutable expert memory bank, collects frozen
GROOT features and retrieval contexts, releases GROOT GPU memory, then trains
the JAX heads. Raw images and episode trajectories are not retained after
feature extraction. The
memory bank is passed separately to the rollout policy and must use the same
horizon and replan-step values as the checkpoint.

## Offline-RL utilities

`OfflineRLConfig` groups IQL/CQL hyperparameters: discount, expectile,
temperature, CQL coefficient, maximum advantage weight, and residual penalty.

| Function | Purpose |
| --- | --- |
| `expectile_loss(value, target_q)` | Asymmetric value regression used by IQL. |
| `iql_advantage(q, value)` | Computes `q - value`. |
| `iql_target(reward, discount, next_value, config)` | Computes the discounted bootstrap target. |
| `conservative_critic_loss(...)` | Twin Q regression plus a CQL log-sum-exp penalty over sampled actions. |

These helpers are independent from the stage-1–3 behavior-cloning update;
they are available when extending the pipeline with a conservative offline-RL
fine-tuning stage.

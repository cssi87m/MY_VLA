# `my_vla.models`

Model adapters and Flax heads that turn a GROOT N1.5 base-policy prediction
into features usable by residual planning. GROOT and PyTorch are optional for
the small JAX heads, but required when running the real base VLM adapter.

## Components

| Module | Main symbols | Role |
| --- | --- | --- |
| `base_vlm.py` | `BaseVLM`, `BaseVLMOutput`, `GrootN15Adapter` | Defines the small base-policy protocol and adapts GROOT output to an action chunk plus backbone features. |
| `projector.py` | `LatentProjector` | Mean-pools non-batch token axes, layer-normalizes, and projects hidden features to the residual latent width. |
| `future_state.py` | `FutureStateHead`, `ActionConditionedTransition`, `future_consistency` | Predicts the horizon-end state, rolls a state through an action prefix, and creates agreement features. |
| `groot_libero_config.py` | `LiberoDataConfig` | GROOT data-config entry point for two RGB views, canonical state/action fields, and an eight-action horizon. |
| `residual_libero_rollout.py` | `ResidualLiberoRolloutPolicy` | Loads a residual checkpoint and alternates base-prefix execution with tail correction. |

## Base-policy contract

A `BaseVLM` is callable as `base_vlm(observation, instruction)` and returns
`BaseVLMOutput`:

- `base_action`: an `(H, 7)` action chunk in canonical LIBERO action order.
- `hidden`: GROOT backbone features. The residual heads accept a batch axis;
  extra token axes are mean-pooled.

`GrootN15Adapter` builds the flattened modality mapping GROOT expects from
`image`, `wrist_image`, canonical `(7,)` `state`, and the language
instruction. It retains GROOT's transform and inverse-transform paths, and
installs a narrow Eagle dtype compatibility hook when that model structure is
present.

## State prediction and consistency

`FutureStateHead(hidden)` produces `(B, 7)` horizon-end states.
`ActionConditionedTransition(state, action_chunk)` iteratively predicts the
state reached after one to `action_horizon` actions. It accepts `(B, H, 7)`
actions (or `(B, 7)` as a one-action chunk).

`future_consistency(predicted, policy)` returns 18 features for seven-value
states: signed difference, absolute difference, then cosine similarity,
position error, wrap-safe Euler-angle error, and gripper error. This is the
default `PretrainConfig.consistency_dim`.

## Online residual rollout

`ResidualLiberoRolloutPolicy.from_checkpoint()` expects a checkpoint directory
created by `scripts/train_residual_libero.py`:

```text
config.json
params.msgpack
retrieval_bank.npz
retrieval_bank.json
```

The first `infer()` call for a plan runs GROOT and returns the first
`replan_steps` base actions (`residual_phase="base_prefix"`). The next call
uses the observed checkpoint state to compare against the transition model,
then returns a corrected remaining tail (`residual_phase="corrected_tail"`)
without invoking GROOT again. Set `reset_residual_plan` in a request to discard
the cached prefix plan.

Serving requests use `observation.base_0_rgb`,
`observation.left_wrist_0_rgb`, `observation.state`, and `prompt`. State may
be canonical `(7,)` or LAP `(10,)` (`xyz + rot6d + gripper`), which is
converted by `lap_rot6d_state_to_libero_state()`.

## Shape guardrails

The canonical state/action dimension is seven. `LatentProjector` requires at
least batch and feature axes. The rollout policy pads short GROOT chunks by
repeating their final action and clips final actions to pose `[-1, 1]` and
gripper `[0, 1]` through the RL utility.

# LIBERO residual VLA architecture

This design combines a base VLM action policy, a future-state consistency signal, retrieval, and a residual actor--critic.

## Decision flow

```text
(main image, wrist image, state, instruction)
                         |
                         v
                       VLM
                         |
       +-----------------+------------------+
       |                                    |
       v                                    v
base action head: a_base,t:t+H   future-state head: s_hat,t+H
       |                                    |
       |                    transition / policy rollout model
       |                              predicts s_policy,t+H
       |                                    |
       +------------------+-----------------+
                          v
     future consistency c_future = F(s_hat,t+H, s_policy,t+H)
                          |
                          v
retrieval context + VLM latent + a_base + c_future -> residual actor
                          |
                          v
              delta_a = pi(z_t, a_base, c_future, context)
              a_final = clip(a_base + delta_a)
```

`s_hat,t+H` is the state predicted by the VLM's future-state head. `s_policy,t+H` is the future state predicted by rolling out the base action through an action-conditioned transition model (or a simulator when one is available). Their agreement is an explicit input to the residual policy.

## Causality

The physical state observed at `t + H` cannot correct the action selected at `t`. Therefore use one of these modes:

- **Immediate residual correction:** set `s_policy,t+H` to a learned dynamics or simulator rollout before taking the action.
- **Receding-horizon correction:** after executing the chunk, compare the observed state at `t + H` with `s_hat,t+H`, then use this mismatch at the next replanning point.

For the first implementation, use a learned transition model so the residual can act immediately, then validate it against actual future LIBERO states during training and evaluation.

## Future consistency feature

Do not reduce the comparison to cosine similarity alone. Provide the residual actor with state-wise error, absolute error, and scalar summaries.

```python
def future_consistency(predicted_future_state, policy_future_state):
    diff = predicted_future_state - policy_future_state
    return jnp.concatenate([
        diff,
        jnp.abs(diff),
        jnp.array([
            cosine_similarity(predicted_future_state, policy_future_state),
            jnp.linalg.norm(diff[:3]),              # position mismatch
            rotation_distance(
                predicted_future_state[3:6],
                policy_future_state[3:6],
            ),
            jnp.abs(diff[-1]),                       # gripper mismatch
        ]),
    ], axis=-1)
```

All state values must use one convention. In particular, normalize or convert LIBERO's axis-angle orientation before calculating differences.

## Residual actor input

```python
actor_input = jnp.concatenate([
    z_t,                 # projected VLM hidden state
    a_base,              # current action or action chunk from action head
    c_future,            # predicted-vs-policy future-state consistency
    retrieval_context,   # aggregated top-k demonstrations
], axis=-1)

delta_action = residual_actor(actor_input)
final_action = clip_libero_action(a_base + delta_action)
```

Bound the residual initially so it cannot override the base policy:

```python
class ResidualPolicy(nn.Module):
    action_dim: int = 7
    max_residual: tuple[float, ...] = (
        0.02, 0.02, 0.02, 0.10, 0.10, 0.10, 1.0,
    )

    @nn.compact
    def __call__(self, actor_input):
        x = nn.relu(nn.Dense(512)(actor_input))
        x = nn.relu(nn.Dense(512)(x))
        return jnp.tanh(nn.Dense(self.action_dim)(x)) * jnp.asarray(self.max_residual)
```

The base action and residual must be in the same LIBERO end-effector coordinate frame before addition.

## Checkpoint residual training targets

Choose a checkpoint ``K`` with ``1 <= K < H``. GR00T plans an ``H``-action
chunk at time ``t``; its first ``K`` actions form the base prefix and the
residual repairs the unexecuted tail. The residual does not call GR00T at the
checkpoint. It uses the cached plan plus the actual checkpoint observation.

Use the RLDS LIBERO dataset at:

```text
/home/vrh3/workspace/vla/lap/data/openvla-modified_libero_rlds
```

For each timestep where `t + H` is valid:

```python
base_chunk, h = base_vlm(obs_t, instruction)
z_t = projector(h)
s_hat_checkpoint = future_state_head(h)
s_policy_checkpoint = transition_model(state_t, base_chunk[:K])
checkpoint_error = future_consistency(state_t_plus_K, s_policy_checkpoint)

residual_target = expert_action_chunk[K:H] - base_chunk[K:H]
future_loss = mse(s_hat_checkpoint, state_t_plus_K) + mse(s_policy_checkpoint, state_t_plus_K)
residual_loss = mse(residual_actor(z_t, base_chunk[K:H], checkpoint_error, context), residual_target)
```

Train in stages:

1. Train or freeze the base VLM and train the future-state and transition heads against the demonstration future state.
2. Build the retrieval index from demonstration embeddings, actions, residual targets, and returns.
3. Train the residual actor by behavior cloning of the remaining expert tail
   minus the remaining base-plan tail.
4. Train a conservative offline critic conditioned on `z_t`, action, future consistency, and retrieval context.
5. Use IQL/AWR-style advantage-weighted residual updates with a penalty for large residuals; then validate in LIBERO.

## Retrieval records

Store one record per timestep or short action chunk:

```python
{
    "key": concat(projected_hidden, projected_state, language_embedding),
    "state": state,
    "instruction": instruction,
    "base_action": base_action,
    "expert_action": expert_action,
    "residual_target": expert_action - base_action,
    "future_state": state_t_plus_h,
    "return_to_go": return_to_go,
}
```

Retrieve top-k records and aggregate them to a fixed-size context vector for the actor and critic.

## Implementation modules

```text
src/my_vla/
  data/libero.py          # RLDS trajectories and t -> t+H transitions
  models/base_vlm.py      # adapter for the selected VLM
  models/projector.py     # VLM hidden state -> z_t
  models/future_state.py  # VLM state head and action-conditioned transition model
  retrieval/bank.py       # index and top-k aggregation
  rl/residual_ac.py       # residual actor and twin critic
  training/pretrain.py    # state/dynamics and residual BC initialization
  training/offline_rl.py  # conservative offline RL updates
scripts/train_residual_libero.py
scripts/serve_residual_libero.py
scripts/libero/main.py
```

Use JAX/Flax/Optax to stay compatible with LAP. Reuse LAP's LIBERO transforms as the source of truth for gripper and rotation conventions: `lap/src/lap/datasets/utils/transforms.py`.

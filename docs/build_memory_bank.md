# Building the Residual Memory Bank

This document describes a memory bank for repairing the unexecuted tail of a
GR00T action plan in LIBERO.

## Goal

At time `t`, GR00T predicts an action chunk of length `H`:

```text
[a0, a1, ..., aH-1]
```

The robot executes the first `K = replan_steps` actions unchanged. At `t + K`,
the residual controller observes the real current state and corrects only the
remaining tail:

```text
base tail:      [aK, ..., aH-1]
corrected tail: [aK', ..., aH-1']
```

The memory bank supplies examples of expert action tails that were successful
in similar checkpoint situations.

## What the bank represents

One record answers two questions:

```text
Key:     What situation was the robot in when correction started?
Payload: What expert actions successfully completed the remaining task?
```

The bank is a database of demonstrations. It is not initially built from the
residual controller's predictions.

## Build the initial expert bank

Build this bank once from RLDS expert demonstrations. For each episode and each
valid starting timestep `t`:

1. Run frozen GR00T on the expert observation at `t` and obtain an `H`-action
   base plan.
2. Select its tail from `K` to `H`.
3. Read the recorded demonstration checkpoint at `t + K`.
4. Save the expert action tail beginning at `t + K`.
5. Write a record for that checkpoint.

The recorded state at `t + K` is a real state from the expert trajectory. It is
not the transition model's prediction.

Run the builder from the repository root:

```bash
PYTHONPATH=$PWD/src python -m scripts.build_memory_bank \
  --groot-model-path "$GROOT_CHECKPOINT" \
  --data-root /path/to/openvla-modified_libero_rlds \
  --output checkpoints/expert_memory_bank \
  --horizon 8 \
  --replan-steps 4
```

It writes `expert_memory_bank.npz`, `expert_memory_bank.json`, and a
`manifest.json` describing the bank version and build configuration.

```text
expert observation at t
  -> frozen GR00T plan [a0, ..., aH-1]
  -> base tail [aK, ..., aH-1]

recorded expert checkpoint at t+K
  -> retrieval key

recorded expert actions from t+K
  -> expert-tail payload
```

## Record contents

Each record should contain:

```text
key:
  checkpoint robot state at t+K
  instruction embedding
  optional checkpoint visual embedding

payload:
  expert tail [eK, ..., eH-1]
  GR00T base tail [aK, ..., aH-1]
  residual target = expert tail - base tail
  checkpoint return-to-go, tail reward, tail discount, or success label
  episode ID and timestep
  source = "expert"
```

The first version can use this key without another GR00T call:

```text
checkpoint robot state + instruction embedding
```

The RLDS dataset already contains the real robot state and images at `t + K`.
Only a new visual-language latent for those images requires another VLM or
vision-encoder forward pass.

## Inference flow

At inference, the system must query the bank at the same point represented by
its records: after the base prefix has executed.

```text
1. At t, call GR00T once and cache its H-action plan.
2. Execute [a0, ..., aK-1].
3. Receive the real simulator observation at t+K.
4. Build a checkpoint retrieval key.
5. Retrieve the k most similar expert records.
6. Extract their expert tails and similarity scores.
7. Give the residual actor:
     - the real checkpoint state,
     - the cached GR00T base tail,
     - retrieved expert tails,
     - retrieval scores and quality values.
8. Return a corrected tail [aK', ..., aH-1'].
```

GR00T supplies the base plan. The bank supplies similar examples of successful
expert tails. The residual actor decides how much to use each retrieved example
to correct the current base tail.

## How to use retrieved actions

Do not initially decode a compressed action latent. Store the complete expert
tail directly, then provide the retrieved tails to the residual actor.

```text
current checkpoint + current base tail + retrieved expert tails
  -> residual actor
  -> corrected current tail
```

A direct weighted average of retrieved actions is possible, but can be unsafe
when the retrieved examples describe different object positions. Letting the
actor combine the candidates is more flexible.

## Avoid leakage during training

When training on a demonstration record, do not allow it to retrieve itself.
Otherwise its exact expert action tail can enter the actor input and make
training loss unrealistically low.

At minimum, exclude the same record. Prefer excluding the entire source
episode, since adjacent timesteps have nearly identical observations and action
tails.

Use separate demonstration splits for bank construction, model training, and
evaluation whenever possible.

## Expert state versus GR00T state

The initial bank contains real checkpoint states reached by expert prefixes:

```text
expert prefix -> real expert checkpoint state
```

During live inference, the checkpoint comes from executing GR00T's prefix:

```text
GR00T prefix -> real simulator checkpoint state
```

Both are real states, but they may differ. This is the distribution gap the
residual controller must handle.

## Later: add verified rollout memory

Do not add trajectories from the untrained residual controller to the trusted
expert bank. Early rollouts may be poor and would teach retrieval to return bad
actions.

After the residual policy has been trained, collect LIBERO rollouts and store
only verified high-quality records in a separate rollout bank:

```text
accepted rollout record:
  task succeeded, or
  return exceeds a chosen quality threshold
```

Keep the source explicit:

```text
expert bank:  immutable, trusted demonstrations
rollout bank: successful residual-policy trajectories
```

The rollout bank can improve coverage of states that GR00T actually reaches,
while the expert bank remains a stable source of high-quality actions.

## Reuse for offline RL

The bank can also support offline-RL fine-tuning. Store the state, action tail,
next/checkpoint state, reward or return, and source metadata needed by the
critic. Keep retrieval memory versioned so a fine-tuning run records exactly
which expert and rollout data it used.

## Implemented integration

`scripts/train_residual_libero.py` now loads the fixed artifact through
`--memory-bank`, creates queries from `checkpoint_state`, excludes the source
episode, and trains on padded top-K expert-tail candidates. The rollout policy
loads the same bank through `scripts/serve_residual_libero.py --memory-bank`
and queries it only after the base prefix executes.

Train a new residual checkpoint after building the bank:

```bash
PYTHONPATH=$PWD/src python -m scripts.train_residual_libero \
  --groot-model-path "$GROOT_CHECKPOINT" \
  --memory-bank checkpoints/expert_memory_bank \
  --output checkpoints/residual_libero \
  --horizon 8 \
  --replan-steps 4 \
  --retrieval-k 8
```

The remainder of this section records the module-level implementation design.

### 1. Return action-tail candidates from `RetrievalBank`

**File:** `src/my_vla/retrieval/bank.py`

Keep `aggregate()` for compatibility, but add a fixed-size candidate API. The
actor needs individual expert tails and scores, not only their weighted mean.

```python
@dataclasses.dataclass(frozen=True)
class RetrievedTails:
    expert_actions: np.ndarray  # (retrieval_k, correction_horizon * action_dim)
    scores: np.ndarray          # (retrieval_k,)
    returns_to_go: np.ndarray   # (retrieval_k,)
    mask: np.ndarray            # (retrieval_k,), 1 for a real record


def retrieve_tails(
    self,
    query_key: np.ndarray,
    *,
    retrieval_k: int,
    exclude_episode_id: str | None = None,
) -> RetrievedTails:
    eligible = np.ones(len(self.records), dtype=bool)
    if exclude_episode_id is not None:
        eligible &= np.asarray([record.episode_id != exclude_episode_id for record in self.records])
    # Score only eligible records, select top retrieval_k, then zero-pad.
```

The method should use `record.expert_action`, not the averaged `_contexts`
array. It must pad results so every actor input has the same static shape.

Add a helper used by both training and serving:

```python
def tail_candidate_features(result: RetrievedTails) -> np.ndarray:
    return np.concatenate(
        [
            result.expert_actions.reshape(-1),
            result.scores,
            result.returns_to_go,
            result.mask,
        ]
    ).astype(np.float32)
```

### 2. Add bank settings to the training configuration

**File:** `scripts/train_residual_libero.py`

Extend `TrainConfig` and `_parse_args()`:

```python
memory_bank: Path = Path("checkpoints/expert_memory_bank")
retrieval_k: int = 8
```

```python
parser.add_argument("--memory-bank", type=Path, required=True)
parser.add_argument("--retrieval-k", type=int, default=8)
```

Replace the temporary bank construction in `ResidualLiberoTrainer.run()`:

```python
# Old
records, samples = LiberoSampleCollector(self.config, adapter).collect()
bank = RetrievalBank(records=records)
bank.save(self.config.output / "retrieval_bank")

# New
samples = LiberoSampleCollector(self.config, adapter).collect_features()
bank = RetrievalBank.load(self.config.memory_bank / "expert_memory_bank")
self._validate_memory_bank(bank)
```

`collect_features()` should retain only the values that are generated from the
current GR00T call and are needed for training:

```python
{
    **sample,
    "hidden": pooled_hidden,
    "base_action_prefix": base_chunk[:args.replan_steps],
    "base_action_tail": base_chunk[args.replan_steps:],
    "checkpoint_state": sample["state_chunk"][args.replan_steps],
}
```

It must no longer build a retrieval key from `hidden + state_t`. The bank key is
created at the checkpoint instead.

### 3. Build training batches from checkpoint retrieval

**File:** `scripts/train_residual_libero.py`

Change `_make_batch()` to query the fixed expert bank for every training sample:

```python
def _make_batch(self, batch_samples: list[dict], bank: RetrievalBank) -> dict:
    candidates = []
    for sample in batch_samples:
        key = checkpoint_retrieval_key(sample["checkpoint_state"], sample["instruction"])
        result = bank.retrieve_tails(
            key,
            retrieval_k=self.config.retrieval_k,
            exclude_episode_id=sample["episode_id"],
        )
        candidates.append(tail_candidate_features(result))

    return {
        "hidden": jnp.asarray(np.stack([sample["hidden"] for sample in batch_samples])),
        "state": jnp.asarray(np.stack([sample["state"] for sample in batch_samples])),
        "base_action_prefix": jnp.asarray(np.stack([sample["base_action_prefix"] for sample in batch_samples])),
        "base_action_tail": jnp.asarray(np.stack([sample["base_action_tail"] for sample in batch_samples])),
        "future_state": jnp.asarray(np.stack([sample["checkpoint_state"] for sample in batch_samples])),
        "retrieval_context": jnp.asarray(np.stack(candidates)),
        "residual_target": jnp.asarray(np.stack([
            sample["expert_action_chunk"][self.config.replan_steps:] - sample["base_action_tail"]
            for sample in batch_samples
        ])),
    }
```

This keeps the existing `pretrain.py` batch field name, but changes its contents
from an averaged record context to explicit retrieved action-tail features.

### 4. Make the actor input dimension explicit

**Files:** `src/my_vla/training/pretrain.py`, `src/my_vla/rl/residual_ac.py`

Add the retrieval setting to `PretrainConfig` and derive the static feature
width from it:

```python
retrieval_k: int = 8

@property
def retrieval_context_dim(self) -> int:
    tail_dim = self.correction_horizon * self.action_dim
    return self.retrieval_k * tail_dim + 3 * self.retrieval_k
```

Replace the existing `context_dim` field with this derived property. When
creating the pretraining configuration, replace:

```python
context_dim=bank.context_dim,
```

with:

```python
retrieval_k=args.retrieval_k,
```

Update `initialize_pretraining()` to use `config.retrieval_context_dim` when
creating its representative `context` array. The actor can keep using `build_actor_input()`, because
`retrieval_context` remains a single flattened `(B, D)` array. Its contents now
include expert-tail candidates, scores, returns, and a validity mask.

This changes the actor's input shape. Existing residual checkpoints cannot be
loaded after this change and must be retrained.

### 5. Query the bank only at the correction checkpoint

**File:** `src/my_vla/models/residual_libero_rollout.py`

The first request should cache GR00T's plan but must not retrieve yet:

```python
if self._cached_plan is None:
    output = self.base_vlm(...)
    base_chunk = self._base_chunk(output.base_action)
    prefix = base_chunk[:, :self.config.replan_steps]
    tail = base_chunk[:, self.config.replan_steps:]
    expected_checkpoint = self.transition.apply(...)
    self._cached_plan = _CachedPlan(
        tail=tail,
        expected_checkpoint=expected_checkpoint,
        latent=latent,
        instruction=instruction,
    )
    return {"actions": np.asarray(clip_libero_action(prefix[0])), ...}
```

Add `instruction: str` to `_CachedPlan`. On the second request, use the real
state sent by the simulator after the prefix has executed:

```python
actual_checkpoint = jnp.asarray(state)[None]
key = checkpoint_retrieval_key(state, cached.instruction)
retrieved = self.retrieval_bank.retrieve_tails(
    key,
    retrieval_k=self.config.retrieval_k,
)
retrieval_context = jnp.asarray(tail_candidate_features(retrieved)[None])

consistency = future_consistency(actual_checkpoint, cached.expected_checkpoint)
actor_input = build_actor_input(
    cached.latent,
    cached.tail.reshape(1, -1),
    consistency,
    retrieval_context,
)
residual = self.actor.apply({"params": self.params["actor"]}, actor_input)[0]
```

The simulator client in `scripts/libero/main.py` already makes this second
request after consuming the base prefix, so it needs no control-flow change.

### 6. Load and validate the artifact in the server

**Files:** `scripts/serve_residual_libero.py`,
`src/my_vla/models/residual_libero_rollout.py`

Make the bank path explicit instead of silently loading the older
`checkpoint/retrieval_bank` directory:

```python
@dataclasses.dataclass(frozen=True)
class Args:
    groot_model_path: str
    checkpoint: Path
    memory_bank: Path
```

```python
policy = ResidualLiberoRolloutPolicy.from_checkpoint(
    args.checkpoint,
    memory_bank=args.memory_bank / "expert_memory_bank",
    groot_model_path=args.groot_model_path,
    ...,
)
```

Before constructing the policy, read `memory_bank/manifest.json` and reject a
bank whose `horizon` or `replan_steps` differs from the model checkpoint. Also
validate the loaded bank's `key_dim` and the actor's expected
`retrieval_context_dim`.

### 7. Prepare the bank for offline RL

**Files:** `src/my_vla/data/libero.py`, `src/my_vla/retrieval/bank.py`, new
`scripts/finetune_residual_offline_rl.py`

The current `offline_rl.py` contains loss functions only. To train from bank
records, add checkpoint-time reward and discount fields when transitions are
created:

```python
# In iter_libero_transitions()
"checkpoint_return_to_go": arrays["returns"][timestep + config.replan_steps],
"tail_reward": arrays["rewards"][timestep + config.replan_steps],
"tail_discount": arrays["discounts"][timestep + config.replan_steps],
```

Store those fields in `RetrievalRecord`. The future offline-RL script can then
load the bank and construct transitions such as:

```python
state = record.state                 # state at t+K
action = record.expert_action        # expert tail
next_state = record.future_state     # state after the tail
reward = record.tail_reward
discount = record.tail_discount
```

Use the same retrieval candidate function as residual pretraining, with
episode exclusion. Fine-tune only after the supervised residual model is
stable.

### 8. Add regression tests before each stage

**File:** `tests/test_scripts.py` and new focused bank tests.

Add checks for:

```text
- builder key uses state_chunk[replan_steps], not state at t
- record payload contains the flattened expert tail
- retrieval excludes every record from the requested episode
- fewer than retrieval_k eligible records produces correctly padded arrays
- trainer passes checkpoint-key candidates into the batch
- server does not query before returning the base prefix
- server queries exactly once on the correction request
- manifest mismatch fails before inference
```

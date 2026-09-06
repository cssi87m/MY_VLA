# `my_vla.retrieval`

A small, dependency-light exact nearest-neighbour bank for demonstration
features. It provides residual-policy context without an external vector
database.

## Records and keys

`RetrievalRecord` stores one demonstration's retrieval key, canonical state,
instruction, flattened base/expert action tails, residual target, future
state, and return-to-go. `RetrievalRecord.from_sample(sample, key,
base_action)` constructs a record and defines the target as:

```text
residual_target = expert_action - base_action
```

The expert memory bank builds keys from the canonical checkpoint state at
`t + replan_steps` plus `hash_text_embedding(instruction)`. The latter is
deterministic and dependency-free, intended as a reproducible fallback—not a
semantic language encoder.

## `RetrievalBank`

| Method / property | Behavior |
| --- | --- |
| `RetrievalBank(records=...)` | Builds an in-memory bank; if keys are omitted, takes each record's key. |
| `add(record)` | Appends one record, enforcing a consistent key width. |
| `query(query_key, k=8)` | Returns up to `k` record indices and cosine scores in descending order using partial top-k selection. |
| `aggregate(query_key, k=8)` | Softmax-weights retrieved contexts and returns one fixed-width `float32` vector. |
| `retrieve_tails(query_key, retrieval_k, exclude_episode_id=...)` | Returns padded expert action-tail candidates, scores, returns, and a mask. |
| `key_dim` / `context_dim` | Expose the key and aggregated-context widths. |
| `save(path)` / `load(path)` | Persist and restore the bank as paired `.npz` arrays and `.json` metadata. |

Each per-record context is the concatenation:

```text
state + base_action + expert_action + residual_target + future_state + return_to_go
```

For the default four-action correction tail, this is
`7 + 28 + 28 + 28 + 7 + 1 = 99` values. The actual width depends on the
action horizon/replan split. The residual actor instead receives individual
candidate tails through `PretrainConfig.retrieval_context_dim`.

An empty bank has context width zero and `aggregate()` returns a zero-length
vector; a non-empty rollout bank must match the checkpoint's context width.

## Persistence

`bank.save(Path("checkpoints/expert_memory_bank/expert_memory_bank"))` creates:

```text
checkpoints/expert_memory_bank/expert_memory_bank.npz   # keys and derived context matrix
checkpoints/expert_memory_bank/expert_memory_bank.json  # textual record metadata
```

The NPZ contains keys, derived contexts, and all numeric record columns; JSON
contains only instructions, episode IDs, and sources. New banks use an
uncompressed NPZ to favor load latency. `load()` remains compatible with older
banks whose numeric metadata is in JSON, reconstructs records, and verifies
that the derived contexts equal the stored context matrix. This avoids pickle
and catches mismatched metadata.

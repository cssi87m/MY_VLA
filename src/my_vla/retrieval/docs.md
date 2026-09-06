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

Training and rollout build keys by concatenating pooled base-VLM hidden
features, canonical state, and `hash_text_embedding(instruction)`. The latter
is deterministic and dependency-free, intended as a reproducible fallback—not
a semantic language encoder.

## `RetrievalBank`

| Method / property | Behavior |
| --- | --- |
| `RetrievalBank(records=...)` | Builds an in-memory bank; if keys are omitted, takes each record's key. |
| `add(record)` | Appends one record, enforcing a consistent key width. |
| `query(query_key, k=8)` | Returns up to `k` record indices and cosine scores in descending, stable order. |
| `aggregate(query_key, k=8)` | Softmax-weights retrieved contexts and returns one fixed-width `float32` vector. |
| `key_dim` / `context_dim` | Expose the key and aggregated-context widths. |
| `save(path)` / `load(path)` | Persist and restore the bank as paired `.npz` arrays and `.json` metadata. |

Each per-record context is the concatenation:

```text
state + base_action + expert_action + residual_target + future_state + return_to_go
```

For the default four-action correction tail, this is
`7 + 28 + 28 + 28 + 7 + 1 = 99` values. The actual width depends on the
action horizon/replan split and is saved in the training checkpoint's
`PretrainConfig.context_dim`.

An empty bank has context width zero and `aggregate()` returns a zero-length
vector; a non-empty rollout bank must match the checkpoint's context width.

## Persistence

`bank.save(Path("checkpoint/retrieval_bank"))` creates:

```text
checkpoint/retrieval_bank.npz   # keys and derived context matrix
checkpoint/retrieval_bank.json  # record metadata
```

`load()` reconstructs records and verifies that the derived contexts equal the
stored context matrix. This avoids pickle and catches mismatched metadata.

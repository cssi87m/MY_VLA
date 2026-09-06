# Residual VLA for LIBERO

This repository adds a lightweight residual controller to a frozen NVIDIA
GR00T N1.5 visual-language-action policy for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
benchmark.

GR00T proposes an action chunk. The controller executes a short prefix,
observes the real checkpoint state, retrieves similar expert action tails from
a demonstration bank, and predicts a bounded correction for the remaining
actions. The result is a receding-horizon policy that keeps GR00T as the base
policy while adapting its unexecuted plan tail to the observed state.

```text
observation + language
        │
        ▼
 frozen GR00T ──► base action chunk [a0, ..., aH-1]
        │                         │
        │                         └── execute prefix [a0, ..., aK-1]
        ▼
 projected latent                         │
                                          ▼
                         real checkpoint state at t + K
                                          │
                     retrieval bank ─────┤
                                          ▼
                        residual actor corrects [aK, ..., aH-1]
                                          │
                                          ▼
                           corrected action tail
```

## Repository layout

```text
src/my_vla/
  data/libero.py                    RLDS loading and LIBERO state/action transforms
  models/base_vlm.py                GR00T N1.5 adapter
  models/projector.py               frozen GR00T feature projection
  models/future_state.py            prefix-conditioned transition model
  models/residual_libero_rollout.py online two-phase residual policy
  retrieval/bank.py                 expert-tail retrieval bank
  rl/residual_ac.py                 bounded residual actor and offline-RL helpers
  training/pretrain.py              compiled residual behavior-cloning update

scripts/
  build_memory_bank.py              build an expert checkpoint memory bank
  train_residual_libero.py          extract features and train residual heads
  serve_residual_libero.py          serve the policy over OpenPI WebSocket RPC
  libero/main.py                    run LIBERO evaluation
```

## Requirements

Use an environment that provides:

- Python 3, NumPy, JAX, Flax, and Optax
- PyTorch and NVIDIA GR00T N1.5
- TensorFlow Datasets with the modified LIBERO RLDS data
- OpenPI for WebSocket serving and the LIBERO simulator dependencies

The project uses a `src/` layout but does not currently provide a root package
installer. Run commands from the repository root with:

```bash
export PYTHONPATH=$PWD/src
```

Initialize the simulator submodules and follow the simulator environment setup
in [scripts/libero/README.md](scripts/libero/README.md):

```bash
git submodule update --init --recursive
```

## End-to-end workflow

Choose a horizon `H` and a prefix length `K` such that `1 <= K < H`. The
default configuration uses `H=8` and `K=4`.

### 1. Build the expert memory bank

The bank is keyed by the real demonstration state at `t + K` plus a
deterministic instruction embedding. Its payload contains the expert action
tail and retrieval metadata.

```bash
PYTHONPATH=$PWD/src python -m scripts.build_memory_bank \
  --groot-model-path "$GROOT_CHECKPOINT" \
  --data-root /path/to/openvla-modified_libero_rlds \
  --output checkpoints/expert_memory_bank \
  --horizon 8 \
  --replan-steps 4
```

This creates:

```text
checkpoints/expert_memory_bank/
  expert_memory_bank.npz
  expert_memory_bank.json
  manifest.json
```

### 2. Train residual heads

Training runs GR00T in inference mode to extract compact features, computes
retrieval context once per sample, releases GR00T GPU memory, and then runs a
JIT-compiled JAX update for the residual heads.

```bash
PYTHONPATH=$PWD/src XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m scripts.train_residual_libero \
  --groot-model-path "$GROOT_CHECKPOINT" \
  --data-root /path/to/openvla-modified_libero_rlds \
  --memory-bank checkpoints/expert_memory_bank \
  --output checkpoints/residual_libero \
  --horizon 8 \
  --replan-steps 4 \
  --batch-size 16 \
  --epochs 10
```

The output directory contains `params.msgpack` and `config.json`. The memory
bank and checkpoint must use the same horizon and replan-step values.

### 3. Serve the policy

```bash
PYTHONPATH=$PWD/src XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m scripts.serve_residual_libero \
  --groot-model-path "$GROOT_CHECKPOINT" \
  --checkpoint checkpoints/residual_libero \
  --memory-bank checkpoints/expert_memory_bank \
  --port 11004
```

### 4. Evaluate in LIBERO

In a second terminal, activate the LIBERO environment configured in
[`scripts/libero/README.md`](scripts/libero/README.md), then run:

```bash
export LIBERO_CONFIG_PATH=$PWD/third_party/openpi/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/openpi/third_party/libero

python scripts/libero/main.py \
  --host 127.0.0.1 \
  --port 11004 \
  --policy-type RESIDUAL \
  --replan-steps 4
```

The evaluator's `--replan-steps` must equal the value stored in the residual
checkpoint.

## Runtime behavior

Each residual plan has two calls:

1. The first request calls GR00T and returns the base prefix.
2. After the prefix executes, the next request retrieves expert tails using the
   real checkpoint state and returns the corrected tail without another GR00T
   forward pass.

Send `reset_residual_plan=true` at the start of every new episode so a cached
plan cannot cross episode boundaries.

## Performance-oriented implementation details

- GR00T hidden tokens are pooled on the Torch device before CPU transfer.
- Training retains compact numeric samples, not raw image observations.
- Retrieval context is computed once during feature collection.
- The JAX update is compiled once; a masked duplicate pads a partial final
  batch so it has the same static shape.
- Retrieval uses cosine top-k partial selection rather than a full sort.
- New memory-bank artifacts store numeric metadata in NPZ and only text fields
  in JSON. Existing bank artifacts remain readable.

## Tests

The orchestration tests stub optional model and simulator dependencies:

```bash
python3 -m pytest -q tests/test_scripts.py
```

They validate data flow and lifecycle behavior, not GPU numerical correctness
or live simulator performance.

## Further documentation

- [Architecture notes](docs/libero_residual_vla_architecture.md)
- [Memory-bank design and build details](docs/build_memory_bank.md)
- [Script reference](scripts/README.md)
- [Retrieval-bank reference](src/my_vla/retrieval/docs.md)
- [LIBERO environment and evaluation setup](scripts/libero/README.md)

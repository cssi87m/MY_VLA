# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## Installation

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 scripts/libero/.venv
source scripts/libero/.venv/bin/activate
uv pip sync scripts/libero/requirements.txt third_party/openpi/third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e third_party/openpi/packages/openpi-client
uv pip install -e third_party/openpi/third_party/libero
```


## Evaluation

```bash
# in one terminal, run the residual server.  ``--replan-steps`` is selected
# during residual training and saved in its checkpoint (4 in this example).
PYTHONPATH=$PWD/src XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  python -m scripts.serve_residual_libero \
  --groot-model-path "$GROOT_CHECKPOINT" \
  --checkpoint checkpoints/residual_libero \
  --memory-bank checkpoints/expert_memory_bank \
  --port 11004


# in another terminal, run the sim
source $PWD/scripts/libero/.venv/bin/activate

export LIBERO_CONFIG_PATH=$PWD/third_party/openpi/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/openpi/third_party/libero

python scripts/libero/main.py --host 127.0.0.1 --port 11004 --policy-type RESIDUAL --replan-steps 4
```

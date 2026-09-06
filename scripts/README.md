# Script entry points

The scripts use application classes to own configuration and execution stages.
Existing command-line flags, defaults, checkpoint files, and result schemas are
preserved. Run residual commands from the repository root with `PYTHONPATH=$PWD/src`,
using the environment containing GR00T, JAX/Flax, and OpenPI.

| Script | Classes | Responsibilities |
| --- | --- | --- |
| `build_memory_bank.py` | `MemoryBankBuildConfig`, `ExpertMemoryBankBuilder` | Build immutable expert checkpoint records for retrieval and offline RL |
| `train_residual_libero.py` | `TrainConfig`, `LiberoSampleCollector`, `ResidualLiberoTrainer` | Collect features, construct batches, train heads, save checkpoints |
| `serve_residual_libero.py` | `Args`, `ResidualLiberoServer` | Load the rollout policy and serve websocket requests |
| `libero/main.py` | `Args`, `LiberoEvaluator` | Manage benchmark tasks, episodes, action phases, videos, and JSON results |

Each application exposes `run()`. Constructors store configuration; `run()` loads
external resources. Pure array and observation conversion helpers remain functions.
The existing `main()`, `collect_samples()`, and `eval_libero()` entry points delegate
to the classes. Simulator environments are closed after each task, including when
an episode fails.

For example, training can also be invoked from Python:

```python
from pathlib import Path
from scripts.train_residual_libero import TrainConfig, ResidualLiberoTrainer

trainer = ResidualLiberoTrainer(TrainConfig(
    groot_model_path="/path/to/groot",
    data_root=Path("/path/to/libero_rlds"),
    output=Path("checkpoints/residual_libero"),
))
trainer.run()
```

`LiberoEvaluator.run()` returns the simulator benchmark result dictionary and writes
the corresponding results and videos. See [LIBERO setup and evaluation](libero/README.md)
for simulator commands.

Run the orchestration regression tests with NumPy and pytest installed:

```bash
python3 -m pytest -q tests/test_scripts.py
```

These tests replace optional ML and simulator dependencies with test doubles.
They do not validate GPU training, model predictions, or a live websocket rollout.

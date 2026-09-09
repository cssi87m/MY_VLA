```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m scripts.build_memory_bank \
  --groot-model-path "$PWD/checkpoints/groot-n15-libero" \
  --groot-embodiment-tag new_embodiment \
  --data-root "$PWD/data/openvla-modified_libero_rlds" \
  --output "$PWD/checkpoints/expert_memory_bank"
```

BEFORE RUNNING: must create libero metadata for gr00t first, via script ```/home/vrh3/workspace/vla/some_fcking_new/scripts/create_groot_libero_metadata.py```

  --groot-model-path "$PWD/checkpoints/groot-n15-libero"
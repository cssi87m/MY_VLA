echo $PWD   
BASE=/home/vrh3/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
TARGET=$PWD/checkpoints/groot-n15-libero

mkdir -p "$TARGET/experiment_cfg"

for file in "$BASE"/*; do
[ "$(basename "$file")" = experiment_cfg ] ||
    ln -s "$file" "$TARGET/$(basename "$file")"
done

PYTHONPATH=. .venv/bin/python scripts/create_groot_libero_metadata.py \
--data-root data/openvla-modified_libero_rlds \
--base-metadata "$BASE/experiment_cfg/metadata.json" \
--output "$TARGET/experiment_cfg/metadata.json" \
--fps 10
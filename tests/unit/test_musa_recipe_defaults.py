"""Exercise the real shell entry without importing a GPU runtime."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_optimized_recipe_defaults_and_opt_out(tmp_path):
    root = Path(__file__).resolve().parents[2]
    metadata = tmp_path / "meta"
    metadata.mkdir()
    (metadata / "episodes.jsonl").touch()
    recorder = tmp_path / "record"
    keys = ["PER_DEVICE_TRAIN_BATCH_SIZE", "WSP_MUSA_VAE_BATCH_SIZE",
            "WSP_ROBOCOIN_VIDEO_CACHE_DIR", "WSP_ROBOCOIN_VIDEO_CACHE_GIB"]
    recorder.write_text(f"#!{sys.executable}\nimport json,os\n"
                        f"print(json.dumps({{k:os.environ[k] for k in {keys!r}}}))\n")
    recorder.chmod(0o755)
    env = os.environ.copy()
    for key in keys:
        env.pop(key, None)
    env.update(WSP_PYTHON="/bin/true", WSP_TRAIN=str(recorder),
               DATA_ROOT=str(tmp_path), PRETRAINED_MODEL_PATH=str(tmp_path),
               TOKENIZER_DIR=str(tmp_path), WSP_CACHE_DIR=str(tmp_path / "kernels"))
    command = ["bash", str(root / "recipes/posttrain/posttrain_agilex_fold_shirt_text_musa.sh")]
    defaults = json.loads(subprocess.check_output(command, env=env, text=True))
    assert defaults == dict(zip(keys, ["16", "16", str(root / ".cache/decoded-video"), "512"]))
    env.update(PER_DEVICE_TRAIN_BATCH_SIZE="8", WSP_MUSA_VAE_BATCH_SIZE="0",
               WSP_ROBOCOIN_VIDEO_CACHE_DIR="", WSP_ROBOCOIN_VIDEO_CACHE_GIB="64")
    overridden = json.loads(subprocess.check_output(command, env=env, text=True))
    assert overridden == dict(zip(keys, ["8", "0", "", "64"]))

# MUSA bring-up

The opt-in MUSA recipe uses [torchada](https://github.com/MooreThreads/torchada)
0.1.96 and an existing `torch_musa` image with a MUSA-enabled DeepSpeed build.
Use the container's default Python; no virtual environment is required.
The tested S5000 stack is torch / torch_musa 2.11.0.post2, torchvision
0.26.0.post2+da055cc, torchaudio 2.11.0.post2 and native MUSA DeepSpeed
0.19.3+3791c4da. Provision those vendor builds first; the base compile image
alone is not a complete training environment. A generic PyPI DeepSpeed install
is not a replacement for the validated MUSA build.

From the source checkout, in **each** container (do not install packages while
training is running):

```bash
# The image must already contain its matching MUSA PyTorch packages.
python -c 'import torch, torch_musa, torchvision, torchaudio; print(torch.__version__, torch.musa.device_count())'
python -m pip install -r requirements-musa.txt
```

If native MUSA DeepSpeed is not already installed, obtain the approved vendor
source for 0.19.3 at revision 3791c4da, unpack it, then install it using its
actual path (replace the example path below):

```bash
DS_ACCELERATOR=musa DS_BUILD_OPS=0 DS_BUILD_STRING=+3791c4da \
  python -m pip install --no-deps --no-build-isolation /path/to/DeepSpeed-3791c4da
```

The requirements file supplies this build's Python dependencies. DS_BUILD_OPS=0
skips optional prebuilt operators; fused/JIT operators may require compiler tools
and writable local cache space. Then register WorldScape and check the entry:

```bash
python -m pip install --no-deps --no-build-isolation -e .
DS_ACCELERATOR=musa python -c 'from deepspeed.accelerator import get_accelerator; assert get_accelerator().device_name() == "musa"'
WSP_USE_TORCHADA=1 TORCHADA_FORCE_CUDA_AVAILABLE=1 DS_ACCELERATOR=musa python -m worldscape_policy.cli.train --help
```

`requirements-musa.txt` includes core, training and Agilex decoding dependencies;
allow pip to resolve their transitive dependencies. It loads
`constraints-musa.txt` to protect the tested vendor stack. For a different image,
review those constraints against its vendor versions first. To preview changes,
add `--dry-run` to the requirements install command.

The editable project install intentionally uses `--no-deps`: upstream
`pyproject.toml` still pins CUDA-era torch and DeepSpeed versions. Do not run
`pip install -r requirements.txt`, plain `pip install -e .` or `pip install
-e '.[train]'` in this MUSA environment. `pip check` can report these known
upstream metadata mismatches; inspect other missing/conflicting dependencies
separately rather than downgrading the MUSA stack to silence them.

This is a source-checkout workflow: recipes and configs are used from the
repository. Simulation/serving extras (MuJoCo, PyBullet, websockets, robot SDKs)
are not needed for Agilex training. External FlashAttention is optional;
the default MUSA SDPA path does not require it.

## Launch

```bash
export DATA_ROOT=/path/to/Agilex_Cobot_Magic_fold_clothes
export PRETRAINED_MODEL_PATH=/path/to/wsp_2_pretrain
export TOKENIZER_DIR=/path/to/umt5-xxl
export OUTPUT_DIR=/path/to/output/unique-run-name
bash recipes/posttrain/posttrain_agilex_fold_shirt_text_musa.sh
```

Defaults: one node, eight processes, interactive text conditioning, BF16,
ZeRO-2, batch 8 per device, AdamW LR 6e-5 / betas (0.9, 0.999), seed 62.
The schedule is 50,000 steps with 5% warmup. For normal training, set
`RUN_STEPS=null SAVE_STEPS=5000 SAVE_AT_END=true` and a fresh output directory.
Set `RUN_STEPS=3` for a short smoke test. W&B is disabled by the MUSA entry.
No model downloads occur. `DATALOADER_NUM_WORKERS` controls workers per rank;
`DATASET_STEP_STRIDE` defaults to 1 and uses all eligible anchors.

The launcher enables `WSP_USE_TORCHADA=1`, `TORCHADA_FORCE_CUDA_AVAILABLE=1`
and `DS_ACCELERATOR=musa`. Importing the package applies torchada before its
CUDA-facing imports. The force-availability flag is required for upstream
CUDA availability gates; it does not turn MUSA tensor device types into CUDA.
Each rank binds its local device before process-group/model initialization.
The full native checkpoint loads on CPU, then casts to BF16 before transfer.
This requires substantial host RAM when all eight ranks load simultaneously.

Wan attention dispatches MUSA tensors to SDPA, preserving lengths, scaling,
bottom-right causal alignment and local windows. CUDA's existing path remains
available. The default backend is native SDPA. Set
`WSP_MUSA_ATTENTION_BACKEND=flash_attn` to use an installed MUSA FlashAttention
extension for supported shapes, or `auto` to allow extension discovery.
Unsupported masks/shapes retain SDPA semantics; extension runtime errors propagate.

## Dataset semantics to review

`robocoin_sim_eef6d_text` reads the original RoboCOIN LeRobot parquet/video
layout. It explicitly uses `eef_sim_pose_state/action` (12D) and
`gripper_open_scale_state/action` (2D); the joint-space 14D `action` column is
not substituted. Each arm becomes XYZ + rotation6D + gripper, giving 20D.
Rotation6D follows this project's first-two-columns layout.

The current default interprets the rotation triples as extrinsic XYZ Euler
angles in radians. `ROBOCOIN_ROTATION_REPRESENTATION=rotvec` selects rotation
vectors instead. Confirm the dataset producer's exact simulated-EEF rotation
convention before substantive training. The input fields must exist and contain
finite values; missing fields fail instead of falling back to joints.

This native adapter does **not** reproduce the unpublished customer plugin.
It uses episode-level instructions, identity normalization, absolute EEF
poses, four-chunk temporal packing, 96 action steps and up to 33 video frames.
Videos resize during decode to 320x160 per view to bound worker memory.
Review camera order, gripper scaling, rotation convention, normalization and
atomic subtask labels against the intended training recipe before interpreting
convergence or evaluating a policy.

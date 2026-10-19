# MUSA bring-up

The opt-in MUSA recipe uses [torchada](https://github.com/MooreThreads/torchada)
0.1.96 and an existing `torch_musa` image with a MUSA-enabled DeepSpeed build.
These installation steps target only this base image:

```text
sh-harbor.mthreads.com/mcctest/musa-compile-s5000:ubuntu22.04-py312-5.2.0-2.11.0.post2
```

Keep its torch / torch_musa / torchvision / torchaudio packages and default
Python. The pristine image does **not** contain DeepSpeed. Existing prepared
training containers already have it and do not need reinstallation.

In each newly created container, from the source checkout:

```bash
python -m pip install -r requirements-musa.txt
# Use the supplied native MUSA DeepSpeed source, revision 3791c4da.
# Replace this example with the unpacked source directory.
DS_ACCELERATOR=musa DS_BUILD_OPS=0 DS_BUILD_STRING=+3791c4da \
  python -m pip install --no-deps --no-build-isolation /path/to/DeepSpeed-3791c4da
python -m pip install --no-deps --no-build-isolation -e .
DS_ACCELERATOR=musa python -c 'import torch, torch_musa, deepspeed; from deepspeed.accelerator import get_accelerator; assert get_accelerator().device_name() == "musa"; print(torch.musa.device_count(), deepspeed.__version__)'
WSP_USE_TORCHADA=1 TORCHADA_FORCE_CUDA_AVAILABLE=1 DS_ACCELERATOR=musa python -m worldscape_policy.cli.train --help
```

The requirements overlay adds training/decoding dependencies and aligns ordinary
Python packages with the tested environment, including setuptools 75.8.0 for the
source build. Already satisfied dependencies are reused. Its constraints retain
the image's MUSA stack. The separate vendor DeepSpeed source is not included in
this repository; DS_BUILD_OPS=0 skips optional operator precompilation.

Only DeepSpeed and WorldScape use --no-deps; ordinary dependencies must resolve
normally. Do not use upstream requirements.txt or plain pip install -e . here:
upstream metadata pins different PyTorch versions. Known pip check mismatches
for torch/vision/audio do not justify replacing the MUSA stack. Use the source
checkout for training recipes and configs. Do not install packages during a run.
The default SDPA backend requires no external FlashAttention installation.

## Launch

```bash
export DATA_ROOT=/path/to/Agilex_Cobot_Magic_fold_clothes
export PRETRAINED_MODEL_PATH=/path/to/wsp_2_pretrain
export TOKENIZER_DIR=/path/to/umt5-xxl
export OUTPUT_DIR=/path/to/output/unique-run-name
bash recipes/posttrain/posttrain_agilex_fold_shirt_text_musa.sh
```

Defaults: one node, eight processes, interactive text conditioning, BF16,
ZeRO-2, batch 16 per device (GBS128 on eight GPUs with accumulation 1),
AdamW LR 6e-5 / betas (0.9, 0.999), seed 62.
The schedule is 50,000 steps with 5% warmup. For normal training, set
`RUN_STEPS=null SAVE_STEPS=5000 SAVE_AT_END=true` and a fresh output directory.
Set `RUN_STEPS=3` for a short smoke test. W&B is disabled by the MUSA entry.
No model downloads occur. `DATALOADER_NUM_WORKERS` controls workers per rank;
the recipe uses eight workers per rank and a prefetch factor of two.
`DATASET_STEP_STRIDE` defaults to 1 and uses all eligible anchors.

The MUSA recipe groups up to 16 independent samples for VAE encoding
(`WSP_MUSA_VAE_BATCH_SIZE=16`). This changes the execution batch, not the video
length, resolution or sample order. Set the variable to 0 or 1 for the original
per-sample path. CPU, CUDA and tiled encoding retain their existing paths.

Decoded RGB videos are reused across workers through a read-only disk cache.
The recipe defaults to `.cache/decoded-video` under the checkout and a 512 GiB
limit (`WSP_ROBOCOIN_VIDEO_CACHE_DIR` / `WSP_ROBOCOIN_VIDEO_CACHE_GIB`). For this
dataset at 320x160 per camera, a full cache is approximately 333 GiB. It fills
on demand, so cold runs include decoding and cache creation. Use writable
storage with sufficient free capacity; shared storage must support cross-node
file locks and atomic rename. A node-local path is also supported, with a
separate cache and capacity requirement on every node.

Cache keys include source content, decoder versions and resize settings.
Pixels are preserved exactly; a full cache uses the original decoder without
deleting existing entries. Interrupted cache writes can conservatively consume
quota until the dedicated cache directory is reset while no jobs use it.
Keep source videos unchanged during a run. Changes detected while hashing,
waiting for the cache lock or decoding raise an error instead of returning a
stale entry. Invalid quota records also fail explicitly; do not edit the quota
journal or remove cache files while jobs are using the directory.
Explicitly set `WSP_ROBOCOIN_VIDEO_CACHE_DIR=""` to disable this cache.
These settings add no package dependencies and do not change dataset sampling.

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

## Optional native RMSNorm

`WSP_MUSA_FUSED_RMSNORM=1` enables the native MUSA BF16 RMS normalization
operator when both forward and backward kernels are available. The learned
weight is multiplied separately, preserving Wan's normalization/cast/weight
order. Unsupported devices, dtypes or layouts, and unavailable kernels use the
original implementation; runtime operator failures propagate. The default is
`0`. This option needs no extra package and leaves model parameter names intact.
Floating-point results can differ slightly from the unfused implementation.

The multi-node launcher forwards this setting to every node.

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

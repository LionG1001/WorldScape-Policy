# MUSA bring-up

The opt-in MUSA recipe uses [torchada](https://github.com/MooreThreads/torchada)
0.1.96 and an existing `torch_musa` image with a MUSA-enabled DeepSpeed build.
Install `requirements-musa.txt` with `python -m pip install --no-deps -r
requirements-musa.txt`. Keep the image's torch / torch_musa stack; the upstream
CUDA dependency pins are not a MUSA installation recipe. Use the container's
default Python. Dataset decoding also needs PyAV, OpenCV, pandas, pyarrow and
SciPy, plus the ordinary native training dependencies.

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

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
The schedule is 50,000 steps with 5% warmup, but `RUN_STEPS=3` bounds the
initial attempt. Checkpoint saving and W&B are disabled by default.
No model downloads occur. For a full run, explicitly set `RUN_STEPS=null`,
`SAVE_STEPS=...`, `SAVE_AT_END=true` and a fresh output directory.
For a short update check, `OPTIMIZER_WARMUP_RATIO=0` uses the target LR immediately.
`DATASET_STEP_STRIDE=24` reduces candidate anchors for bring-up; the default 1
uses all eligible anchors. `DATALOADER_NUM_WORKERS` defaults to 4 per rank.

The launcher enables `WSP_USE_TORCHADA=1`, `TORCHADA_FORCE_CUDA_AVAILABLE=1`
and `DS_ACCELERATOR=musa`. Importing the package applies torchada before its
CUDA-facing imports. The force-availability flag is required for upstream
CUDA availability gates; it does not turn MUSA tensor device types into CUDA.
Each rank binds its local device before process-group/model initialization.
The full native checkpoint loads on CPU, then casts to BF16 before transfer.
This requires substantial host RAM when all eight ranks load simultaneously.

Wan attention dispatches MUSA tensors to SDPA, preserving lengths, scaling,
bottom-right causal alignment and local windows. CUDA's existing path remains
available. This is a correctness fallback, not a performance claim about
FlashAttention equivalence.

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

## Validation evidence

Each rank writes `rank-N-steps.jsonl` in the output directory, containing step
metrics, peak allocated MUSA memory and changes in samples from eight trainable
tensors. Nonfinite metrics fail the run. Sampled parameter changes are evidence
of updates, not a full-model numerical parity check. Use a unique output folder
for every run because the evidence files append.

Validated on eight S5000 80-GiB GPUs with torch / torch_musa 2.11.0.post2,
MUSA DeepSpeed 0.19.3 and torchada 0.1.96: all eight ranks completed three
optimizer steps at batch 8 per device, exit 0. Mean loss was
0.902375 / 0.470615 / 0.421118, gradients were finite and sampled weights
changed. Peak allocated memory was 38.18–38.95 GiB per device. This attempt
used stride 24 and no warmup. The focused test suite passed 14 tests.

Use a container-local `TMPDIR` and `TRITON_CACHE_DIR` when possible; NFS-backed
multiprocessing temporary directories produced cleanup `EBUSY` warnings after
successful training in this environment.

Focused tests:

```bash
python -m pytest -q tests/unit/test_musa_sdpa.py \
  tests/unit/test_robocoin_eef.py tests/unit/test_native_training_runtime.py
```

Before a long run: verify multi-rank steps and finite gradients, confirm data
semantics, then validate checkpoint save/resume, numerical behavior and task
quality. A three-step run does not establish convergence or CUDA parity.

## Multi-node launch

Run the following on **each node**, using a different `NODE_RANK` (0, 1, ...)
and that host's actual training RoCE adapter names:

```bash
export MASTER_ADDR=<node-0-reachable-ip> MASTER_PORT=29620
export NNODES=2 NODE_RANK=0 NUM_GPUS=8
export MCCL_IB_HCA='=mlx5_2:1,mlx5_3:1,...'
export MCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0
export DATA_ROOT=/shared/path/to/Agilex_Cobot_Magic_fold_clothes
export PRETRAINED_MODEL_PATH=/shared/path/to/wsp_2_pretrain
export TOKENIZER_DIR=/shared/path/to/umt5-xxl
export OUTPUT_DIR=/shared/path/to/output/unique-run-name
export TMPDIR=/tmp/worldscape-policy TRITON_CACHE_DIR=/tmp/worldscape-triton
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
bash recipes/posttrain/posttrain_agilex_fold_shirt_text_musa_multinode.sh
```

The multi-node entry reuses the single-node recipe and maps node parameters to
its `MLP_WORKER_*` launcher contract. It defaults to 50 steps, eight processes
per node and batch 8 per process (global batch 128 with two nodes). It does not
SSH to other machines: start it once on each node with the same rendezvous
address/port, output directory and training options. Keep the launchers running
until both exit. Python, MUSA libraries and source files must agree across nodes.

Communication defaults reference the
[starVLA MUSA launcher](https://github.com/LionG1001/starVLA/blob/gl_musa-dev/examples/Robotwin/train_files/run_robotwin_train_qwen3_5_musa.sh)
and [ms-swift node launcher](https://github.com/caizhi-mt/ms-swift/blob/v4.0.2-musa/run_qwen3.5_27b_node.sh):
RoCE v2 / GID 3, IB enabled, 20 MiB communication buffers, up to 14 channels,
and explicit per-host HCA selection. Verify these values against the actual
network; adapter numbering can differ between otherwise identical machines.
`MCCL_CROSS_NIC` is not set by the launcher; the library default applies unless
the caller sets it explicitly. All defaults can be overridden in the environment.
Async launch remains enabled
(`MUSA_LAUNCH_BLOCKING=0`) for measurement. Qwen/Megatron-specific patches and
debug flags from the reference scripts are not required by this training path.

With host-network containers, check rendezvous and SSH ports for conflicts
before starting services. Container SSH is optional for this launcher; a host
SSH connection plus `docker exec` is sufficient. Verify a 16-rank collective
before the first two-node training run, and use node-local temporary/cache paths.

Validated the two-node entry on 2 × 8 S5000 GPUs: all 16 ranks completed
50 optimizer steps at global batch 128, both launchers exited 0, metrics were
finite and sampled weights changed. The run used stride 1 and 5% warmup,
with `MCCL_CROSS_NIC` unset. With a synchronized measurement callback, the
preselected steps 11–50 averaged 6.829 s/step (P90 7.223 s); all 50 steps
including the cold first step averaged 7.993 s. Peak allocated memory was
33.69–34.34 GiB per device. These are bring-up measurements, not a quality or
cross-platform parity result; checkpoint save/resume remains unvalidated.

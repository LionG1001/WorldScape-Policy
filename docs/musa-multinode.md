# MUSA training and multi-node launch

The MUSA recipe runs the full 50,000-step schedule by default and saves every
5,000 steps and at the end. Set RUN_STEPS=3 for a short smoke test. Set a fresh
OUTPUT_DIR for each single-node run; the multi-node controller generates a
unique run name and output directory unless overridden.

Deployment variables are explicit at the top of the existing shell entries.
No .env file is loaded. With deployment paths configured, change to the checkout
and execute the desired shell entry directly. Single-node run names are generated
automatically. Both modes retain full-training defaults; use RUN_STEPS=50,
SAVE_STEPS=0 and SAVE_AT_END=false for a short validation run.

## One-command launch

Create `hostfile` in the repository root, using the
[ms-swift format](https://github.com/caizhi-mt/ms-swift/blob/v4.0.2-musa/hostfile):

```text
node0 slots=8
node1 slots=8
```

Use reachable hostnames or IPs. Each node must have equal positive slots.
The first host is the default rendezvous address; row order assigns ranks.
Only host and slots belong in this file.

From the source checkout, after installing the MUSA dependencies on all nodes:

```bash
export DATA_ROOT=/path/to/dataset
export PRETRAINED_MODEL_PATH=/path/to/checkpoint
export TOKENIZER_DIR=/path/to/tokenizer
bash recipes/posttrain/posttrain_agilex_fold_shirt_text_musa_multinode.sh
```

SSH keys and host-key trust must already be configured. Code, data and model
paths must exist on every node. The launcher does not install or synchronize
them. It runs the existing training recipe through its internal --node mode.

## Optional connection settings

- HOSTFILE: defaults to hostfile in the repository root.
- SSH_USER / SSH_PORT: deployment defaults are mccxadmin and port 22; edit or override for your environment.
- REMOTE_CONTAINER: defaults to gl-worldscape-test; set an empty value for direct SSH.
- REMOTE_WORKDIR: the deployment path is explicit at the top of the multi-node script; edit or override it for another environment.
- MASTER_ADDR / MASTER_PORT: override only when the first host is not reachable
  over the training network or port 29620 is occupied.

MCCL adapter selection is automatic by default. If the deployment requires a
specific set of RoCE adapters, configure MCCL_IB_HCA in the node environment;
exporting it on the controller applies the same list to all nodes. Network
defaults are in recipes/common/musa_runtime.sh and can be overridden.
Single-node training does not cap channels; multi-node training defaults to
MCCL_MAX_NCHANNELS=32 and accepts an override.

DRYRUN=1 prints commands without SSH. CHECK_ONLY=1 checks connections and paths
without starting training. Actual launch performs the same checks first.
Per-node logs and exit codes are saved under OUTPUT_DIR; controller SSH logs
are under logs/multinode-<id>. Failure or Ctrl-C terminates this launch's process
groups. If SSH becomes unreachable, inspect the remote processes manually.

## Optional ZeRO-2 alignment

WSP_MUSA_ZERO_ALIGNMENT_BYTES=128 pads flattened BF16/FP16 parameter groups so
each rank's shard is aligned for MCCL all-gather. The default 0 disables it.
Model tensor shapes do not change. Restore checkpoints with the same alignment;
the loader rejects mismatches. This option applies only to MUSA ZeRO-2.

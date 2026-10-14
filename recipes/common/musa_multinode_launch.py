"""SSH coordinator and scoped remote supervisor; Python standard library only."""
import base64
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime

HELPER = 'recipes/common/musa_multinode_launch.py'
ENTRY = 'recipes/posttrain/posttrain_agilex_fold_shirt_text_musa_multinode.sh'
# Explicit transfer: never forward credentials or the controller's full environment.
KEYS = '''DATA_ROOT PRETRAINED_MODEL_PATH TOKENIZER_DIR RUN_NAME OUTPUT_DIR NUM_GPUS
RUN_STEPS MAX_STEPS PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
DATALOADER_NUM_WORKERS OPTIMIZER_WARMUP_RATIO DATASET_STEP_STRIDE LEARNING_RATE
LR_SCHEDULE ADAM_BETA1 ADAM_BETA2 SEED SAVE_STEPS SAVE_AT_END WANDB_ENABLED WANDB_MODE
WSP_MUSA_ATTENTION_BACKEND WSP_MUSA_ZERO_ALIGNMENT_BYTES WSP_PYTHON
MUSA_VISIBLE_DEVICES OMP_NUM_THREADS TMPDIR TRITON_CACHE_DIR
MCCL_DEBUG MCCL_IB_HCA MCCL_MAX_NCHANNELS MCCL_MIN_NCHANNELS MCCL_IB_GID_INDEX MCCL_IB_ROCE_VERSION_NUM
MCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME MCCL_BUFFSIZE MCCL_NET_SHARED_BUFFERS
MUSA_LAUNCH_BLOCKING MUSA_DEVICE_MAX_CONNECTIONS PYTORCH_MUSA_ALLOC_CONF'''.split()


def encode(config):
    return base64.urlsafe_b64encode(json.dumps(config).encode()).decode()


def state_path(config):
    return Path('/tmp') / ('wsp-launch-' + config['token'] + '-' + config['env']['NODE_RANK'])


def node(mode, payload):
    config = json.loads(base64.urlsafe_b64decode(payload))
    state = state_path(config)
    cancelled = state.with_suffix('.cancel')
    if mode == '_stop':
        cancelled.touch()  # Also prevents a delayed SSH startup after cancellation.
        # Signal only a live supervisor whose exact argv matches this launch.
        if state.exists():
            pid = int(state.read_text())
            try:
                argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                if b'_node' in argv and payload.encode() in argv:
                    os.kill(pid, signal.SIGTERM)
                    for _ in range(250):
                        if not state.exists():
                            break
                        time.sleep(.1)
                    else:
                        raise RuntimeError('Remote supervisor cleanup timed out')
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass
        return 0
    os.chdir(config['workdir'])
    env = dict(os.environ, **config['env'])
    entry = Path(config['entry'])
    if not entry.is_file():
        raise RuntimeError(f'Missing remote script: {entry}')
    output = Path(env['OUTPUT_DIR'])
    if mode == '_check':
        if output.exists():
            raise RuntimeError(f'Use a new OUTPUT_DIR: {output}')
        for key in ('DATA_ROOT', 'PRETRAINED_MODEL_PATH', 'TOKENIZER_DIR'):
            if not Path(env[key]).is_dir():
                raise RuntimeError(f'Missing {key}: {env[key]}')
        print('Preflight passed:', env['NODE_RANK'], flush=True)
        return 0
    if cancelled.exists():
        raise RuntimeError('Launch already cancelled')
    output.mkdir(parents=True, exist_ok=True)
    for key in ('TMPDIR', 'TRITON_CACHE_DIR'):
        if env.get(key):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    rank = env['NODE_RANK']
    logpath = output / f'launcher-node-{rank}.log'
    child = None

    def terminate(signum, frame):
        # Unwind Popen.wait first; waiting inside a signal handler can deadlock
        # on subprocess's internal waitpid lock.
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        with state.open('x') as f:
            f.write(str(os.getpid()))
        with logpath.open('x') as log:
            print(f'node={rank} log={logpath}', flush=True)
            if cancelled.exists():
                raise RuntimeError('Launch already cancelled')
            blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
            try:
                child = subprocess.Popen(['bash', str(entry), *config['args']], env=env,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, blocked))
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
            code = child.wait()
        (output / f'launcher-node-{rank}.exit').write_text(str(code) + '\n')
        return code if code >= 0 else 128 - code
    except BaseException as exc:
        if state.exists():
            code = exc.code if isinstance(exc, SystemExit) else 1
            (output / f'launcher-node-{rank}.exit').write_text(str(code) + '\n')
        raise
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                pass
        state.unlink(missing_ok=True)


def hosts(path):
    result = []
    for line in Path(path).read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if not parts:
            continue
        if len(parts) != 2 or not parts[1].startswith('slots='):
            raise ValueError('Hostfile rows must be: host slots=N')
        target, slot_field = parts
        slots = slot_field.removeprefix('slots=')
        if not slots.isdigit() or int(slots) < 1:
            raise ValueError('Hostfile requires positive slots=N')
        port = os.environ.get('SSH_PORT', '62218')
        if target.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_.@:-]+', target):
            raise ValueError('Invalid SSH target')
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError('Invalid SSH port')
        result.append((target, port, int(slots)))
    if len(result) < 2 or len({(h, p) for h, p, _ in result}) != len(result):
        raise ValueError('At least two distinct SSH endpoints required')
    if len({s for _, _, s in result}) != 1:
        raise ValueError('This torchrun recipe requires equal slots on every node')
    if os.environ.get('NUM_GPUS') and os.environ['NUM_GPUS'] != str(result[0][2]):
        raise ValueError('NUM_GPUS must match hostfile slots')
    return result


def main(args):
    env = {k: os.environ[k] for k in KEYS if k in os.environ}
    for key in ('DATA_ROOT', 'PRETRAINED_MODEL_PATH', 'TOKENIZER_DIR'):
        if not env.get(key):
            raise ValueError(f'Set {key}')
    workdir = str(Path(__file__).resolve().parents[2])
    env.setdefault('RUN_NAME', 'musa-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    env.setdefault('OUTPUT_DIR', str(Path(workdir) / 'checkpoints' / env['RUN_NAME']))
    if not Path(env['OUTPUT_DIR']).is_absolute():
        raise ValueError('OUTPUT_DIR must be an absolute remote path')
    nodes = hosts(os.environ.get('HOSTFILE', 'hostfile'))
    master = os.environ.get('MASTER_ADDR', nodes[0][0].split('@')[-1])
    port = os.environ.get('MASTER_PORT', '29620')
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError('Invalid MASTER_PORT')
    token = uuid.uuid4().hex
    commands = []
    for rank, (host, ssh_port, slots) in enumerate(nodes):
        rank_env = dict(env, NNODES=str(len(nodes)), NODE_RANK=str(rank), MASTER_ADDR=master,
                        MASTER_PORT=port, NUM_GPUS=str(slots))
        entry = ENTRY
        node_args = ['--node', *args]
        config = dict(env=rank_env, workdir=workdir, entry=entry,
                      args=node_args, token=token)
        payload = encode(config)
        remote = ['python3', str(Path(workdir) / HELPER)]
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-p', ssh_port, host]
        commands.append((ssh, remote, payload))
        print(f'node={rank} ssh={host}:{ssh_port} '
              f'master={master}:{port} slots={slots}', flush=True)
    if os.environ.get('DRYRUN') == '1':
        for ssh, remote, payload in commands:
            print(shlex.join([*ssh, shlex.join([*remote, '_node', payload])]))
        return 0
    # Complete every preflight before any node starts training.
    for ssh, remote, payload in commands:
        subprocess.run([*ssh, shlex.join([*remote, '_check', payload])], check=True, timeout=45)
    if os.environ.get('CHECK_ONLY') == '1':
        return 0
    logdir = Path(os.environ.get('LAUNCH_LOG_DIR', f'logs/multinode-{token}'))
    logdir.mkdir(parents=True, exist_ok=False)
    processes, logs = [], []
    success = False

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for rank, (ssh, remote, payload) in enumerate(commands):
            log = (logdir / f'node-{rank}.ssh.log').open('w')
            logs.append(log)
            processes.append(subprocess.Popen([*ssh, shlex.join([*remote, '_node', payload])],
                                               stdout=log, stderr=subprocess.STDOUT))
        print(f'Controller logs: {logdir}; training logs: {env["OUTPUT_DIR"]}/launcher-node-*.log', flush=True)
        while True:
            codes = [p.poll() for p in processes]
            if any(c not in (None, 0) for c in codes):
                raise RuntimeError(f'Node/SSH failure: {codes}; see {logdir}')
            if all(c == 0 for c in codes):
                success = True
                print('All nodes exited successfully.', flush=True)
                return 0
            time.sleep(1)
    finally:
        if not success:
            for ssh, remote, payload in commands:
                try:
                    subprocess.run([*ssh, shlex.join([*remote, '_stop', payload])], timeout=35, check=True)
                except (subprocess.SubprocessError, OSError) as exc:
                    print(f'Cleanup not confirmed for {ssh[-1]}: {exc}; inspect remote logs.', file=sys.stderr)
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        for log in logs:
            log.close()


if __name__ == '__main__':
    try:
        if len(sys.argv) == 3 and sys.argv[1] in ('_check', '_node', '_stop'):
            sys.exit(node(sys.argv[1], sys.argv[2]))
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

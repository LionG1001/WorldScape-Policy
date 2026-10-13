"""Exercise real supervisor processes with local SSH transport (no GPU required)."""
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'recipes/posttrain/launch_agilex_fold_shirt_text_musa_multinode.sh'


def setup(tmp_path, script):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    ssh = bindir / 'ssh'
    ssh.write_text('#!/usr/bin/env python3\nimport os,sys\nos.execvp("bash", ["bash", "-c", sys.argv[-1]])\n')
    ssh.chmod(0o755)
    hostfile = tmp_path / 'hosts'
    hostfile.write_text('node0 22 =mlx5_0:1\nnode1 62218 =mlx5_1:1\n')
    entry = tmp_path / 'entry with spaces.sh'
    entry.write_text(script)
    env = dict(os.environ, PATH=str(bindir)+':'+os.environ['PATH'], HOSTFILE=str(hostfile),
               REMOTE_WORKDIR=str(ROOT), REMOTE_SCRIPT=str(entry), REMOTE_CONTAINER='',
               MASTER_ADDR='training-address', RUN_NAME='test', OUTPUT_DIR=str(tmp_path/'out'),
               DATA_ROOT=str(tmp_path), PRETRAINED_MODEL_PATH=str(tmp_path), TOKENIZER_DIR=str(tmp_path),
               LAUNCH_LOG_DIR=str(tmp_path/'logs'))
    env.pop('DRYRUN', None)
    env.pop('CHECK_ONLY', None)
    return env


def test_launch_quotes_and_ranks(tmp_path):
    env=setup(tmp_path, 'printf "%s\\n" "$NODE_RANK" "$NNODES" "$MASTER_ADDR" "$MCCL_IB_HCA" "$1"\n')
    literal='literal $(touch SHOULD_NOT_EXIST); "x y"'
    result=subprocess.run(['bash',str(LAUNCHER),literal],env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    for rank in range(2):
        lines=(tmp_path/f'out/launcher-node-{rank}.log').read_text().splitlines()
        assert lines==[str(rank),'2','training-address',f'=mlx5_{rank}:1',literal]
        assert (tmp_path/f'out/launcher-node-{rank}.exit').read_text().strip()=='0'


def test_preflight_rejects_existing_output(tmp_path):
    env=setup(tmp_path, 'echo should-not-run\n')
    (tmp_path/'out').mkdir()
    result=subprocess.run(['bash',str(LAUNCHER)],env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode!=0
    assert not list((tmp_path/'out').iterdir())


def test_failure_stops_peer(tmp_path):
    env=setup(tmp_path, 'if [[ "$NODE_RANK" == 0 ]]; then sleep 1; exit 7; fi\ntrap "echo stopped; exit 143" TERM\necho ready\nsleep 60 & wait\n')
    start=time.monotonic()
    result=subprocess.run(['bash',str(LAUNCHER)],env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode!=0
    assert time.monotonic()-start<10
    assert 'stopped' in (tmp_path/'out/launcher-node-1.log').read_text()


def test_interrupt_stops_both(tmp_path):
    env=setup(tmp_path, 'trap "echo stopped; exit 143" TERM\necho ready\nsleep 60 & wait\n')
    process=subprocess.Popen(['bash',str(LAUNCHER)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        for _ in range(100):
            if all((tmp_path/f'out/launcher-node-{r}.log').exists() and 'ready' in (tmp_path/f'out/launcher-node-{r}.log').read_text() for r in range(2)):
                break
            time.sleep(.05)
        else:
            raise AssertionError('Nodes did not start')
        process.send_signal(signal.SIGINT)
        process.communicate(timeout=15)
        assert process.returncode==130
        assert all('stopped' in (tmp_path/f'out/launcher-node-{r}.log').read_text() for r in range(2))
    finally:
        if process.poll() is None:
            process.kill()

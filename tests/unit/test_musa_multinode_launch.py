"""Exercise real supervisor processes with local SSH transport (no GPU required)."""
import os
import shutil
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'recipes/posttrain/posttrain_agilex_fold_shirt_text_musa_multinode.sh'


def setup(tmp_path, script):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    ssh = bindir / 'ssh'
    ssh.write_text('#!/usr/bin/env python3\nimport os,sys\nos.execvp("bash", ["bash", "-c", sys.argv[-1]])\n')
    ssh.chmod(0o755)
    hostfile = tmp_path / 'hosts'
    hostfile.write_text('node0 slots=8\nnode1 slots=8\n')
    project = tmp_path / 'project with spaces'
    for relative in ['recipes/common/musa_multinode_launch.py',
                     'recipes/posttrain/posttrain_agilex_fold_shirt_text_musa_multinode.sh']:
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    (project / 'recipes/posttrain/posttrain_agilex_fold_shirt_text_musa.sh').write_text(script)
    env = dict(os.environ, PATH=str(bindir)+':'+os.environ['PATH'], HOSTFILE=str(hostfile),
               REMOTE_WORKDIR=str(project), REMOTE_CONTAINER='', MCCL_IB_HCA='=mlx5_0:1',
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
        assert lines[1:]==[str(rank),'2','training-address','=mlx5_0:1',literal]
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


def test_default_entry_dispatches_node_mode(tmp_path):
    import shutil
    env = setup(tmp_path, 'exit 99\n')
    project = tmp_path / 'remote project'
    for relative in ['recipes/common/musa_multinode_launch.py',
                     'recipes/posttrain/posttrain_agilex_fold_shirt_text_musa_multinode.sh']:
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    (project / 'recipes/posttrain/posttrain_agilex_fold_shirt_text_musa.sh').write_text(
        'printf "node=%s world=%s arg=%s\\n" "$NODE_RANK" "$MLP_WORKER_NUM" "$1"\n')
    env['REMOTE_WORKDIR'] = str(project)
    result = subprocess.run(['bash', str(LAUNCHER), 'literal x'], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    for rank in range(2):
        assert f'node={rank} world=2 arg=literal x' in (tmp_path / f'out/launcher-node-{rank}.log').read_text()


def test_hostfile_slots_and_validation(tmp_path):
    import runpy
    parser = runpy.run_path(str(ROOT / 'recipes/common/musa_multinode_launch.py'))['hosts']
    path = tmp_path / 'hostfile'
    path.write_text('# nodes\nnode0 slots=4\nnode1 slots=4 # comment\n')
    nodes = parser(path)
    assert [row[2] for row in nodes] == [4, 4]
    for content in [
        'node0 slots=8\nnode1 slots=4\n',
        'node0 slots=0\nnode1 slots=0\n',
        'node0 slots=8 port=99999\nnode1 slots=8\n',
        'node0 slots=8 slots=8\nnode1 slots=8\n',
        'node0 22 =mlx5_0:1\nnode1 22 =mlx5_1:1\n',
    ]:
        path.write_text(content)
        try:
            parser(path)
        except ValueError:
            pass
        else:
            raise AssertionError(content)


def test_plain_hostfile_sets_gpu_count(tmp_path):
    env = setup(tmp_path, 'printf "%s\\n" "$NUM_GPUS"\n')
    Path(env['HOSTFILE']).write_text('node0 slots=4\nnode1 slots=4\n')
    env.pop('NUM_GPUS', None)
    result = subprocess.run(['bash', str(LAUNCHER)], env=env, capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    for rank in range(2):
        assert (tmp_path / f'out/launcher-node-{rank}.log').read_text().splitlines()[-1] == '4'


def test_launch_defaults_without_extra_configuration(tmp_path):
    env = setup(tmp_path, 'printf "master=%s gpus=%s\\n" "$MASTER_ADDR" "$NUM_GPUS"\n')
    for key in ('MASTER_ADDR', 'RUN_NAME', 'OUTPUT_DIR', 'LAUNCH_LOG_DIR'):
        env.pop(key, None)
    env['CHECK_ONLY'] = '1'
    result = subprocess.run(['bash', str(LAUNCHER)], env=env, capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'master=node0:29620 slots=8' in result.stdout
    assert 'Preflight passed: 0' in result.stdout
    assert 'Preflight passed: 1' in result.stdout

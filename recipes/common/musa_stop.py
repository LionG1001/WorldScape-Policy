"""Stop only this checkout's training jobs, locally and on hostfile peers."""
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from musa_multinode_launch import HELPER, hosts, node

ROOT = Path(__file__).resolve().parents[2]


def processes():
    found = {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            argv = [v.decode() for v in (path / 'cmdline').read_bytes().split(b'\0') if v]
            found[int(path.name)] = (int(stat[1]), stat[19], stat[0], argv)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def matches(argv, root):
    helper = str(root / HELPER)
    if len(argv) >= 2 and argv[1] == helper:
        if len(argv) == 2:
            return 'controller'
        if len(argv) == 4 and argv[2] == '_node':
            return 'supervisor'
    if (argv[1:3] == ['-m', 'torch.distributed.run']
            and '--module' in argv
            and argv[argv.index('--module') + 1:argv.index('--module') + 2] == ['worldscape_policy.cli.train']):
        try:
            config = Path(argv[argv.index('--config') + 1])
            if config.is_absolute() and config.is_relative_to(root / 'configs'):
                return 'trainer'
        except (ValueError, IndexError):
            pass
    return None


def signal_if_same(pid, original, signum):
    current = processes().get(pid)
    if current and current[1] == original[1] and current[2] != 'Z':
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


def stop_local():
    snapshot = processes()
    targets = {pid: value for pid, value in snapshot.items()
               if matches(value[3], ROOT)}
    # Controllers cancel launches that may still be starting on another node.
    for pid, value in targets.items():
        if matches(value[3], ROOT) == 'controller':
            signal_if_same(pid, value, signal.SIGINT)
    for pid, value in targets.items():
        if matches(value[3], ROOT) == 'supervisor':
            try:
                node('_stop', value[3][3])
            except (FileNotFoundError, ProcessLookupError):
                pass
    # Also handle standalone torchrun, and children remaining after a failed supervisor.
    owned = set(targets)
    while True:
        children = {pid for pid, value in snapshot.items() if value[0] in owned}
        if children <= owned:
            break
        owned |= children
    for pid in owned:
        signal_if_same(pid, snapshot[pid], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while owned and time.monotonic() < deadline:
        live = processes()
        owned = {pid for pid in owned if pid in live and live[pid][1] == snapshot[pid][1]
                 and live[pid][2] != 'Z'}
        if owned:
            time.sleep(.2)
    for pid in owned:
        signal_if_same(pid, snapshot[pid], signal.SIGKILL)
    for _ in range(25):
        live = processes()
        remaining = [pid for pid in owned if pid in live and live[pid][1] == snapshot[pid][1]
                     and live[pid][2] != 'Z']
        if not remaining:
            print(f'Stopped {len(targets)} training launcher(s) in {ROOT}', flush=True)
            return
        time.sleep(.2)
    raise RuntimeError(f'Could not confirm process cleanup: {remaining}')


def main():
    peers = hosts(os.environ.get('HOSTFILE', str(ROOT / 'hostfile')))
    stop_local()
    failed = []
    remote = shlex.join(['python3', str(ROOT / 'recipes/common/musa_stop.py'), '--local'])
    for host, port, _ in peers:
        print(f'Stopping project training on {host}:{port}', flush=True)
        try:
            subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                            '-p', port, host, remote], check=True, timeout=90)
        except (subprocess.SubprocessError, OSError) as exc:
            failed.append(host)
            print(f'Cleanup unconfirmed on {host}: {exc}', file=sys.stderr)
    if failed:
        raise RuntimeError(f'Check these nodes manually: {failed}')


if __name__ == '__main__':
    try:
        if sys.argv[1:] == ['--local']:
            stop_local()
        elif sys.argv[1:]:
            raise ValueError('Usage: musa_stop.py [--local]')
        else:
            main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

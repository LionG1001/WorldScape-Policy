"""Opt-in, bounded disk cache for deterministic decoded RGB frames."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


def cached_video(source, directory, decoder, *, signature, max_bytes):
    # Imported only when enabled: the normal decoder remains portable.
    import fcntl

    source = Path(source)
    directory = Path(directory)
    if max_bytes <= 0:
        raise ValueError("video cache size must be positive")
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    hashed = source.stat()
    if (before.st_size, before.st_mtime_ns) != (hashed.st_size, hashed.st_mtime_ns):
        raise RuntimeError(f"Video changed during hashing: {source}")
    identity = json.dumps([digest.hexdigest(), signature], sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (key + ".npy")
    with (directory / (key + ".lock")).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            return _read_frames(destination)
        frames = decoder()
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Video changed during decoding: {source}")
        _validate_frames(frames)
        # Serialize quota checks and commits across different videos/workers.
        with (directory / ".quota.lock").open("a+b") as quota_lock:
            fcntl.flock(quota_lock, fcntl.LOCK_EX)
            # A shared reservation journal avoids relying on a client's cached
            # NFS directory listing. Reserve before publishing; interrupted
            # writes can conservatively consume quota but cannot overspend it.
            quota_lock.seek(0)
            journal = quota_lock.read()
            if journal and not journal.endswith(b"\n"):
                raise ValueError("Incomplete video cache quota journal")
            used = (sum(int(line) for line in journal.splitlines()) if journal else
                    sum(p.stat().st_size for p in directory.glob("*.npy")))
            estimated = frames.nbytes + 4096
            if used + estimated > max_bytes or shutil.disk_usage(directory).free < estimated + 2**30:
                return frames  # Exact uncached decode when the cache is full.
            reservation = (f"{used}\n" if not journal else "") + f"{estimated}\n"
            quota_lock.write(reservation.encode())
            quota_lock.flush()
            os.fsync(quota_lock.fileno())
            temp = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
                    temp = Path(stream.name)
                    np.save(stream, frames, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, destination)
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
        return _read_frames(destination)


def _validate_frames(frames):
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError("Video cache expects nonempty uint8 [T,H,W,3] frames")


def _read_frames(path):
    frames = np.load(path, mmap_mode="r", allow_pickle=False)
    _validate_frames(frames)
    return frames

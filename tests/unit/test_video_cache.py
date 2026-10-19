import multiprocessing
import os
from pathlib import Path

import numpy as np
import pytest

from worldscape_policy.data.video_cache import cached_video


def test_hit_signature_and_content_invalidation(tmp_path):
    source = tmp_path / "video"
    source.write_bytes(b"aaaa")
    cache = tmp_path / "cache"
    frames = np.arange(4*2*3*3, dtype=np.uint8).reshape(4,2,3,3)
    calls = []
    def decode():
        calls.append(1)
        return frames.copy()
    def read(signature="rgb24-v1"):
        return cached_video(source, cache, decode, signature=signature, max_bytes=2**20)
    np.testing.assert_array_equal(read(), frames)
    hit = read()
    np.testing.assert_array_equal(hit, frames)
    assert not hit.flags.writeable and len(calls) == 1
    read("different-resize")
    assert len(calls) == 2
    original = source.stat()
    source.write_bytes(b"bbbb")
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
    read()
    assert len(calls) == 3  # Content matters even when size/mtime are unchanged.


def test_full_cache_preserves_uncached_pixels(tmp_path):
    source = tmp_path / "video"
    source.write_bytes(b"source")
    frames = np.ones((2,3,4,3), dtype=np.uint8)
    result = cached_video(source, tmp_path/"cache", lambda: frames,
                          signature="test", max_bytes=1)
    np.testing.assert_array_equal(result, frames)
    assert not list((tmp_path/"cache").glob("*.npy"))


def test_corruption_is_not_silently_hidden(tmp_path):
    source = tmp_path / "video"
    source.write_bytes(b"source")
    cache = tmp_path / "cache"
    def read():
        return cached_video(source, cache, lambda: np.zeros((2,3,4,3), dtype=np.uint8),
                            signature="test", max_bytes=2**20)
    read()
    next(cache.glob("*.npy")).write_bytes(b"invalid")
    with pytest.raises(ValueError):
        read()


def test_concurrent_workers_decode_once(tmp_path):
    source = tmp_path / "video"
    source.write_bytes(b"source")
    calls = tmp_path / "calls"
    frames = np.arange(72, dtype=np.uint8).reshape(2,3,4,3)
    def worker():
        def decode():
            with calls.open("a") as f:
                f.write("decode\n")
            return frames.copy()
        result = cached_video(source, tmp_path/"cache", decode,
                              signature="test", max_bytes=2**20)
        np.testing.assert_array_equal(result, frames)
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=worker) for _ in range(4)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert calls.read_text().splitlines() == ["decode"]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_different_videos_share_one_capacity_limit(tmp_path):
    frames = np.ones((2, 3, 4, 3), dtype=np.uint8)
    cache = tmp_path / "cache"
    # Enough for one entry and its header, but not two reservations.
    limit = frames.nbytes + 4096
    sources = [tmp_path / f"video-{i}" for i in range(4)]
    for i, source in enumerate(sources):
        source.write_bytes(bytes([i]))

    def worker(source):
        actual = cached_video(source, cache, lambda: frames.copy(),
                              signature="same-decoder", max_bytes=limit)
        np.testing.assert_array_equal(actual, frames)

    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=worker, args=(source,)) for source in sources]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        files = list(cache.glob("*.npy"))
        assert len(files) == 1
        assert files[0].stat().st_size <= limit
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


@pytest.mark.parametrize("warm", [False, True])
def test_source_replaced_while_waiting_for_lock(tmp_path, monkeypatch, warm):
    import fcntl

    source = tmp_path / "video"
    source.write_bytes(b"old")
    frames = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    cache = tmp_path / "cache"
    def read():
        return cached_video(source, cache, lambda: frames,
                            signature="test", max_bytes=2**20)
    if warm:
        read()
    original = source.stat()
    real_flock = fcntl.flock
    def replace_then_lock(fd, operation):
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"new")
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        replacement.replace(source)
        real_flock(fd, operation)
    monkeypatch.setattr(fcntl, "flock", replace_then_lock)
    with pytest.raises(RuntimeError, match="Video changed"):
        read()


@pytest.mark.parametrize("journal", [b"-100000\n", b"10\n-20\n", b"12", b"bad\n"])
def test_invalid_quota_fails_before_publishing(tmp_path, journal):
    source = tmp_path / "video"
    source.write_bytes(b"source")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ".quota.lock").write_bytes(journal)
    with pytest.raises(ValueError):
        cached_video(source, cache, lambda: np.zeros((2, 3, 4, 3), dtype=np.uint8),
                     signature="test", max_bytes=1)
    assert not list(cache.glob("*.npy"))


def test_source_replaced_during_decode_is_not_cached(tmp_path):
    source = tmp_path / "video"
    source.write_bytes(b"old")
    before = source.stat()
    cache = tmp_path / "cache"
    def decode():
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"new")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(source)
        return np.zeros((2, 3, 4, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Video changed during decoding"):
        cached_video(source, cache, decode, signature="test", max_bytes=2**20)
    assert not list(cache.glob("*.npy"))


def test_unrelated_writes_do_not_hold_global_quota_lock(tmp_path, monkeypatch):
    """Both writers must reach np.save before either can finish its write."""
    import worldscape_policy.data.video_cache as module
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    frames = np.ones((2, 3, 4, 3), dtype=np.uint8)
    real_save = module.np.save
    def concurrent_save(*args, **kwargs):
        barrier.wait(timeout=5)
        return real_save(*args, **kwargs)
    monkeypatch.setattr(module.np, "save", concurrent_save)
    sources = [tmp_path / f"video-{i}" for i in range(2)]
    for i, source in enumerate(sources):
        source.write_bytes(bytes([i]))
    cache = tmp_path / "cache"
    limit = 2 * (frames.nbytes + 4096)
    def worker(source):
        actual = cached_video(source, cache, lambda: frames.copy(),
                              signature="test", max_bytes=limit)
        np.testing.assert_array_equal(actual, frames)
    processes = [context.Process(target=worker, args=(source,)) for source in sources]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert len(list(cache.glob("*.npy"))) == 2
        assert sum(map(int, (cache/".quota.lock").read_text().splitlines())) == limit
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_failed_write_keeps_reservation_and_no_partial_entry(tmp_path, monkeypatch):
    import worldscape_policy.data.video_cache as module
    source = tmp_path / "video"
    source.write_bytes(b"source")
    frames = np.ones((2, 3, 4, 3), dtype=np.uint8)
    cache = tmp_path / "cache"
    limit = frames.nbytes + 4096
    def failed_save(stream, *args, **kwargs):
        stream.write(b"partial")
        raise OSError("simulated disk failure")
    monkeypatch.setattr(module.np, "save", failed_save)
    with pytest.raises(OSError, match="simulated disk failure"):
        cached_video(source, cache, lambda: frames, signature="test", max_bytes=limit)
    assert not list(cache.glob("*.npy")) and not list(cache.glob("*.tmp"))
    assert sum(map(int, (cache/".quota.lock").read_text().splitlines())) == limit
    # The failed reservation consumes quota: retry decodes without publishing.
    np.testing.assert_array_equal(
        cached_video(source, cache, lambda: frames, signature="test", max_bytes=limit), frames)


def test_free_space_counts_unpublished_reservations(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import worldscape_policy.data.video_cache as module
    source = tmp_path / "video"
    source.write_bytes(b"source")
    frames = np.ones((2, 3, 4, 3), dtype=np.uint8)
    estimated = frames.nbytes + 4096
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / ".quota.lock").write_text(f"{estimated}\n")
    monkeypatch.setattr(module.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=2**30 + estimated))
    result = cached_video(source, cache, lambda: frames,
                          signature="test", max_bytes=3 * estimated)
    np.testing.assert_array_equal(result, frames)
    assert not list(cache.glob("*.npy"))
    assert (cache / ".quota.lock").read_text() == f"{estimated}\n"

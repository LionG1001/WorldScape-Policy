"""Camera concurrency must preserve pixels, ordering, and decoder failures."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from worldscape_policy.data.robocoin import RoboCoinSimEEF6DDataset


def dataset(tmp_path):
    value = object.__new__(RoboCoinSimEEF6DDataset)
    names = ('front', 'left', 'right')
    value.infos = [dict(features={name: {'dtype': 'video'} for name in names},
                        video_path='{video_key}.mp4', chunks_size=1000)]
    value.episodes = [{'episode_index': 0}]
    value.episode_roots = [tmp_path]
    value.cache_video_resize_width = value.cache_video_resize_height = None
    value._read_tabular_arrays = lambda _: {'tabular': np.arange(14)}
    for i, name in enumerate(names):
        (tmp_path/f'{name}.mp4').write_bytes(bytes([i]))
    return value, names


def fake_av(monkeypatch, names, barrier=None, fail=None):
    import av
    class Container:
        def __init__(self, path):
            self.name = path.rsplit('/', 1)[-1].split('.')[0]
            self.streams = SimpleNamespace(video=[SimpleNamespace(codec_context=SimpleNamespace(thread_count=0))])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def decode(self, **kwargs):
            if self.name == fail:
                raise RuntimeError('decoder failure')
            if barrier is not None:
                barrier.wait(timeout=5)
            pixels = np.full((2, 3, 3), names.index(self.name), dtype=np.uint8)
            return iter([SimpleNamespace(to_ndarray=lambda **kw: pixels.copy()) for _ in range(2)])
    monkeypatch.setattr(av, 'open', Container)


def test_parallel_cameras_match_serial_and_cache(tmp_path, monkeypatch):
    value, names = dataset(tmp_path)
    monkeypatch.delenv('WSP_ROBOCOIN_VIDEO_CACHE_DIR', raising=False)
    monkeypatch.delenv('WSP_ROBOCOIN_VIDEO_DECODE_THREADS', raising=False)
    fake_av(monkeypatch, names)
    expected = value._load_episode_arrays(0)
    monkeypatch.setenv('WSP_ROBOCOIN_VIDEO_DECODE_THREADS', '3')
    fake_av(monkeypatch, names, threading.Barrier(3))
    for cached in (False, True):
        if cached:
            monkeypatch.setenv('WSP_ROBOCOIN_VIDEO_CACHE_DIR', str(tmp_path/'cache'))
        actual = value._load_episode_arrays(0)
        assert list(actual) == list(expected)
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])
    import av
    def unexpected_decode(*args, **kwargs):
        raise AssertionError('warm cache must not decode')
    monkeypatch.setattr(av, 'open', unexpected_decode)
    actual = value._load_episode_arrays(0)
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_parallel_decoder_error_propagates(tmp_path, monkeypatch):
    value, names = dataset(tmp_path)
    monkeypatch.delenv('WSP_ROBOCOIN_VIDEO_CACHE_DIR', raising=False)
    monkeypatch.setenv('WSP_ROBOCOIN_VIDEO_DECODE_THREADS', '3')
    fake_av(monkeypatch, names, fail='left')
    with pytest.raises(RuntimeError, match='decoder failure'):
        value._load_episode_arrays(0)


@pytest.mark.parametrize('workers', ['0', '-1', '4'])
def test_invalid_camera_concurrency_rejected(tmp_path, monkeypatch, workers):
    value, _ = dataset(tmp_path)
    monkeypatch.setenv('WSP_ROBOCOIN_VIDEO_DECODE_THREADS', workers)
    with pytest.raises(ValueError, match='between 1 and 3'):
        value._load_episode_arrays(0)

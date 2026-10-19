"""Frame chunking must preserve augmented pixels and the caller's RNG."""
import numpy as np
import pytest

from worldscape_policy.data.augmentation import NativeVideoAugmentation


@pytest.mark.filterwarnings("error:The given NumPy array is not writable")
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('views', [False, True])
@pytest.mark.parametrize('bit_generator', [np.random.PCG64, np.random.Philox])
def test_chunking_preserves_pixels_rng_and_input(training, views, bit_generator):
    shape = (3, 3, 16, 24, 3) if views else (9, 16, 24, 3)
    values = np.random.default_rng(23).integers(0, 256, shape, dtype=np.uint8)
    original = values.copy()
    values.flags.writeable = False
    for seed in (0, 42, 62):
        rng = np.random.Generator(bit_generator(seed))
        options = dict(training=training, width=21, height=13)
        expected = NativeVideoAugmentation(**options)(values, rng=rng)
        following = rng.random(16)
        for chunk in (1, 4, 8, 20):
            other = np.random.Generator(bit_generator(seed))
            result = NativeVideoAugmentation(**options, frame_chunk_size=chunk)(values, rng=other)
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(other.random(16), following)
            np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize('chunk', [-1, 1.5, 4.0, '4', None, True, False])
def test_invalid_chunk_rejected_before_loading_video(chunk):
    with pytest.raises(ValueError, match='non-negative integer'):
        NativeVideoAugmentation(frame_chunk_size=chunk)


def test_numpy_integer_chunk_accepted():
    assert NativeVideoAugmentation(frame_chunk_size=np.int64(4)).frame_chunk_size == 4

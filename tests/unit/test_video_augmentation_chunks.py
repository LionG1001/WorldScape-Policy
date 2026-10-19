"""Frame chunking must preserve augmented pixels and the caller's RNG."""
import numpy as np
import pytest

from worldscape_policy.data.augmentation import NativeVideoAugmentation


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


def test_negative_chunk_rejected():
    with pytest.raises(ValueError, match='non-negative'):
        NativeVideoAugmentation(frame_chunk_size=-1)

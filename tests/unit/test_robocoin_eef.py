import numpy as np
import pytest

from worldscape_policy.action_space import rotation6d_to_matrix
from worldscape_policy.data.robocoin import bimanual_eef6d


def test_eef_converts_rotation_and_preserves_positions_grippers():
    pose = np.array([[1, 2, 3, 0, 0, np.pi/2, 4, 5, 6, 0, 0, 0]])
    result = bimanual_eef6d(pose, [[0.2, 0.8]])
    assert result.shape == (1, 20)
    np.testing.assert_allclose(result[0, [0, 1, 2, 9, 10, 11, 12, 19]],
                               [1, 2, 3, .2, 4, 5, 6, .8])
    np.testing.assert_allclose(rotation6d_to_matrix(result[:, 3:9])[0],
                               [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
    np.testing.assert_allclose(rotation6d_to_matrix(result[:, 13:19])[0], np.eye(3))


def test_rejects_joint_vectors_and_nonfinite_eef():
    with pytest.raises(ValueError, match='expected simulated EEF'):
        bimanual_eef6d(np.zeros((2, 14)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match='non-finite'):
        bimanual_eef6d(np.full((2, 12), np.nan), np.zeros((2, 2)))

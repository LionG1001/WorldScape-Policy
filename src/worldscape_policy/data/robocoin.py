"""Explicit RoboCOIN simulated-EEF adapter for the MUSA recipe.

Pose convention is configurable: the recipe uses extrinsic XYZ Euler radians.
This is a native adapter, not a copy of the customer's unpublished plugin.
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation

from worldscape_policy.action_space import matrix_to_rotation6d
from worldscape_policy.data.adapters.lerobot import NativeLeRobotDataset
from worldscape_policy.data.registry import DATASETS


def bimanual_eef6d(pose, gripper, *, rotation_representation="euler_xyz"):
    pose = np.asarray(pose, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 12 or gripper.shape != (len(pose), 2):
        raise ValueError("expected simulated EEF [T,12] and gripper [T,2]")
    if not np.isfinite(pose).all() or not np.isfinite(gripper).all():
        raise ValueError("non-finite RoboCOIN EEF/gripper values")
    arms = []
    for arm in range(2):
        xyz, angles = pose[:, arm*6:arm*6+3], pose[:, arm*6+3:arm*6+6]
        if rotation_representation == "euler_xyz":
            rotation = Rotation.from_euler("xyz", angles, degrees=False)
        elif rotation_representation == "rotvec":
            rotation = Rotation.from_rotvec(angles)
        else:
            raise ValueError("rotation_representation must be euler_xyz or rotvec")
        arms.append(np.concatenate((xyz, matrix_to_rotation6d(rotation.as_matrix()),
                                    gripper[:, arm:arm+1]), axis=-1))
    return np.concatenate(arms, axis=-1).astype(np.float32)


@DATASETS.register("robocoin_sim_eef6d_text")
class RoboCoinSimEEF6DDataset(NativeLeRobotDataset):
    def __init__(self, *args, rotation_representation="euler_xyz", step_stride=1, **kwargs):
        if step_stride < 1:
            raise ValueError("step_stride must be positive")
        self.rotation_representation = rotation_representation
        self.step_stride = int(step_stride)
        super().__init__(*args, **kwargs)

    def _build_index(self):
        return [(episode, anchor) for episode, anchor in super()._build_index()
                if anchor % self.step_stride == 0]

    def _read_tabular_arrays(self, episode_index):
        arrays = super()._read_tabular_arrays(episode_index)
        for suffix, destination in (("state", "robot_state"), ("action", "action.eef6d")):
            pose = np.stack(arrays[f"eef_sim_pose_{suffix}"])
            gripper = np.stack(arrays[f"gripper_open_scale_{suffix}"])
            arrays[destination] = bimanual_eef6d(
                pose, gripper, rotation_representation=self.rotation_representation)
        return arrays

    def _load_episode_arrays(self, episode_index):
        # Resize while decoding; avoid retaining entire 720p episodes per worker.
        import av
        import cv2
        cv2.setNumThreads(1)
        arrays = self._read_tabular_arrays(episode_index)
        info, episode = self.infos[episode_index], self.episodes[episode_index]
        eid = int(episode.get("episode_index", episode_index))
        decode_threads = int(os.environ.get("WSP_ROBOCOIN_VIDEO_DECODE_THREADS", "1"))
        if not 1 <= decode_threads <= 3:
            raise ValueError("video decode threads must be between 1 and 3")
        videos = [(name, feature) for name, feature in info["features"].items()
                  if feature.get("dtype") == "video"]

        def load_video(item):
            name, feature = item
            relative = info["video_path"].format(
                episode_index=eid, episode_chunk=eid//int(info.get("chunks_size", 1000)), video_key=name)
            path = self.episode_roots[episode_index]/relative
            def decode():
                frames = []
                with av.open(str(path)) as container:
                    container.streams.video[0].codec_context.thread_count = 2
                    for frame in container.decode(video=0):
                        rgb = frame.to_ndarray(format="rgb24")
                        if self.cache_video_resize_width is not None:
                            rgb = cv2.resize(rgb, (self.cache_video_resize_width,
                                                 self.cache_video_resize_height),
                                             interpolation=cv2.INTER_AREA)
                        frames.append(rgb)
                return np.stack(frames)

            cache = os.environ.get("WSP_ROBOCOIN_VIDEO_CACHE_DIR")
            if cache:
                from worldscape_policy.data.video_cache import cached_video
                frames = cached_video(
                    path, cache, decode,
                    signature=dict(format="rgb24-inter-area-v1", av=av.__version__,
                                   libraries=av.library_versions, cv2=cv2.__version__,
                                   width=self.cache_video_resize_width,
                                   height=self.cache_video_resize_height),
                    max_bytes=int(os.environ.get("WSP_ROBOCOIN_VIDEO_CACHE_GIB", "64")) * 2**30,
                )
            else:
                frames = decode()
            return name, frames

        if decode_threads > 1 and len(videos) > 1:
            # Each task owns its PyAV container. map preserves camera order;
            # decoding/resizing has no RNG and does not alter sampled frames.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(decode_threads, len(videos))) as pool:
                arrays.update(pool.map(load_video, videos))
        else:
            arrays.update(map(load_video, videos))
        return arrays

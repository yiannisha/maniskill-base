from __future__ import annotations

import numpy as np
import sapien
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


@register_env("PickCubeThirdPersonCamera-v0", max_episode_steps=50)
class PickCubeThirdPersonCameraEnv(PickCubeEnv):
    """PickCube with additional third-person and Panda wrist sensor cameras."""

    @property
    def _default_sensor_configs(self):
        sensor_configs = list(super()._default_sensor_configs)
        third_person_pose = sapien_utils.look_at(
            eye=(0.9, -1.1, 1.2),
            target=(0.0, 0.0, 0.2),
        )
        sensor_configs.append(
            CameraConfig(
                "third_person_camera",
                third_person_pose,
                640,
                480,
                np.pi / 3,
                0.01,
                100,
            )
        )
        wrist_mount = self.agent.robot.links_map.get("panda_hand")
        if wrist_mount is not None:
            sensor_configs.append(
                CameraConfig(
                    "wrist_camera",
                    sapien.Pose(
                        p=[0.0464982, -0.0200011, 0.0360011],
                        q=[0.0, 0.70710678, 0.0, 0.70710678],
                    ),
                    128,
                    128,
                    np.pi / 2,
                    0.01,
                    100,
                    mount=wrist_mount,
                )
            )
        return sensor_configs

from __future__ import annotations

import numpy as np
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


@register_env("PickCubeThirdPersonCamera-v0", max_episode_steps=50)
class PickCubeThirdPersonCameraEnv(PickCubeEnv):
    """PickCube with an additional fixed third-person sensor camera."""

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
        return sensor_configs

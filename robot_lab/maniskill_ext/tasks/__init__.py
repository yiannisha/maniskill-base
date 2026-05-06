"""Custom ManiSkill tasks live here."""

from robot_lab.maniskill_ext.tasks.pick_cube_third_person_camera import (
    PickCubeThirdPersonCameraEnv,
)
from robot_lab.maniskill_ext.tasks.pick_cube_with_guy import PickCubeWithGuyEnv

__all__ = ["PickCubeThirdPersonCameraEnv", "PickCubeWithGuyEnv"]

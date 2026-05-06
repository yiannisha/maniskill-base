from __future__ import annotations

from pathlib import Path

import sapien
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env

REPO_ROOT = Path(__file__).resolve().parents[3]
THE_GUY_URDF_PATH = REPO_ROOT / "the_guy" / "the_guy.urdf"


@register_env("PickCubeWithGuy-v0", max_episode_steps=50)
class PickCubeWithGuyEnv(PickCubeEnv):
    """PickCube with the_guy loaded as a fixed scene articulation."""

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        if not THE_GUY_URDF_PATH.is_file():
            raise FileNotFoundError(
                f"Expected extracted URDF at {THE_GUY_URDF_PATH}. "
                "Keep the extracted the_guy directory in the repo root."
            )

        loader = self.scene.create_urdf_loader()
        loader.name = "the_guy"
        loader.fix_root_link = True
        builder = loader.parse(str(THE_GUY_URDF_PATH))["articulation_builders"][0]
        builder.initial_pose = sapien.Pose(p=[0.35, -0.55, -0.9196429])
        self.the_guy = builder.build(name="the_guy")

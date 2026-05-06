from __future__ import annotations

from typing import Any

from robot_lab.policies.base import Policy
from robot_lab.policies.crazy_policy import CrazyPolicy
from robot_lab.policies.demo_replay_policy import DemoReplayPolicy

POLICY_REGISTRY: dict[str, type[Policy]] = {}


def register_policy(name: str, policy_cls: type[Policy]) -> None:
    POLICY_REGISTRY[name] = policy_cls


def build_policy(policy_cfg: dict[str, Any]) -> Policy:
    policy_type = policy_cfg.get("type", "random")
    params = policy_cfg.get("params", {})
    try:
        policy_cls = POLICY_REGISTRY[policy_type]
    except KeyError as exc:
        available = ", ".join(sorted(POLICY_REGISTRY))
        raise KeyError(f"Unknown policy type '{policy_type}'. Available: {available}") from exc
    return policy_cls(params=params)


from robot_lab.policies.random_policy import RandomPolicy
from robot_lab.policies.remote_policy_stub import RemotePolicyStub
from robot_lab.policies.scripted_policy import ScriptedPolicy

register_policy("crazy", CrazyPolicy)
register_policy("demo_replay", DemoReplayPolicy)
register_policy("random", RandomPolicy)
register_policy("scripted", ScriptedPolicy)
register_policy("remote_stub", RemotePolicyStub)

__all__ = [
    "CrazyPolicy",
    "DemoReplayPolicy",
    "POLICY_REGISTRY",
    "Policy",
    "RandomPolicy",
    "RemotePolicyStub",
    "ScriptedPolicy",
    "build_policy",
    "register_policy",
]

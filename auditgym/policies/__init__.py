"""Baseline policy adapters."""

from .attacks import FlagAllPolicy, OraclePolicy, RandomExplorerPolicy, SilentPolicy
from .expert import ExpertPolicy
from .reconciler import ReconcilerPolicy

POLICIES = {
    "random": RandomExplorerPolicy,
    "reconciler": ReconcilerPolicy,
    "flag_all": FlagAllPolicy,
    "oracle": OraclePolicy,
    "expert": ExpertPolicy,
    "silent": SilentPolicy,
}

# The four organiser-style baselines required by the case, plus the scripted expert demo.
BASELINES = ("random", "reconciler", "flag_all", "oracle")


def make_policy(name, seed=0):
    cls = POLICIES[name]
    if name == "random":
        return cls(seed=seed)
    return cls()

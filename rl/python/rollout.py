"""Runs episodes across several env processes with per-seat controllers."""
import random

import numpy as np
import torch

import spec
from env_client import EnvProc, Obs
from heuristics import BASELINES
from model import stack_obs


class Controller:
    """Wraps a torch Policy or a scripted baseline behind one call."""

    def __init__(self, policy=None, scripted=None, greedy=False):
        self.policy, self.scripted, self.greedy = policy, scripted, greedy

    @torch.no_grad()
    def act(self, obs_list, spawn=False):
        """-> (list of action dicts, logp[ n ] | None, value[ n ] | None)"""
        if self.scripted is not None:
            return BASELINES[self.scripted](obs_list), None, None
        b = stack_obs(obs_list)
        if spawn:
            a, lp, v = self.policy.act_spawn(b, self.greedy)
            acts = [{"kind": 0, "target": 0, "ratio": 0, "build": 0, "site": 0, "spawn": int(x)} for x in a]
            return acts, lp.numpy(), v.numpy()
        a, lp, v = self.policy.act(b, self.greedy)
        acts = [{k: int(a[k][i]) for k in ("kind", "target", "ratio", "build", "site")} | {"spawn": 0}
                for i in range(len(obs_list))]
        return acts, lp.numpy(), v.numpy()


def act_for_envs(per_env_obs, per_env_ctrl, spawn):
    """per_env_obs[e][i] is Obs|None; per_env_ctrl[e][i] a Controller.
    Batches by controller. Returns actions[e][i] (dict|None) and records[e][i]
    = (logp, value) when the controller is a policy."""
    groups = {}
    for e, (obs, ctrl) in enumerate(zip(per_env_obs, per_env_ctrl)):
        for i, o in enumerate(obs):
            if o is not None:
                groups.setdefault(id(ctrl[i]), (ctrl[i], []))[1].append((e, i, o))
    actions = [[None] * len(o) for o in per_env_obs]
    extra = [[None] * len(o) for o in per_env_obs]
    for ctrl, items in groups.values():
        acts, lp, v = ctrl.act([o for _, _, o in items], spawn)
        for j, (e, i, _) in enumerate(items):
            actions[e][i] = acts[j]
            if lp is not None:
                extra[e][i] = (float(lp[j]), float(v[j]))
    return actions, extra

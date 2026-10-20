import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import spec

NEG = -1e9


def stack_obs(obs_list):
    t = lambda xs: torch.from_numpy(np.stack(xs))
    return {
        "scalars": t([o.scalars for o in obs_list]),
        "slots": t([o.slots for o in obs_list]),
        "grid": t([o.grid for o in obs_list]),
        "m_kind": t([o.m_kind for o in obs_list]),
        "m_target": t([o.m_target for o in obs_list]),
        "m_build": t([o.m_build for o in obs_list]),
        "m_ally": t([o.m_ally for o in obs_list]),
    }


def masked_logits(logits, mask):
    return logits.masked_fill(mask < 0.5, NEG)


class Policy(nn.Module):
    """Shared actor-critic. Heads are factorised; which heads matter depends on
    the chosen `kind` (attack: target+ratio, build: build+site, ally: target)."""

    def __init__(self, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(spec.GRID_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 12x12
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),   # 6x6
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, hidden), nn.ReLU(),
        )
        self.vec = nn.Sequential(
            nn.Linear(spec.N_SCALARS + spec.N_SLOTS * spec.SLOT_FEATURES, hidden), nn.ReLU(),
        )
        self.trunk = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.h_kind = nn.Linear(hidden, spec.N_KINDS)
        self.h_target = nn.Linear(hidden, spec.N_TARGETS)
        self.h_ratio = nn.Linear(hidden, spec.N_RATIOS)
        self.h_build = nn.Linear(hidden, spec.N_BUILDS)
        self.h_site = nn.Linear(hidden, spec.N_SITES)
        self.h_spawn = nn.Linear(hidden, spec.N_SPAWN)
        # target slots are conditioned per-slot so "slot k" means the same thing everywhere
        self.slot_bias = nn.Sequential(nn.Linear(spec.SLOT_FEATURES, 16), nn.ReLU(), nn.Linear(16, 1))

    def encode(self, b):
        z = torch.cat([self.conv(b["grid"]), self.vec(torch.cat([b["scalars"], b["slots"]], -1))], -1)
        return self.trunk(z)

    def heads(self, b):
        h = self.encode(b)
        slots = b["slots"].view(-1, spec.N_SLOTS, spec.SLOT_FEATURES)
        sb = self.slot_bias(slots).squeeze(-1)  # [B, N_SLOTS]
        tl = self.h_target(h)
        tl = tl + F.pad(sb, (1, 0))  # slot 0 (wilderness) gets no per-slot bias
        return {
            "value": self.value(h).squeeze(-1),
            "kind": masked_logits(self.h_kind(h), b["m_kind"]),
            "target_attack": masked_logits(tl, b["m_target"]),
            "target_ally": masked_logits(tl, b["m_ally"]),
            "ratio": self.h_ratio(h),
            "build": masked_logits(self.h_build(h), b["m_build"]),
            "site": self.h_site(h),
            "spawn": self.h_spawn(h),
        }

    @staticmethod
    def _safe(logits):
        # rows with every action masked (can't happen for kind: noop is always legal)
        # fall back to uniform so Categorical stays valid
        bad = (logits.max(-1, keepdim=True).values <= NEG / 2)
        return torch.where(bad, torch.zeros_like(logits), logits)

    def act(self, b, greedy=False):
        """Sample actions. Returns (actions dict of tensors, logp, value)."""
        o = self.heads(b)
        d = {k: torch.distributions.Categorical(logits=self._safe(o[k]))
             for k in ("kind", "target_attack", "target_ally", "ratio", "build", "site")}
        pick = (lambda dist: dist.probs.argmax(-1)) if greedy else (lambda dist: dist.sample())
        a = {k: pick(v) for k, v in d.items()}
        a["target"] = torch.where(a["kind"] == 3, a["target_ally"], a["target_attack"])
        logp = self._logp(d, a)
        return a, logp, o["value"]

    @staticmethod
    def _logp(d, a):
        k = a["kind"]
        lp = d["kind"].log_prob(k)
        atk, bld, aly = (k == 1).float(), (k == 2).float(), (k == 3).float()
        lp = lp + atk * (d["target_attack"].log_prob(a["target"]) + d["ratio"].log_prob(a["ratio"]))
        lp = lp + bld * (d["build"].log_prob(a["build"]) + d["site"].log_prob(a["site"]))
        lp = lp + aly * d["target_ally"].log_prob(a["target"])
        return lp

    def evaluate(self, b, a):
        o = self.heads(b)
        d = {k: torch.distributions.Categorical(logits=self._safe(o[k]))
             for k in ("kind", "target_attack", "target_ally", "ratio", "build", "site")}
        logp = self._logp(d, a)
        k = a["kind"]
        atk, bld, aly = (k == 1).float(), (k == 2).float(), (k == 3).float()
        ent = d["kind"].entropy()
        ent = ent + atk * (d["target_attack"].entropy() + d["ratio"].entropy())
        ent = ent + bld * (d["build"].entropy() + d["site"].entropy())
        ent = ent + aly * d["target_ally"].entropy()
        return logp, ent, o["value"]

    def act_spawn(self, b, greedy=False):
        o = self.heads(b)
        dist = torch.distributions.Categorical(logits=o["spawn"])
        a = dist.probs.argmax(-1) if greedy else dist.sample()
        return a, dist.log_prob(a), o["value"]

    def evaluate_spawn(self, b, a):
        o = self.heads(b)
        dist = torch.distributions.Categorical(logits=o["spawn"])
        return dist.log_prob(a), dist.entropy(), o["value"]

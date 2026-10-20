"""Scripted opponents / baselines. Same interface as the neural policy:
take a list of Obs, return a list of action dicts."""
import random

import spec


def _blank():
    return {"kind": 0, "target": 0, "ratio": 1, "build": 0, "site": 0, "spawn": 0}


def random_agent(obs_list, rng=random):
    out = []
    for o in obs_list:
        a = _blank()
        kinds = [k for k in range(spec.N_KINDS) if o.m_kind[k] > 0.5]
        a["kind"] = rng.choice(kinds)
        pool = o.m_ally if a["kind"] == 3 else o.m_target
        ts = [t for t in range(spec.N_TARGETS) if pool[t] > 0.5]
        a["target"] = rng.choice(ts) if ts else 0
        bs = [b for b in range(spec.N_BUILDS) if o.m_build[b] > 0.5]
        a["build"] = rng.choice(bs) if bs else 0
        a["ratio"] = rng.randrange(spec.N_RATIOS)
        a["site"] = rng.randrange(spec.N_SITES)
        a["spawn"] = rng.randrange(spec.N_SPAWN)
        out.append(a)
    return out


def rush_agent(obs_list, rng=random):
    """Expand into wilderness, then hit the weakest non-ally neighbour; build
    cities/ports whenever affordable. A reasonable scripted baseline."""
    out = []
    for o in obs_list:
        a = _blank()
        a["spawn"] = rng.randrange(spec.N_SPAWN)
        if o.m_kind[2] > 0.5 and rng.random() < 0.35:
            a["kind"] = 2
            a["build"] = next((b for b in (0, 1, 3, 2) if o.m_build[b] > 0.5), 0)
            a["site"] = rng.randrange(spec.N_SITES)
        elif o.m_kind[1] > 0.5:
            a["kind"] = 1
            if o.m_target[0] > 0.5:
                a["target"], a["ratio"] = 0, 1
            else:
                best, bt = 9e9, 0
                for k in range(spec.N_SLOTS):
                    f = k * spec.SLOT_FEATURES
                    if o.slots[f] > 0.5 and o.m_target[k + 1] > 0.5 and o.slots[f + 2] < best:
                        best, bt = o.slots[f + 2], k + 1
                a["target"] = bt
                a["ratio"] = 2 if best < 0 else 1
                if bt == 0:
                    a["kind"] = 0
        out.append(a)
    return out


BASELINES = {"random": random_agent, "rush": rush_agent}

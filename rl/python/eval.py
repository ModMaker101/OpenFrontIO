"""Play fixed seat line-ups and report per-seat results.

  python eval.py --seats policy:runs/a/best.pt,rush,rush,random --episodes 20
Seat kinds: random | rush | policy:<checkpoint>
"""
import argparse
import random

import numpy as np
import torch

import spec
from env_client import EnvProc
from model import Policy
from rollout import Controller, act_for_envs


def make(kind, greedy):
    if kind.startswith("policy:"):
        p = Policy()
        p.load_state_dict(torch.load(kind[7:])["model"])
        p.eval()
        return Controller(policy=p, greedy=greedy)
    return Controller(scripted=kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seats", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--maps", default="iceland")
    ap.add_argument("--bots", type=int, default=20)
    ap.add_argument("--max-ticks", type=int, default=6000)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    seats = args.seats.split(",")
    n = len(seats)
    cache = {}
    ctrls = []
    for s in seats:
        cache.setdefault(s, make(s, args.greedy))
        ctrls.append(cache[s])
    maps = args.maps.split(",")
    base = {"numAgents": n, "bots": args.bots, "nations": "default", "decisionInterval": 10,
            "maxTicks": args.max_ticks, "difficulty": "Medium"}
    results = []
    todo = list(range(args.episodes))
    procs = [EnvProc(base) for _ in range(min(args.envs, args.episodes))]

    def start(p):
        ep = todo.pop(0)
        p.opts = dict(base, map=maps[ep % len(maps)])
        p.send_reset(f"eval-{args.seed}-{ep}")
        # rotate seats so no controller keeps a fixed spawn slot
        rot = ep % n
        p.rot = rot
        p.obs = EnvProc.obs_list(p.recv())
        seat_ctrls = [ctrls[(i + rot) % n] for i in range(n)]
        p.seat = [(i + rot) % n for i in range(n)]
        acts, _ = act_for_envs([p.obs], [seat_ctrls], spawn=True)
        p.send_spawn(acts[0])
        p.obs = EnvProc.obs_list(p.recv())
        p.ctrls = seat_ctrls
        p.live = True

    for p in procs:
        start(p)
    active = list(procs)
    while active:
        acts, _ = act_for_envs([p.obs for p in active], [p.ctrls for p in active], spawn=False)
        for p, a in zip(active, acts):
            p.send_step(a)
        for p, a in zip(list(active), acts):
            res = p.recv()
            p.obs = EnvProc.obs_list(res)
            if res["episodeDone"]:
                info = res["info"]
                for i in range(n):
                    results.append((p.seat[i], info["shares"][i], info["ranks"][i], 1.0 if info["winner"] == i else 0.0, info["peak"][i], info["survival"][i]))
                if todo:
                    start(p)
                else:
                    active.remove(p)
    for p in procs:
        p.close()
    r = np.asarray(results)
    print(f"{len(results)//n} episodes, maps={maps}")
    for s in sorted(set(seats)):
        idx = [k for k, x in enumerate(seats) if x == s]
        m = np.isin(r[:, 0], idx)
        print(f"{s:34s} peak_share={r[m,4].mean():.4f} survival={r[m,5].mean():.2f} rank={r[m,2].mean():.2f} win={r[m,3].mean():.2f}")


if __name__ == "__main__":
    main()

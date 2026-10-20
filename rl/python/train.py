"""PPO self-play on OpenFront multiplayer games.

Every game has `--agents` learner seats in one multiplayer-config match (plus
the map's nations and tribes). All seats share one network. With probability
`--p-old` a non-first seat is instead played by a frozen earlier snapshot
(a small league), which keeps the learner from over-fitting to its own
current quirks. Only learner-seat experience is trained on.
"""
import argparse
import copy
import csv
import os
import random
import time

import numpy as np
import torch

import spec
from env_client import EnvProc
from heuristics import BASELINES
from model import Policy, stack_obs
from rollout import Controller, act_for_envs

ACTION_KEYS = ("kind", "target", "ratio", "build", "site")


class Traj:
    __slots__ = ("obs", "act", "logp", "val", "rew", "done", "spawn")

    def __init__(self):
        self.obs, self.act, self.logp, self.val, self.rew, self.done, self.spawn = [], [], [], [], [], [], []

    def __len__(self):
        return len(self.obs)


def gae(traj, last_val, gamma, lam):
    n = len(traj)
    adv = np.zeros(n, np.float32)
    g = 0.0
    for t in reversed(range(n)):
        nv = last_val if t == n - 1 else traj.val[t + 1]
        nonterm = 0.0 if traj.done[t] else 1.0
        delta = traj.rew[t] + gamma * nv * nonterm - traj.val[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    ret = adv + np.asarray(traj.val, np.float32)
    return adv, ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--bots", type=int, default=20)
    ap.add_argument("--maps", default="iceland")
    ap.add_argument("--max-ticks", type=int, default=6000)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--steps-per-update", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--p-old", type=float, default=0.25)
    ap.add_argument("--pool-every", type=int, default=10)
    ap.add_argument("--pool-size", type=int, default=6)
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    maps = args.maps.split(",")

    policy = Policy()
    if args.resume:
        policy.load_state_dict(torch.load(args.resume)["model"])
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    learner = Controller(policy=policy)
    pool = []  # frozen Controllers

    base = {"numAgents": args.agents, "bots": args.bots, "nations": "default",
            "decisionInterval": 10, "maxTicks": args.max_ticks, "difficulty": "Medium"}
    envs = [EnvProc(base) for _ in range(args.envs)]
    ep_counter = 0

    class EnvState:
        pass

    states = []

    def new_episode(e):
        """Resets env e and runs its spawn step. Blocking."""
        nonlocal ep_counter
        s = states[e]
        ep_counter += 1
        opts = dict(base, map=random.choice(maps))
        envs[e].opts = opts
        envs[e].send_reset(f"ep{args.seed}-{ep_counter}")
        obs = EnvProc.obs_list(envs[e].recv())
        ctrls = [learner] + [random.choice(pool) if pool and random.random() < args.p_old else learner
                             for _ in range(args.agents - 1)]
        acts, extra = act_for_envs([obs], [ctrls], spawn=True)
        for i in range(args.agents):
            if ctrls[i] is learner:
                tr = s.trajs[i] = Traj()
                tr.obs.append(obs[i]); tr.act.append(acts[0][i]); tr.logp.append(extra[0][i][0])
                tr.val.append(extra[0][i][1]); tr.rew.append(0.0); tr.done.append(False); tr.spawn.append(True)
            else:
                s.trajs[i] = None
        envs[e].send_spawn(acts[0])
        res = envs[e].recv()
        s.ctrls = ctrls
        s.obs = EnvProc.obs_list(res)
        s.finished_agents = [False] * args.agents
        # the spawn transition's "next" is the first in-game obs; mark agents dead on arrival
        for i in range(args.agents):
            if res["done"][i] and s.trajs[i] is not None:
                s.trajs[i].done[-1] = True
                s.finished_agents[i] = True
        s.ep_return = np.zeros(args.agents)
        s.map = opts["map"]

    for e in range(args.envs):
        st = EnvState()
        st.trajs = [None] * args.agents
        states.append(st)
        new_episode(e)

    finished = []   # completed Traj objects
    ep_stats = []   # per finished episode (learner seats only)
    log = open(os.path.join(args.out, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["update", "env_steps", "episodes", "ep_return", "peak_share", "survival", "win_rate",
                "pg_loss", "v_loss", "entropy", "sps"])
    best = -1e9
    env_steps = 0
    t_start = time.time()

    def flush_batch():
        data_obs, data_act, data_logp, data_adv, data_ret, data_spawn = [], [], [], [], [], []
        # bootstrap values for still-running trajectories
        live = [(e, i) for e in range(args.envs) for i in range(args.agents)
                if states[e].trajs[i] is not None and len(states[e].trajs[i]) > 0
                and not states[e].finished_agents[i] and states[e].obs[i] is not None]
        boots = {}
        if live:
            b = stack_obs([states[e].obs[i] for e, i in live])
            with torch.no_grad():
                v = policy.heads(b)["value"].numpy()
            boots = {k: float(v[j]) for j, k in enumerate(live)}
        todo = [(t, 0.0) for t in finished]
        for (e, i), v in boots.items():
            todo.append((states[e].trajs[i], v))
        for tr, lv in todo:
            adv, ret = gae(tr, lv, args.gamma, args.lam)
            data_obs += tr.obs; data_act += tr.act; data_logp += tr.logp
            data_adv.append(adv); data_ret.append(ret); data_spawn += tr.spawn
        for e in range(args.envs):
            for i in range(args.agents):
                if states[e].trajs[i] is not None and not states[e].finished_agents[i]:
                    states[e].trajs[i] = Traj()
        finished.clear()
        return (data_obs, data_act, np.asarray(data_logp, np.float32), np.concatenate(data_adv),
                np.concatenate(data_ret), np.asarray(data_spawn))

    pending_steps = 0
    update = 0
    while update < args.updates:
        # 1. act
        per_obs = [s.obs for s in states]
        per_ctrl = [s.ctrls for s in states]
        acts, extra = act_for_envs(per_obs, per_ctrl, spawn=False)
        for e in range(args.envs):
            envs[e].send_step([acts[e][i] for i in range(args.agents)])
        results = [envs[e].recv() for e in range(args.envs)]
        # 2. record
        for e, res in enumerate(results):
            s = states[e]
            for i in range(args.agents):
                tr = s.trajs[i]
                if tr is None or s.obs[i] is None or s.finished_agents[i]:
                    continue
                tr.obs.append(s.obs[i]); tr.act.append(acts[e][i]); tr.logp.append(extra[e][i][0])
                tr.val.append(extra[e][i][1]); tr.rew.append(res["reward"][i]); tr.done.append(bool(res["done"][i]))
                tr.spawn.append(False)
                s.ep_return[i] += res["reward"][i]
                pending_steps += 1
                env_steps += 1
                if res["done"][i]:
                    s.finished_agents[i] = True
                    finished.append(tr)
                    s.trajs[i] = None
            s.obs = EnvProc.obs_list(res)
            if res["episodeDone"]:
                info = res["info"]
                for i in range(args.agents):
                    if s.ctrls[i] is learner:
                        ep_stats.append((s.ep_return[i], info["peak"][i], info["survival"][i],
                                         1.0 if info["winner"] == i else 0.0))
                for i in range(args.agents):
                    if s.trajs[i] is not None and not s.finished_agents[i] and len(s.trajs[i]) > 0:
                        s.trajs[i].done[-1] = True
                        finished.append(s.trajs[i])
                new_episode(e)

        # 3. update
        if pending_steps >= args.steps_per_update:
            pending_steps = 0
            update += 1
            d_obs, d_act, d_logp, d_adv, d_ret, d_spawn = flush_batch()
            n = len(d_obs)
            B = stack_obs(d_obs)
            A = {k: torch.tensor([a[k] for a in d_act]) for k in ACTION_KEYS}
            A_spawn = torch.tensor([a["spawn"] for a in d_act])
            old_lp = torch.from_numpy(d_logp)
            adv = torch.from_numpy(d_adv); ret = torch.from_numpy(d_ret)
            sp = torch.from_numpy(d_spawn)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_l = v_l = ent_l = 0.0; cnt = 0
            for _ in range(args.epochs):
                perm = torch.randperm(n)
                for st in range(0, n, args.minibatch):
                    idx = perm[st:st + args.minibatch]
                    mb = {k: v[idx] for k, v in B.items()}
                    ma = {k: v[idx] for k, v in A.items()}
                    msp = sp[idx]
                    lp = torch.zeros(len(idx)); ent = torch.zeros(len(idx)); val = torch.zeros(len(idx))
                    r = ~msp
                    if r.any():
                        l, en, v = policy.evaluate({k: x[r] for k, x in mb.items()}, {k: x[r] for k, x in ma.items()})
                        lp[r], ent[r], val[r] = l, en, v
                    if msp.any():
                        l, en, v = policy.evaluate_spawn({k: x[msp] for k, x in mb.items()}, A_spawn[idx][msp])
                        lp[msp], ent[msp], val[msp] = l, en, v
                    ratio = (lp - old_lp[idx]).exp()
                    a_ = adv[idx]
                    pg = -torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_).mean()
                    vl = 0.5 * ((val - ret[idx]) ** 2).mean()
                    loss = pg + 0.5 * vl - args.ent * ent.mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    opt.step()
                    pg_l += pg.item(); v_l += vl.item(); ent_l += ent.mean().item(); cnt += 1

            if update % args.pool_every == 0:
                frozen = copy.deepcopy(policy)
                frozen.eval()
                pool.append(Controller(policy=frozen))
                pool[:] = pool[-args.pool_size:]

            if ep_stats:
                es = np.asarray(ep_stats, np.float64)
                row = [update, env_steps, ep_counter, es[:, 0].mean(), es[:, 1].mean(), es[:, 2].mean(), es[:, 3].mean()]
            else:
                row = [update, env_steps, ep_counter, 0, 0, 0, 0]
            sps = env_steps / (time.time() - t_start)
            w.writerow(row + [pg_l / cnt, v_l / cnt, ent_l / cnt, sps]); log.flush()
            print(f"[{update:4d}] steps={env_steps} eps={ep_counter} ret={row[3]:.3f} peak={row[4]:.4f} "
                  f"surv={row[5]:.2f} win={row[6]:.2f} pg={pg_l/cnt:.3f} v={v_l/cnt:.3f} ent={ent_l/cnt:.2f} sps={sps:.0f}", flush=True)
            torch.save({"model": policy.state_dict(), "args": vars(args)}, os.path.join(args.out, "latest.pt"))
            if ep_stats and row[3] > best:
                best = row[3]
                torch.save({"model": policy.state_dict(), "args": vars(args)}, os.path.join(args.out, "best.pt"))
            ep_stats.clear()

    for e in envs:
        e.close()


if __name__ == "__main__":
    main()

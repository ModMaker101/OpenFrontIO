# OpenFront RL

Reinforcement learning for OpenFront: several learning agents play the same
multiplayer-config game against each other, the map's nations and tribes, and
frozen earlier versions of themselves.

```
rl/env/OpenFrontEnv.ts   multi-agent env around the real engine (GameRunner + Executor)
rl/env/server.ts         JSON-lines bridge, one env per process
rl/python/               PPO self-play (PyTorch): train.py, eval.py, model.py, heuristics.py
tests/rl/                env smoke tests (npx vitest tests/rl --run)
```

## What "multiplayer" means here

`OpenFrontEnv` builds the game exactly like a public lobby: `GameType.Public`,
the production `EngineConfig`, a spawn phase, the map's nations, tribes. The
learners are `PlayerType.Human` players identified by client ids. They act by
emitting the same intents a client sends (`spawn`, `attack`, `build_unit`,
`allianceRequest`), which the real `Executor` turns into executions. Nothing in
the rules is simplified, and the game is bit-for-bit what a human match runs.

It is **not** connected to the live OpenFront servers. Training needs
hundreds of thousands of games; running that against public lobbies would be
slow, unfair to the people in them, and presumably against the rules. The same
policy could be driven through a client connection later (see "Next steps").

## Interface

One `step()` = 10 ticks (1 s of game time).

- **Observation** per agent: 24 scalars (territory share, troops, gold, unit
  counts, attacks in/out, ...), 8 neighbour-player slots x 8 features, and a
  5x24x24 map image (self / ally / other / free land / water).
- **Actions** (factorised, with legality masks): `noop`, `attack` (target =
  wilderness or a neighbour slot, 10/25/50/90 % of troops), `build`
  (City / Port / DefensePost / Factory, on a random or border tile),
  `ally` (request an alliance with a neighbour). The first step of a game is a
  spawn pick on a 16x16 grid; rejected picks are retried while the spawn phase
  is open.
- **Reward**: 20 x change in land share, -1 on elimination, and at the end a
  placement bonus in [-1, 1] among the agents plus +3 for winning. Placement
  ranks survivors by territory and the dead by how long they lasted.

## Running it

```bash
npm run inst                         # (Node 24; on Node 22 use: npm ci --ignore-scripts --engine-strict=false)
python3 -m venv .venv && .venv/bin/pip install -r rl/python/requirements.txt

cd rl/python
../../.venv/bin/python train.py --envs 4 --agents 4 --maps iceland --out runs/a
../../.venv/bin/python eval.py --seats policy:runs/a/best.pt,rush,rush,random --episodes 20
```

`train.py` logs one CSV row per update (`runs/<name>/log.csv`) and writes
`latest.pt` / `best.pt`. Useful flags: `--maps a,b,c` (random map per game),
`--p-old` (share of non-first seats played by frozen snapshots),
`--agents`, `--bots`, `--max-ticks`, `--resume`.

`eval.py` rotates seats between games and reports peak land share, survival
(fraction of the time limit) and win rate per seat kind. `rush` is a scripted
expand-then-attack baseline, `random` picks legal actions uniformly.

## Next steps

- Naval play (transport ships), nukes, and unit upgrades are not in the action
  space yet.
- A WebSocket client adapter to run a trained policy on a private/dev server.
- Bigger maps and team modes; population-based opponents instead of the small
  snapshot pool.

import { Difficulty } from "@openfront/engine-api/game/GameTypes";
import { describe, expect, it } from "vitest";
import {
  AgentAction,
  GRID,
  GRID_CHANNELS,
  N_SCALARS,
  N_SLOTS,
  N_TARGETS,
  OpenFrontEnv,
  SLOT_FEATURES,
  SPAWN_CELLS,
} from "../../rl/env/OpenFrontEnv";

const opts = {
  map: "iceland",
  numAgents: 3,
  bots: 10,
  nations: "default" as const,
  decisionInterval: 10,
  maxTicks: 600,
  difficulty: Difficulty.Medium,
};

const noop: AgentAction = {
  kind: 0,
  target: 0,
  ratio: 0,
  build: 0,
  site: 0,
  spawn: 0,
};

describe("OpenFrontEnv", () => {
  it("spawns every agent into a multiplayer game and steps to the time limit", async () => {
    const env = new OpenFrontEnv(opts);
    const obs = await env.reset("rl-test");
    expect(obs).toHaveLength(3);
    const o = obs[0];
    expect(o.scalars).toHaveLength(N_SCALARS);
    expect(o.slots).toHaveLength(N_SLOTS * SLOT_FEATURES);
    expect(o.grid).toHaveLength(GRID_CHANNELS * GRID * GRID);
    expect(o.masks.target).toHaveLength(N_TARGETS);
    expect(o.masks.spawn).toHaveLength(SPAWN_CELLS * SPAWN_CELLS);

    let res = env.spawn(obs.map((_, i) => ({ ...noop, spawn: 40 + 70 * i })));
    expect(res.done).toEqual([false, false, false]);
    expect(res.obs.every((x) => x !== null)).toBe(true);

    let steps = 0;
    while (!res.episodeDone && steps < 200) {
      res = env.step(res.obs.map((x) => (x === null ? null : noop)));
      steps++;
    }
    expect(res.episodeDone).toBe(true);
    expect(steps).toBe(60);
    expect(res.info.fatal).toBeUndefined();
  }, 60_000);

  it("attacking unclaimed land grows territory; masks forbid impossible actions", async () => {
    const env = new OpenFrontEnv({
      ...opts,
      numAgents: 1,
      bots: 0,
      nations: "disabled",
    });
    const obs = await env.reset("rl-test-2");
    let res = env.spawn([{ ...noop, spawn: 100 }]);
    const first = res.obs[0]!;
    expect(first.masks.kind[1]).toBe(1); // wilderness is attackable
    expect(first.masks.target[0]).toBe(1);
    expect(first.masks.kind[3]).toBe(0); // nobody to ally with
    const before = first.scalars[1];
    const attack: AgentAction = { ...noop, kind: 1, target: 0, ratio: 2 };
    for (let i = 0; i < 20; i++) res = env.step([attack]);
    expect(res.obs[0]!.scalars[1]).toBeGreaterThan(before);
    expect(obs).toHaveLength(1);
  }, 60_000);
});

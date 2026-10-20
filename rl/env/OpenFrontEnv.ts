import { TileRef } from "@openfront/engine-api/game/GameMap";
/**
 * Multi-agent RL environment around the real OpenFront simulation.
 *
 * Several learner-controlled players ("agents") share one multiplayer-config
 * game (GameType.Public: spawn phase, nations, tribes, production rules) with
 * the built-in nation AI and bots as the rest of the field. Agents act by
 * emitting the same intents a human client would send; the Executor turns them
 * into executions exactly as in a real match, so nothing about the rules is
 * simplified.
 *
 * One env.step() = DECISION_INTERVAL game ticks. Every agent gets a structured
 * observation, a set of action masks and a scalar reward per step.
 */
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  PlayerInfo,
  PlayerType,
  TerraNullius,
  UnitType,
} from "@openfront/engine-api/game/GameTypes";
import {
  GameConfig,
  GameStartInfo,
  StampedIntent,
} from "@openfront/engine-api/Schemas";
import { loadTerrainMap } from "@openfront/engine-lib/game/TerrainMapLoader";
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { EngineConfig } from "@openfront/engine/configuration/EngineConfig";
import { Executor } from "@openfront/engine/execution/ExecutionManager";
import { Game, Player } from "@openfront/engine/game/Game";
import { createGame } from "@openfront/engine/game/GameImpl";
import { createNationsForGame } from "@openfront/engine/game/NationCreation";
import { GameRunner } from "@openfront/engine/GameRunner";
import { loadMapFiles } from "@openfront/shared/GameMapLoader";
import path from "path";
import { fileURLToPath } from "url";
import { NodeGameMapLoader } from "../../tests/perf/fullgame/NodeGameMapLoader";

const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

// ── Observation / action layout (mirrored in rl/python/spec.py) ──
export const GRID = 24; // map is summarised as a GRID x GRID image
export const GRID_CHANNELS = 5; // self, ally, other player, free land, water
export const N_SCALARS = 24;
export const N_SLOTS = 8; // nearest neighbouring players the agent can target
export const SLOT_FEATURES = 8;
export const SPAWN_CELLS = 16; // spawn is chosen on a SPAWN_CELLS x SPAWN_CELLS grid

export const KINDS = ["noop", "attack", "build", "ally"] as const;
export const BUILDS = [
  UnitType.City,
  UnitType.Port,
  UnitType.DefensePost,
  UnitType.Factory,
] as const;
export const RATIOS = [0.1, 0.25, 0.5, 0.9] as const;
export const SITES = ["random", "border"] as const;
// target slot 0 = unclaimed wilderness, 1..N_SLOTS = neighbouring players
export const N_TARGETS = N_SLOTS + 1;

export interface EnvOptions {
  map: string;
  numAgents: number;
  bots: number;
  /** "default" = the map's nations, "disabled", or a count. */
  nations: "default" | "disabled" | number;
  /** Ticks per env.step(). 10 ticks = 1 second of game time. */
  decisionInterval: number;
  /** Episode is truncated after this many game ticks (after spawn). */
  maxTicks: number;
  difficulty: Difficulty;
}

export const DEFAULT_OPTIONS: EnvOptions = {
  map: "plains",
  numAgents: 4,
  bots: 20,
  nations: "default",
  decisionInterval: 10,
  maxTicks: 6000,
  difficulty: Difficulty.Medium,
};

export interface AgentAction {
  kind: number;
  target: number;
  ratio: number;
  build: number;
  site: number;
  /** Only read during the spawn step. */
  spawn: number;
}

export interface AgentObs {
  scalars: number[];
  slots: number[]; // N_SLOTS * SLOT_FEATURES
  grid: number[]; // GRID_CHANNELS * GRID * GRID
  masks: {
    kind: number[];
    target: number[];
    build: number[];
    ally_target: number[];
    spawn: number[];
  };
}

export interface StepResult {
  obs: (AgentObs | null)[];
  reward: number[];
  /** per agent: eliminated or episode over */
  done: boolean[];
  episodeDone: boolean;
  info: Record<string, unknown>;
}

function resolveMap(name: string): GameMapType {
  const key = Object.keys(GameMapType).find(
    (k) => k.toLowerCase() === name.toLowerCase(),
  );
  if (key === undefined) throw new Error(`unknown map "${name}"`);
  return GameMapType[key as keyof typeof GameMapType];
}

export class OpenFrontEnv {
  private game!: Game;
  private runner!: GameRunner;
  private executor!: Executor;
  private agents: Player[] = [];
  private clientIDs: string[] = [];
  private turn = 0;
  private spawnTicks = 0;
  private startTicks = 0;
  private prevShare: number[] = [];
  private peakShare: number[] = [];
  private diedAt: number[] = [];
  private alive: boolean[] = [];
  private landTiles = 1;
  private rng = new PseudoRandom(1);
  private fatal: string | undefined;
  private episodeDone = false;
  private spawned = false;

  constructor(readonly opts: EnvOptions = DEFAULT_OPTIONS) {}

  // Not cached: a game takes over (and mutates) the maps it is built from.
  private async terrain(map: GameMapType, size: GameMapSize) {
    const loader = new NodeGameMapLoader(path.join(ROOT, "resources/maps"));
    return loadTerrainMap(await loadMapFiles(loader, map, size));
  }

  /** Starts a new game; returns the spawn-phase observations. */
  async reset(seed: string): Promise<AgentObs[]> {
    console.debug = () => {};
    const o = this.opts;
    const gameMap = resolveMap(o.map);
    const gameConfig: GameConfig = {
      gameMap,
      gameMapSize: GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Public,
      difficulty: o.difficulty,
      nations: o.nations,
      donateGold: false,
      donateTroops: false,
      bots: o.bots,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
    };
    const humans: PlayerInfo[] = [];
    const random = new PseudoRandom(simpleHash(seed));
    this.clientIDs = [];
    for (let i = 0; i < o.numAgents; i++) {
      const cid = `AGENT${String(i).padStart(3, "0")}`;
      this.clientIDs.push(cid);
      humans.push(
        new PlayerInfo(`agent${i}`, PlayerType.Human, cid, random.nextID()),
      );
    }
    const start: GameStartInfo = {
      gameID: seed,
      lobbyCreatedAt: 0,
      config: gameConfig,
      players: [],
    };
    const t = await this.terrain(gameMap, GameMapSize.Normal);
    const nations = createNationsForGame(
      start,
      t.nations,
      t.additionalNations,
      humans.length,
      random,
    );
    this.game = createGame(
      humans,
      nations,
      t.gameMap,
      t.miniGameMap,
      new EngineConfig(gameConfig, false),
      t.teamGameSpawnAreas,
    );
    this.executor = new Executor(this.game, seed, undefined);
    this.fatal = undefined;
    this.runner = new GameRunner(this.game, this.executor, (gu) => {
      if ("errMsg" in gu) this.fatal = `${gu.errMsg}\n${gu.stack ?? ""}`;
    });
    this.runner.init();
    this.turn = 0;
    this.rng = new PseudoRandom(simpleHash(seed) + 7);
    this.agents = this.clientIDs.map((c) => this.game.playerByClientID(c)!);
    this.landTiles = Math.max(1, this.game.numLandTiles());
    this.alive = this.agents.map(() => true);
    this.prevShare = this.agents.map(() => 0);
    this.peakShare = this.agents.map(() => 0);
    this.diedAt = this.agents.map(() => -1);
    this.episodeDone = false;
    this.spawned = false;
    // Tick once so the game is live, still inside the spawn phase.
    this.tick([]);
    return this.agents.map((_, i) => this.observe(i));
  }

  /**
   * Spawn step: each agent picks a cell of the SPAWN_CELLS grid. Runs the
   * remaining spawn phase and returns the first in-game observations.
   */
  spawn(actions: AgentAction[]): StepResult {
    const g = this.game;
    // A pick can be rejected (not enough free land around it, or too close to
    // another player). Retry those agents on random cells while the spawn
    // phase is still open, like a human re-clicking.
    let cells = actions.map((a) => a.spawn);
    for (let attempt = 0; attempt < 8 && g.inSpawnPhase(); attempt++) {
      const intents: StampedIntent[] = [];
      this.agents.forEach((p, i) => {
        if (p.hasSpawned() && attempt > 0) return;
        const tile = this.spawnTile(cells[i]);
        if (tile !== null) {
          intents.push({
            type: "spawn",
            tile,
            clientID: this.clientIDs[i],
          } as StampedIntent);
        }
      });
      this.tick(intents);
      this.tick([]);
      if (this.agents.every((p) => p.hasSpawned())) break;
      cells = cells.map(() => this.rng.nextInt(0, SPAWN_CELLS * SPAWN_CELLS));
    }
    let guard = 0;
    while (g.inSpawnPhase() && guard++ < 5000) this.tick([]);
    this.spawned = true;
    this.spawnTicks = g.ticks();
    this.startTicks = g.ticks();
    this.agents.forEach((p, i) => {
      this.alive[i] = p.hasSpawned() && p.isAlive();
      this.prevShare[i] = p.numTilesOwned() / this.landTiles;
      this.peakShare[i] = this.prevShare[i];
      if (!this.alive[i]) this.diedAt[i] = g.ticks();
    });
    return {
      obs: this.agents.map((_, i) => (this.alive[i] ? this.observe(i) : null)),
      reward: this.agents.map(() => 0),
      done: this.alive.map((a) => !a),
      episodeDone: this.alive.every((a) => !a),
      info: {},
    };
  }

  step(actions: (AgentAction | null)[]): StepResult {
    if (!this.spawned) throw new Error("call spawn() before step()");
    const g = this.game;
    const intents: StampedIntent[] = [];
    actions.forEach((a, i) => {
      if (a === null || !this.alive[i]) return;
      for (const intent of this.toIntents(i, a)) intents.push(intent);
    });
    this.tick(intents);
    for (let k = 1; k < this.opts.decisionInterval; k++) this.tick([]);

    const n = this.agents.length;
    const reward = new Array<number>(n).fill(0);
    const done = new Array<boolean>(n).fill(false);
    const winner = g.getWinner();
    const timeout = g.ticks() - this.startTicks >= this.opts.maxTicks;
    const gameOver = winner !== null || this.fatal !== undefined;
    const aliveCount = this.agents.filter((p) => p.isAlive()).length;
    const end = timeout || gameOver || aliveCount === 0;

    const ranks = this.ranks();
    this.agents.forEach((p, i) => {
      if (!this.alive[i]) {
        done[i] = true;
        return;
      }
      const share = p.numTilesOwned() / this.landTiles;
      // dense: land-share growth (x20 so a 5% gain ~ +1)
      reward[i] += 20 * (share - this.prevShare[i]);
      this.prevShare[i] = share;
      this.peakShare[i] = Math.max(this.peakShare[i], share);
      if (!p.isAlive()) {
        this.diedAt[i] = g.ticks();
        reward[i] -= 1; // eliminated
        this.alive[i] = false;
        done[i] = true;
      } else if (end) {
        // terminal: placement among agents + win bonus
        reward[i] += n > 1 ? 1 - (2 * ranks[i]) / (n - 1) : 0;
        if (winner === p) reward[i] += 3;
        done[i] = true;
      }
    });
    this.episodeDone = end;
    return {
      obs: this.agents.map((_, i) =>
        this.alive[i] && !end ? this.observe(i) : null,
      ),
      reward,
      done,
      episodeDone: end,
      info: {
        tick: g.ticks(),
        shares: this.agents.map((p) => p.numTilesOwned() / this.landTiles),
        ranks,
        peak: this.peakShare,
        survival: this.agents.map(
          (p, i) =>
            ((p.isAlive() ? g.ticks() : this.diedAt[i]) - this.startTicks) /
            this.opts.maxTicks,
        ),
        alive: this.agents.map((p) => p.isAlive()),
        winner: this.agents.findIndex((p) => p === winner),
        fatal: this.fatal,
      },
    };
  }

  // ── internals ──

  private tick(intents: StampedIntent[]) {
    this.runner.addTurn({ turnNumber: this.turn++, intents });
    if (!this.runner.executeNextTick() && this.fatal === undefined) {
      this.fatal = "executeNextTick returned false";
    }
  }

  /** rank 0 = most tiles, among agents only */
  private ranks(): number[] {
    // alive agents rank by territory, above the dead; the dead by how long they lasted
    const order = this.agents
      .map(
        (p, i) =>
          [p.isAlive() ? 1e12 + p.numTilesOwned() : this.diedAt[i], i] as const,
      )
      .sort((a, b) => b[0] - a[0]);
    const r = new Array<number>(this.agents.length).fill(0);
    order.forEach(([, i], pos) => (r[i] = pos));
    return r;
  }

  private spawnTile(cell: number): TileRef | null {
    const g = this.game;
    const W = g.width();
    const H = g.height();
    const cx = (((cell % SPAWN_CELLS) + 0.5) / SPAWN_CELLS) * W;
    const cy = ((Math.floor(cell / SPAWN_CELLS) + 0.5) / SPAWN_CELLS) * H;
    let best: TileRef | null = null;
    let bestD = Infinity;
    const step = Math.max(1, Math.floor(Math.min(W, H) / 120));
    for (let y = 0; y < H; y += step) {
      for (let x = 0; x < W; x += step) {
        const d = (x - cx) ** 2 + (y - cy) ** 2;
        if (d >= bestD) continue;
        const t = g.ref(x, y);
        if (!g.isLand(t) || g.hasOwner(t)) continue;
        best = t;
        bestD = d;
      }
    }
    return best;
  }

  /** Neighbouring players ordered by how much border they share with p. */
  private neighbours(p: Player): (Player | TerraNullius)[] {
    const g = this.game;
    const counts = new Map<number, { who: Player | TerraNullius; n: number }>();
    const out: TileRef[] = [0, 0, 0, 0];
    for (const t of p.borderTiles()) {
      const c = g.neighbors4(t, out);
      for (let k = 0; k < c; k++) {
        const nt = out[k];
        if (!g.isLand(nt)) continue;
        const o = g.owner(nt);
        if (o === p) continue;
        const key = o.isPlayer() ? (o as Player).smallID() : 0;
        const e = counts.get(key);
        if (e) e.n++;
        else counts.set(key, { who: o, n: 1 });
      }
    }
    const list = [...counts.values()].sort((a, b) => b.n - a.n);
    const players = list.filter((e) => e.who.isPlayer()).slice(0, N_SLOTS);
    return players.map((e) => e.who);
  }

  private hasWilderness(p: Player): boolean {
    const g = this.game;
    const out: TileRef[] = [0, 0, 0, 0];
    for (const t of p.borderTiles()) {
      const c = g.neighbors4(t, out);
      for (let k = 0; k < c; k++) {
        if (g.isLand(out[k]) && !g.hasOwner(out[k])) return true;
      }
    }
    return false;
  }

  private affordable(p: Player, type: UnitType): boolean {
    return p.gold() >= this.game.unitInfo(type).cost(this.game, p);
  }

  private observe(i: number): AgentObs {
    const g = this.game;
    const p = this.agents[i];
    const cfg = g.config();
    const maxT = Math.max(1, cfg.maxTroops(p));
    const nbrs = this.neighbours(p);
    const wild = this.hasWilderness(p);
    const allAlive = g.players();
    const myRank = this.ranksAll(p);
    const attacksOut = p.outgoingAttacks();
    const attacksIn = p.incomingAttacks();
    const outT = attacksOut.reduce((s, a) => s + a.troops(), 0);
    const inT = attacksIn.reduce((s, a) => s + a.troops(), 0);
    const builds = BUILDS.map((b) => this.affordable(p, b));
    const tOf = (t: UnitType) => p.unitCount(t);
    const log = (x: number) => Math.log1p(Math.max(0, x));

    const scalars = [
      (g.ticks() - this.startTicks) / this.opts.maxTicks,
      p.numTilesOwned() / this.landTiles,
      p.troops() / maxT,
      log(Number(p.gold())) / 15,
      log(p.troops()) / 15,
      allAlive.length / 100,
      myRank,
      log(tOf(UnitType.City)) / 3,
      log(tOf(UnitType.Port)) / 3,
      log(tOf(UnitType.DefensePost)) / 3,
      log(tOf(UnitType.Factory)) / 3,
      attacksOut.length / 4,
      log(outT) / 15,
      attacksIn.length / 4,
      log(inT) / 15,
      p.allies().length / 4,
      g.isSpawnImmunityActive() ? 1 : 0,
      p.borderTiles().size / Math.max(1, p.numTilesOwned()),
      builds[0] ? 1 : 0,
      builds[1] ? 1 : 0,
      builds[2] ? 1 : 0,
      builds[3] ? 1 : 0,
      wild ? 1 : 0,
      nbrs.length / N_SLOTS,
    ];

    const slots = new Array<number>(N_SLOTS * SLOT_FEATURES).fill(0);
    const targetMask = new Array<number>(N_TARGETS).fill(0);
    const allyMask = new Array<number>(N_TARGETS).fill(0);
    targetMask[0] = wild ? 1 : 0;
    nbrs.forEach((o, k) => {
      const q = o as Player;
      const f = k * SLOT_FEATURES;
      slots[f] = 1;
      slots[f + 1] = q.numTilesOwned() / this.landTiles;
      slots[f + 2] = Math.tanh(Math.log((q.troops() + 1) / (p.troops() + 1)));
      slots[f + 3] = p.isAlliedWith(q) ? 1 : 0;
      slots[f + 4] = q.type() === PlayerType.Human ? 1 : 0;
      slots[f + 5] = q.type() === PlayerType.Nation ? 1 : 0;
      slots[f + 6] = attacksOut.some((a) => a.target() === q) ? 1 : 0;
      slots[f + 7] = attacksIn.some((a) => a.attacker() === q) ? 1 : 0;
      targetMask[k + 1] = p.canAttackPlayer(q) ? 1 : 0;
      allyMask[k + 1] = p.canSendAllianceRequest(q) ? 1 : 0;
    });

    // coarse map image
    const grid = new Array<number>(GRID_CHANNELS * GRID * GRID).fill(0);
    const W = g.width();
    const H = g.height();
    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const t = g.ref(
          Math.min(W - 1, Math.floor(((gx + 0.5) / GRID) * W)),
          Math.min(H - 1, Math.floor(((gy + 0.5) / GRID) * H)),
        );
        const idx = gy * GRID + gx;
        if (!g.isLand(t)) {
          grid[4 * GRID * GRID + idx] = 1;
          continue;
        }
        const o = g.owner(t);
        if (!o.isPlayer()) grid[3 * GRID * GRID + idx] = 1;
        else if (o === p) grid[idx] = 1;
        else if (p.isAlliedWith(o as Player)) grid[GRID * GRID + idx] = 1;
        else grid[2 * GRID * GRID + idx] = 1;
      }
    }

    const canAttackAny = p.troops() > 50 && (wild || nbrs.length > 0);
    const kindMask = [
      1,
      canAttackAny ? 1 : 0,
      builds.some(Boolean) ? 1 : 0,
      allyMask.some(Boolean) ? 1 : 0,
    ];
    return {
      scalars,
      slots,
      grid,
      masks: {
        kind: kindMask,
        target: targetMask,
        build: builds.map((b) => (b ? 1 : 0)),
        ally_target: allyMask,
        spawn: new Array<number>(SPAWN_CELLS * SPAWN_CELLS).fill(1),
      },
    };
  }

  private ranksAll(p: Player): number {
    const mine = p.numTilesOwned();
    let better = 0;
    let total = 0;
    for (const q of this.game.players()) {
      total++;
      if (q.numTilesOwned() > mine) better++;
    }
    return total > 1 ? better / (total - 1) : 0;
  }

  private toIntents(i: number, a: AgentAction): StampedIntent[] {
    const p = this.agents[i];
    const clientID = this.clientIDs[i];
    const nbrs = this.neighbours(p);
    const at = (k: number): Player | null =>
      k >= 1 && k <= nbrs.length ? (nbrs[k - 1] as Player) : null;
    switch (KINDS[a.kind]) {
      case "attack": {
        const troops = Math.floor(p.troops() * RATIOS[a.ratio]);
        if (troops < 1) return [];
        if (a.target === 0) {
          return [
            {
              type: "attack",
              targetID: null,
              troops,
              clientID,
            } as StampedIntent,
          ];
        }
        const q = at(a.target);
        if (q === null || !p.canAttackPlayer(q)) return [];
        return [
          {
            type: "attack",
            targetID: q.id(),
            troops,
            clientID,
          } as StampedIntent,
        ];
      }
      case "build": {
        const type = BUILDS[a.build];
        if (!this.affordable(p, type)) return [];
        const tile = this.findSite(p, type, a.site);
        if (tile === null) return [];
        return [
          { type: "build_unit", unit: type, tile, clientID } as StampedIntent,
        ];
      }
      case "ally": {
        const q = at(a.target);
        if (q === null || !p.canSendAllianceRequest(q)) return [];
        return [
          {
            type: "allianceRequest",
            recipient: q.id(),
            clientID,
          } as StampedIntent,
        ];
      }
      default:
        return [];
    }
  }

  private findSite(p: Player, type: UnitType, site: number): TileRef | null {
    const pool: TileRef[] = [
      ...(SITES[site] === "border" ? p.borderTiles() : p.tiles()),
    ];
    if (pool.length === 0) return null;
    for (let tries = 0; tries < 40; tries++) {
      const t = pool[this.rng.nextInt(0, pool.length)];
      if (p.canBuild(type, t) !== false) return t;
    }
    return null;
  }

  closeInfo() {
    return { fatal: this.fatal };
  }
}

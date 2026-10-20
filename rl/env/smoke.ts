import {
  OpenFrontEnv,
  AgentAction,
  N_TARGETS,
  BUILDS,
  RATIOS,
  SITES,
} from "./OpenFrontEnv";

const map = process.argv[2] ?? "plains";
const env = new OpenFrontEnv({
  map,
  numAgents: 4,
  bots: 20,
  nations: "default",
  decisionInterval: 10,
  maxTicks: 3000,
  difficulty: "Medium" as any,
});
const r = (n: number) => Math.floor(Math.random() * n);
const t0 = performance.now();
const obs = await env.reset("smoke-1");
console.log("reset ms", (performance.now() - t0).toFixed(0));
let res = env.spawn(
  obs.map(() => ({
    kind: 0,
    target: 0,
    ratio: 0,
    build: 0,
    site: 0,
    spawn: r(256),
  })),
);
console.log(
  "spawned; alive",
  res.done.map((d) => !d),
);
let steps = 0;
const t1 = performance.now();
while (!res.episodeDone) {
  const acts: (AgentAction | null)[] = res.obs.map(
    (o) =>
      o && {
        kind: r(4),
        target: r(N_TARGETS),
        ratio: r(RATIOS.length),
        build: r(BUILDS.length),
        site: r(SITES.length),
        spawn: 0,
      },
  );
  res = env.step(acts);
  steps++;
  if (steps % 50 === 0)
    console.log(
      steps,
      JSON.stringify(res.info.shares),
      res.reward.map((x) => x.toFixed(2)).join(","),
    );
}
console.log(
  "done steps",
  steps,
  "ms/step",
  ((performance.now() - t1) / steps).toFixed(1),
  JSON.stringify(res.info),
);

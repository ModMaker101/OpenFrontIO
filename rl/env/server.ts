/**
 * JSON-lines bridge: one OpenFrontEnv per process, driven over stdin/stdout.
 *
 *   {"cmd":"reset","seed":"s","opts":{...}}  -> {"obs":[...]}
 *   {"cmd":"spawn","actions":[...]}           -> StepResult
 *   {"cmd":"step","actions":[...|null]}       -> StepResult
 *
 * stdout carries the protocol only; everything else goes to stderr.
 */
import readline from "node:readline";
import { DEFAULT_OPTIONS, EnvOptions, OpenFrontEnv } from "./OpenFrontEnv";

// The engine logs with console.log; keep stdout clean for the protocol.
const send = process.stdout.write.bind(process.stdout);
console.log = (...a: unknown[]) => console.error(...a);
console.warn = () => {};

let env: OpenFrontEnv | null = null;

async function handle(msg: any): Promise<unknown> {
  switch (msg.cmd) {
    case "reset": {
      env = new OpenFrontEnv({
        ...DEFAULT_OPTIONS,
        ...(msg.opts as Partial<EnvOptions>),
      });
      return { obs: await env.reset(String(msg.seed)) };
    }
    case "spawn":
      return env!.spawn(msg.actions);
    case "step":
      return env!.step(msg.actions);
    default:
      throw new Error(`unknown cmd ${msg.cmd}`);
  }
}

const rl = readline.createInterface({ input: process.stdin });
let chain = Promise.resolve();
rl.on("line", (line) => {
  chain = chain.then(async () => {
    try {
      send(JSON.stringify(await handle(JSON.parse(line))) + "\n");
    } catch (e) {
      send(JSON.stringify({ error: String((e as Error).stack ?? e) }) + "\n");
    }
  });
});
rl.on("close", () => process.exit(0));

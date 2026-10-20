"""Talks to rl/env/server.ts over stdin/stdout (JSON lines)."""
import json
import os
import subprocess
import numpy as np

import spec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class Obs:
    """One agent's observation, as numpy arrays."""

    __slots__ = ("scalars", "slots", "grid", "m_kind", "m_target", "m_build", "m_ally", "m_spawn")

    def __init__(self, d):
        self.scalars = np.asarray(d["scalars"], np.float32)
        self.slots = np.asarray(d["slots"], np.float32)
        self.grid = np.asarray(d["grid"], np.float32).reshape(spec.GRID_CHANNELS, spec.GRID, spec.GRID)
        m = d["masks"]
        self.m_kind = np.asarray(m["kind"], np.float32)
        self.m_target = np.asarray(m["target"], np.float32)
        self.m_build = np.asarray(m["build"], np.float32)
        self.m_ally = np.asarray(m["ally_target"], np.float32)
        self.m_spawn = np.asarray(m["spawn"], np.float32)


def _parse(res):
    if "error" in res:
        raise RuntimeError(res["error"])
    return res


class EnvProc:
    def __init__(self, opts):
        self.opts = opts
        tsx = os.path.join(ROOT, "node_modules", ".bin", "tsx")
        self.p = subprocess.Popen(
            [tsx, os.path.join(ROOT, "rl", "env", "server.ts")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=ROOT, text=True, bufsize=1,
        )

    # split send/recv so many envs can run in parallel
    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def recv(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("env process died")
        return _parse(json.loads(line))

    def send_reset(self, seed):
        self.send({"cmd": "reset", "seed": seed, "opts": self.opts})

    def send_spawn(self, actions):
        self.send({"cmd": "spawn", "actions": actions})

    def send_step(self, actions):
        self.send({"cmd": "step", "actions": actions})

    @staticmethod
    def obs_list(res):
        key = res["obs"]
        return [None if o is None else Obs(o) for o in key]

    def close(self):
        try:
            self.p.stdin.close()
            self.p.terminate()
        except Exception:
            pass

"""Client for the environment server: the same reset/step contract, but truth stays in another process."""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RemoteEnvError(RuntimeError):
    pass


class RemoteEnv:
    def __init__(self, mode="train", grader_log=None, schedule_path=None, python=None):
        args = [python or sys.executable, "-m", "auditgym.server", "--mode", mode]
        if grader_log:
            args += ["--grader-log", grader_log]
        if schedule_path:
            args += ["--schedule", schedule_path]
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                     env=env, cwd=ROOT)

    def _call(self, req):
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RemoteEnvError("server closed")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RemoteEnvError(resp.get("error", "error"))
        return resp

    def reset(self, seed=None, config=None, options=None):
        r = self._call({"op": "reset", "seed": seed, "config": config})
        return r["observation"], r["info"]

    def step(self, action):
        r = self._call({"op": "step", "action": action})
        return r["observation"], r["reward"], r["terminated"], r["truncated"], r["info"]

    def close(self):
        if self.proc.poll() is None:
            try:
                self._call({"op": "close"})
            except Exception:
                pass
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        for f in (self.proc.stdin, self.proc.stdout):
            if f and not f.closed:
                f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

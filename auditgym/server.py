"""Environment as a separate process (JSON lines over stdin/stdout).

The truth lives only in this process. Requests:
    {"op": "reset", "seed": 7, "config": "train"}      (train mode; eval mode ignores seed/config)
    {"op": "step", "action": {...}}
    {"op": "close"}
Responses: {"ok": true, ...} or {"ok": false, "error": "<code>"} (no tracebacks, no paths).

In eval mode the episodes come from a schedule file chosen by the evaluator, the
reward returned to the client is always 0.0 and grades go to --grader-log only.

    python -m auditgym.server --mode eval --schedule sched.json --grader-log grader.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys

from .env import AuditEnv


def serve(mode: str, grader_log=None, schedule=None, stdin=None, stdout=None):
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    env = AuditEnv(mode=mode, grader_log=grader_log)
    queue = list(schedule or [])

    def send(obj):
        stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
        stdout.flush()

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op") if isinstance(req, dict) else None
        except ValueError:
            send({"ok": False, "error": "bad_request"})
            continue
        try:
            if op == "reset":
                if mode == "eval":
                    if not queue:
                        send({"ok": False, "error": "schedule_exhausted"})
                        continue
                    spec = queue.pop(0)
                    if "episode_dir" in spec:
                        obs, info = env.reset(options={"episode_dir": spec["episode_dir"]})
                    else:
                        obs, info = env.reset(seed=spec["seed"], config=spec.get("config"))
                else:
                    cfg = req.get("config")
                    if isinstance(cfg, dict) and ("episode_dir" in cfg):
                        send({"ok": False, "error": "bad_request"})
                        continue
                    obs, info = env.reset(seed=req.get("seed"), config=cfg)
                send({"ok": True, "observation": obs, "info": info})
            elif op == "step":
                obs, r, term, trunc, info = env.step(req.get("action"))
                send({"ok": True, "observation": obs, "reward": r, "terminated": term, "truncated": trunc,
                      "info": info})
            elif op == "close":
                send({"ok": True})
                break
            else:
                send({"ok": False, "error": "unknown_op"})
        except Exception:  # never leak internals to the agent side
            send({"ok": False, "error": "request_failed"})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m auditgym.server")
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--grader-log")
    ap.add_argument("--schedule", help="JSON list of {seed, config} or {episode_dir} (eval mode)")
    a = ap.parse_args(argv)
    sched = None
    if a.schedule:
        with open(a.schedule, encoding="utf-8") as fh:
            sched = json.load(fh)
    serve(a.mode, a.grader_log, sched)


if __name__ == "__main__":
    main()

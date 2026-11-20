"""Evaluator-side runner: drives policies against the environment, records trajectories
and collects grades. Non-privileged policies run in a separate process (isolation='process')
and only ever see observations. The oracle runs in-process because it is evaluator-only.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time

from .env import AuditEnv
from .generator import SECRET_ENV
from .policies import POLICIES, make_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class InProcessAgent:
    def __init__(self, policy):
        self.policy = policy

    def start(self, obs, info, env=None):
        if self.policy.privileged:
            self.policy.set_truth(env.truth)  # evaluator-only shortcut, documented
        self.policy.reset(obs, info)
        return self.policy.act(obs, info), None

    def next(self, obs, reward, term, trunc, info, timeout=None):
        return self.policy.act(obs, info), None

    def close(self):
        pass


class ProcessAgent:
    def __init__(self, name, seed=0):
        self.name, self.seed = name, seed
        self._spawn()

    def _spawn(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.pop(SECRET_ENV, None)  # the agent process never sees the evaluator's seed secret
        self.proc = subprocess.Popen([sys.executable, "-m", "auditgym.agent_proc", "--policy", self.name,
                                      "--seed", str(self.seed)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1, env=env, cwd=ROOT)

    def _send(self, msg, timeout):
        self.proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        r, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not r:
            self._kill()
            self._spawn()
            return None, "agent_timeout"
        line = self.proc.stdout.readline()
        if not line:
            self._kill()
            self._spawn()
            return None, "agent_crashed"
        reply = json.loads(line)
        return reply.get("action"), reply.get("error")

    def start(self, obs, info, env=None, timeout=60.0):
        return self._send({"type": "reset", "observation": obs, "info": info}, timeout)

    def next(self, obs, reward, term, trunc, info, timeout=60.0):
        return self._send({"type": "step", "observation": obs, "reward": reward, "terminated": term,
                           "truncated": trunc, "info": info}, timeout)

    def _kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except Exception:
                pass

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"type": "close"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                pass
        self._kill()


def episode_kwargs(spec):
    if "episode_dir" in spec:
        return {"options": {"episode_dir": spec["episode_dir"]}}
    return {"seed": spec["seed"], "config": spec.get("config")}


def run_policy(policy_name, episodes, mode="eval", isolation="process", trajectories=None, grader_log=None,
               policy_seed=0, tag=None):
    """Run one policy on a list of episode specs. Returns a list of per-episode result dicts.

    ``trajectories``: open file handle (jsonl) or None. Lines contain exactly what the agent saw.
    """
    env = AuditEnv(mode=mode, grader_log=grader_log)
    privileged = POLICIES[policy_name].privileged
    if isolation == "process" and not privileged:
        agent = ProcessAgent(policy_name, policy_seed)
    else:
        agent = InProcessAgent(make_policy(policy_name, policy_seed))
    results = []
    try:
        for spec in episodes:
            t0 = time.perf_counter()
            obs, info = env.reset(**episode_kwargs(spec))
            eid = obs["episode_id"]

            def log(step, action, o, r, te, tr):
                if trajectories is not None:
                    trajectories.write(json.dumps({"episode_id": eid, "policy": policy_name, "mode": mode,
                                                   "step": step, "action": action, "observation": o, "reward": r,
                                                   "terminated": te, "truncated": tr},
                                                  separators=(",", ":"), ensure_ascii=False) + "\n")
            log(0, None, obs, 0.0, False, False)
            remaining = lambda: max(0.1, env.time_limit - (time.monotonic() - env._t0) + 0.5)
            if isinstance(agent, ProcessAgent):
                action, aerr = agent.start(obs, info, timeout=remaining())
            else:
                action, aerr = agent.start(obs, info, env)
            agent_errors = []
            while True:
                if aerr:
                    agent_errors.append(aerr)
                if action is None:
                    action = {"action": "agent_timeout"}
                obs, reward, term, trunc, info = env.step(action)
                log(env.steps_used, action, obs, reward, term, trunc)
                if term or trunc:
                    break
                if isinstance(agent, ProcessAgent):
                    action, aerr = agent.next(obs, reward, term, trunc, info, timeout=remaining())
                else:
                    action, aerr = agent.next(obs, reward, term, trunc, info)
            ev = env.evaluation()
            g = ev["grade"]
            results.append({
                "policy": policy_name, "tag": tag or spec.get("tag"), "episode_id": eid, "mode": mode,
                "config": spec.get("config"), "end": ev["end"], "valid": g["valid"],
                "tp": g["tp"], "fp": g["fp"], "fn": g["fn"], "precision": g["precision"], "recall": g["recall"],
                "f1": g["f1"], "return": g["reward"], "steps": ev["steps_used"],
                "inspection_steps": g["inspection_steps"], "env_time_s": ev["elapsed_s"],
                "wall_s": round(time.perf_counter() - t0, 4), "n_truth": len(env.truth["issues"]),
                "clean": env.truth["clean"],
                "interaction": any(t.get("scenario") for t in env.truth["issues"]),
                "fp_reasons": [d["reason"] for d in g["findings"] if d["status"] == "FP"],
                "fp_types": [d["type"] for d in g["findings"] if d["status"] == "FP"],
                "fn_types": [m["type"] for m in g["missed"]],
                "fn_variants": [next((t["variant"] for t in env.truth["issues"] if t["issue_id"] == m["issue_id"]), None)
                                for m in g["missed"]],
                "agent_errors": agent_errors,
            })
    finally:
        agent.close()
        env.close()
    return results


def summarize(results):
    import statistics as st
    out = {}
    for name in sorted({r["policy"] for r in results}, key=lambda n: list(POLICIES).index(n)):
        rs = [r for r in results if r["policy"] == name]
        mean = lambda k: round(st.mean(r[k] for r in rs), 4)
        out[name] = {"episodes": len(rs), "precision": mean("precision"), "recall": mean("recall"),
                     "f1": mean("f1"), "return": mean("return"), "steps": mean("steps"),
                     "wall_s": mean("wall_s"), "tp": sum(r["tp"] for r in rs), "fp": sum(r["fp"] for r in rs),
                     "fn": sum(r["fn"] for r in rs),
                     "return_clean": round(st.mean([r["return"] for r in rs if r["clean"]]), 4) if any(r["clean"] for r in rs) else None,
                     "return_interaction": round(st.mean([r["return"] for r in rs if r["interaction"]]), 4) if any(r["interaction"] for r in rs) else None,
                     "invalid": sum(1 for r in rs if not r["valid"]),
                     "truncated": sum(1 for r in rs if r["end"] != "submitted")}
    return out

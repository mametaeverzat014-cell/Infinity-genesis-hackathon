"""Agent side of the closed-evaluation protocol: a policy running in its own process.

It receives only what the environment shows a policy (observation, reward, flags, info)
as JSON lines on stdin and answers with {"action": {...}} on stdout. It never gets
seeds, configs, truth or grader output.
"""

from __future__ import annotations

import argparse
import json
import sys

from .policies import POLICIES, make_policy


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m auditgym.agent_proc")
    ap.add_argument("--policy", required=True, choices=sorted(k for k, v in POLICIES.items() if not v.privileged))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    policy = make_policy(a.policy, a.seed)
    out = sys.stdout
    for line in sys.stdin:
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "close":
            break
        try:
            if kind == "reset":
                policy.reset(msg["observation"], msg["info"])
            action = policy.act(msg["observation"], msg["info"])
            reply = {"action": action}
        except Exception as exc:  # report, the evaluator charges a step for an invalid action
            reply = {"action": {"action": "policy_error"}, "error": type(exc).__name__}
        out.write(json.dumps(reply, separators=(",", ":")) + "\n")
        out.flush()


if __name__ == "__main__":
    main()

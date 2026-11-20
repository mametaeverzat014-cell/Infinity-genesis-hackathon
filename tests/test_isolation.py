import json
import os
import subprocess
import sys
import tempfile
import unittest

from auditgym.client import RemoteEnv, RemoteEnvError
from auditgym.env import AuditEnv
from auditgym.runner import run_policy

from .helpers import TRUTH_TOKENS, erroneous, oracle_submission, play

def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed, cls.pub, cls.prv = erroneous("smoke")

    def test_no_truth_in_observations(self):
        for policy in ("expert", "random", "flag_all", "reconciler"):
            for mode in ("train", "eval"):
                env = AuditEnv(mode=mode)
                traj, r, info = play(env, policy, seed=self.seed, config="smoke")
                for a, obs, rew, te, tr in traj:
                    s = json.dumps(obs)
                    for tok in TRUTH_TOKENS:
                        self.assertNotIn(tok, s, (policy, tok))
                    if mode == "eval":
                        self.assertEqual(rew, 0.0)
                truth_ids = {"ISS-%02d" % (i + 1) for i in range(12)}
                self.assertFalse(any(t in json.dumps(info) for t in truth_ids))
                if mode == "eval":
                    self.assertNotIn("metrics", info)

    def test_reset_isolates_state(self):
        env = AuditEnv(mode="train")
        env.reset(seed=self.seed, config="smoke")
        for _ in range(7):
            env.step({"action": "list_records", "table": "ledger"})
        t1, r1, _ = play(env, "expert", seed=self.seed + 1, config="smoke")
        t2, r2, _ = play(AuditEnv(mode="train"), "expert", seed=self.seed + 1, config="smoke")
        self.assertEqual(json.dumps(t1), json.dumps(t2))
        self.assertEqual(r1, r2)

    def test_remote_env_train_mode(self):
        with RemoteEnv(mode="train") as env:
            obs, info = env.reset(seed=self.seed, config="smoke")
            self.assertEqual(obs["episode_id"], self.pub["manifest"]["episode_id"])
            o, r, te, tr, info = env.step({"action": "list_records", "table": "fx", "limit": 3})
            self.assertEqual(len(o["result"]["rows"]), 3)
            o, r, te, tr, info = env.step({"action": "submit", "findings": oracle_submission(self.pub, self.prv)})
            self.assertTrue(te)
            self.assertEqual(r, 0.998)  # one inspection step

    def test_remote_env_eval_mode_uses_evaluator_schedule(self):
        with tempfile.TemporaryDirectory() as d:
            sched = os.path.join(d, "schedule.json")
            log = os.path.join(d, "grader.jsonl")
            with open(sched, "w") as fh:
                json.dump([{"seed": self.seed, "config": "smoke"}], fh)
            with RemoteEnv(mode="eval", grader_log=log, schedule_path=sched) as env:
                # the agent's seed/config are ignored in eval mode
                obs, info = env.reset(seed=123, config="train")
                self.assertEqual(obs["episode_id"], self.pub["manifest"]["episode_id"])
                o, r, te, tr, info = env.step({"action": "submit", "findings": oracle_submission(self.pub, self.prv)})
                self.assertEqual(r, 0.0)
                self.assertNotIn("metrics", info)
                with self.assertRaises(RemoteEnvError):
                    env.reset(seed=1)
            self.assertEqual(json.loads(_read(log))["grade"]["reward"], 1.0)

    def test_server_rejects_bad_requests_without_details(self):
        p = subprocess.run([sys.executable, "-m", "auditgym.server", "--mode", "train"], cwd=ROOT,
                           input='not json\n{"op":"truth"}\n{"op":"step","action":{}}\n{"op":"reset","seed":1,"config":{"episode_dir":"/etc"}}\n',
                           capture_output=True, text=True, timeout=60)
        lines = [json.loads(x) for x in p.stdout.strip().splitlines()]
        self.assertEqual([x["error"] for x in lines], ["bad_request", "unknown_op", "request_failed", "bad_request"])
        self.assertNotIn("Traceback", p.stdout + p.stderr)

    def test_oracle_cannot_run_as_agent_process(self):
        p = subprocess.run([sys.executable, "-m", "auditgym.agent_proc", "--policy", "oracle"], cwd=ROOT,
                           capture_output=True, text=True, timeout=60)
        self.assertNotEqual(p.returncode, 0)

    def test_process_isolated_runner(self):
        res = run_policy("expert", [{"seed": self.seed, "config": "smoke"}], mode="eval", isolation="process")
        self.assertEqual(res[0]["f1"], 1.0)
        res = run_policy("random", [{"seed": self.seed, "config": "smoke"}], mode="eval", isolation="process",
                         policy_seed=3)
        res2 = run_policy("random", [{"seed": self.seed, "config": "smoke"}], mode="eval", isolation="inprocess",
                          policy_seed=3)
        self.assertEqual(res[0]["return"], res2[0]["return"])
        self.assertEqual(res[0]["steps"], res2[0]["steps"])


if __name__ == "__main__":
    unittest.main()


class SeedSecretTests(unittest.TestCase):
    def test_secret_changes_data_and_is_reproducible(self):
        from auditgym.generator import generate
        a, _ = generate(3, "smoke", secret="")
        b, _ = generate(3, "smoke", secret="k1")
        c, _ = generate(3, "smoke", secret="k1")
        self.assertNotEqual(a["manifest"]["episode_id"], b["manifest"]["episode_id"])
        self.assertNotEqual(a["ledger"], b["ledger"])
        self.assertEqual(b, c)

    def test_agent_process_does_not_inherit_secret(self):
        from auditgym.runner import ProcessAgent
        os.environ["AUDITGYM_SEED_SECRET"] = "top-secret"
        try:
            ag = ProcessAgent("silent")
            pid = ag.proc.pid
            with open("/proc/%d/environ" % pid, "rb") as fh:
                env = fh.read()
            ag.close()
        finally:
            del os.environ["AUDITGYM_SEED_SECRET"]
        self.assertNotIn(b"AUDITGYM_SEED_SECRET", env)

    def test_strict_evidence_requires_observed_ids(self):
        from .helpers import erroneous, oracle_submission
        seed, pub, prv = erroneous("smoke")
        e = AuditEnv(mode="train")
        e.reset(seed=seed, config={"preset": "smoke", "env": {"strict_evidence": True}})
        o, r, te, tr, info = e.step({"action": "submit", "findings": oracle_submission(pub, prv)})
        self.assertEqual(info["metrics"]["tp"], 0)  # evidence never shown to the agent does not count

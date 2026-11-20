import hashlib
import json
import os
import subprocess
import sys
import unittest

from auditgym.config import load_config
from auditgym.env import AuditEnv
from auditgym.episode import public_file_texts
from auditgym.generator import generate
from auditgym.runner import run_policy

from .helpers import play

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSFORM = {"id_salt": "x", "order_salt": "y", "shuffle_records": True}


def digest(seed, cfg):
    pub, prv = generate(seed, cfg)
    h = hashlib.sha256()
    for k, v in sorted(public_file_texts(pub).items()):
        h.update(k.encode() + v.encode())
    h.update(json.dumps(prv["truth"], sort_keys=True).encode())
    return h.hexdigest()


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_same_files(self):
        self.assertEqual(digest(5, "train"), digest(5, "train"))
        self.assertNotEqual(digest(5, "train"), digest(6, "train"))
        self.assertNotEqual(digest(5, "train"), digest(5, "interaction"))

    def test_cross_process_hash_seed_independent(self):
        code = ("import hashlib,json;from auditgym.generator import generate;from auditgym.episode import public_file_texts;"
                "p,v=generate(5,'train');h=hashlib.sha256();"
                "[h.update(k.encode()+t.encode()) for k,t in sorted(public_file_texts(p).items())];"
                "h.update(json.dumps(v['truth'],sort_keys=True).encode());print(h.hexdigest())")
        outs = set()
        for hs in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=ROOT)
            outs.add(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                                    text=True, check=True).stdout.strip())
        self.assertEqual(outs, {digest(5, "train")})

    def test_same_actions_same_transitions_and_rewards(self):
        for policy in ("expert", "random", "reconciler"):
            t1, r1, _ = play(AuditEnv(mode="train"), policy, seed=17, config="smoke")
            t2, r2, _ = play(AuditEnv(mode="train"), policy, seed=17, config="smoke")
            self.assertEqual(json.dumps(t1), json.dumps(t2))
            self.assertEqual(r1, r2)

    def test_id_and_order_changes_preserve_oracle_answer_and_reward(self):
        for seed in range(4):
            for preset in ("eval_mixed", "eval_interaction"):
                cfg = load_config(preset)
                cfg_t = load_config(dict(cfg, **TRANSFORM))
                pa, va = generate(seed, cfg)
                pb, vb = generate(seed, cfg_t)
                self.assertNotEqual({r["row_id"] for r in pa["ledger"]}, {r["row_id"] for r in pb["ledger"]})
                self.assertNotEqual([r["account"] for r in pa["ledger"]], [r["account"] for r in pb["ledger"]])
                key = lambda v: sorted((t["type"], t["variant"], t["expected_amount_minor"]) for t in v["truth"]["issues"])
                self.assertEqual(key(va), key(vb))
                for pol in ("oracle", "expert"):
                    ra = run_policy(pol, [{"seed": seed, "config": cfg}], mode="eval", isolation="inprocess")[0]
                    rb = run_policy(pol, [{"seed": seed, "config": cfg_t}], mode="eval", isolation="inprocess")[0]
                    self.assertEqual(ra["f1"], rb["f1"])
                    self.assertEqual(ra["tp"], rb["tp"])
                    if pol == "oracle":
                        self.assertEqual(ra["return"], rb["return"])

    def test_config_versions_recorded(self):
        pub, prv = generate(1, "train")
        self.assertEqual(pub["manifest"]["config_version"], load_config("train")["config_version"])
        self.assertEqual(len(pub["manifest"]["config_sha256"]), 64)
        self.assertEqual(prv["provenance"]["config"], load_config("train"))


if __name__ == "__main__":
    unittest.main()

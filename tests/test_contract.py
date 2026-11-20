import json
import os
import tempfile
import unittest

from auditgym.constants import ISSUE_TYPES, PAGE_LIMIT
from auditgym.env import AuditEnv, EnvError

from .helpers import clean, erroneous, oracle_submission


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed, cls.pub, cls.prv = erroneous("smoke")
        cls.cseed, cls.cpub, cls.cprv = clean("clean")

    def env(self, mode="train", **kw):
        e = AuditEnv(mode=mode, **kw)
        obs, info = e.reset(seed=self.seed, config="smoke")
        return e, obs, info

    def test_reset_observation(self):
        e, obs, info = self.env()
        json.dumps(obs), json.dumps(info)
        self.assertEqual(obs["steps_remaining"], 100)
        for k in ("task", "manifest", "policy", "tools", "episode_id"):
            self.assertIn(k, obs)
        m = obs["manifest"]
        for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
                  "counts", "rounding", "currency_exponents"):
            self.assertIn(k, m)
        self.assertEqual(set(m["issue_types"]), set(ISSUE_TYPES))
        self.assertEqual(info["episode_id"], m["episode_id"])
        self.assertNotIn("seed", json.dumps(obs))

    def test_step_signature_and_zero_exploration_reward(self):
        e, obs, _ = self.env()
        out = e.step({"action": "list_records", "table": "ledger"})
        self.assertEqual(len(out), 5)
        obs, r, te, tr, info = out
        json.dumps(obs), json.dumps(info)
        self.assertEqual((r, te, tr), (0.0, False, False))
        self.assertEqual(obs["steps_remaining"], 99)
        self.assertEqual(set(obs), {"kind", "step", "action", "result", "error", "steps_remaining"})

    def test_page_limit_and_stable_cursors(self):
        e, obs, _ = self.env()
        o = e.step({"action": "list_records", "table": "ledger", "limit": 500})[0]
        self.assertEqual(len(o["result"]["rows"]), PAGE_LIMIT)
        cur = o["result"]["next_cursor"]
        a = e.step({"action": "list_records", "table": "ledger", "cursor": cur})[0]["result"]
        b = e.step({"action": "list_records", "table": "ledger", "cursor": cur})[0]["result"]
        self.assertEqual(a, b)
        e2, _, _ = self.env()
        seen, cur = [], None
        while True:
            res = e2.step({"action": "list_records", "table": "ledger", "cursor": cur})[0]["result"]
            seen += [r["row_id"] for r in res["rows"]]
            cur = res["next_cursor"]
            if cur is None:
                break
        self.assertEqual(len(seen), self.pub["manifest"]["counts"]["ledger"])
        self.assertEqual(len(set(seen)), len(seen))

    def test_bounded_errors_consume_steps(self):
        e, obs, _ = self.env()
        cases = [
            ({"action": "get_record", "table": "ledger", "id": "LR-NOPE"}, "not_found"),
            ({"action": "get_record", "table": "nope", "id": "x"}, "unknown_table"),
            ({"action": "get_document", "doc_id": "X" * 60}, "not_found"),
            ({"action": "list_records", "table": "ledger", "cursor": "c1.ledger.999999"}, "bad_cursor"),
            ({"action": "list_records", "table": "ledger", "cursor": "c1.invoices.0"}, "bad_cursor"),
            ({"action": "list_records", "table": "ledger", "limit": 0}, "invalid_argument"),
            ({"action": "rm -rf /"}, "unknown_action"),
            ({"nothing": 1}, "invalid_action"),
            ("not an object", "invalid_action"),
            ({"action": "calculator", "expression": "import os"}, "calc_error"),
        ]
        for i, (act, code) in enumerate(cases):
            o, r, te, tr, _ = e.step(act)
            self.assertEqual(o["error"]["code"], code, act)
            self.assertLessEqual(len(o["error"]["message"]), 200)
            self.assertNotIn("XXXXXXXX", o["error"]["message"])
            self.assertNotIn("/", o["error"]["message"].replace("+ - * /", ""))
            self.assertEqual(o["steps_remaining"], 100 - (i + 1))
            self.assertEqual((r, te, tr), (0.0, False, False))

    def test_records_and_documents(self):
        e, obs, _ = self.env()
        inv = e.step({"action": "list_records", "table": "invoices", "limit": 1})[0]["result"]["rows"][0]
        self.assertNotIn("line_items", inv)
        d = e.step({"action": "get_document", "doc_id": inv["doc_id"]})[0]["result"]["document"]
        self.assertIn("line_items", d)
        self.assertEqual(d["invoice_id"], inv["invoice_id"])
        r1 = e.step({"action": "get_record", "table": "invoices", "id": inv["invoice_id"]})[0]["result"]["record"]
        r2 = e.step({"action": "get_record", "table": "invoices", "id": inv["doc_id"].lower()})[0]["result"]["record"]
        self.assertEqual(r1, r2)
        c = e.step({"action": "calculator", "expression": "2*(3+4)"})[0]
        self.assertEqual(c["result"]["value"], "14")

    def test_submit_terminates_and_resubmission_is_ignored(self):
        e, obs, _ = self.env()
        sub = oracle_submission(self.pub, self.prv)
        o, r, te, tr, info = e.step({"action": "submit", "findings": sub})
        self.assertTrue(te)
        self.assertFalse(tr)
        self.assertEqual(r, 1.0)
        self.assertEqual(info["metrics"]["f1"], 1.0)
        ev = e.evaluation()
        o2, r2, te2, tr2, _ = e.step({"action": "submit", "findings": sub})
        self.assertEqual(o2["error"]["code"], "episode_finished")
        self.assertEqual(r2, 0.0)
        self.assertTrue(te2)
        self.assertEqual(e.evaluation(), ev)

    def test_step_limit_truncates_with_empty_answer(self):
        e, obs, _ = self.env()
        for i in range(99):
            o, r, te, tr, _ = e.step({"action": "calculator", "expression": "1+1"})
            self.assertFalse(te or tr)
        o, r, te, tr, info = e.step({"action": "calculator", "expression": "1+1"})
        self.assertTrue(tr)
        self.assertFalse(te)
        self.assertEqual(o["steps_remaining"], 0)
        self.assertEqual(r, -1.0)  # empty answer on erroneous books
        ce = AuditEnv(mode="train")
        ce.reset(seed=self.cseed, config="clean")
        for _ in range(100):
            o, r, te, tr, info = ce.step({"action": "calculator", "expression": "1"})
        self.assertTrue(tr)
        self.assertAlmostEqual(r, 1 - 0.2)  # empty answer on clean books, 100 inspection steps

    def test_submit_on_last_step_is_allowed(self):
        e, obs, _ = self.env()
        for _ in range(99):
            e.step({"action": "calculator", "expression": "1"})
        o, r, te, tr, info = e.step({"action": "submit", "findings": oracle_submission(self.pub, self.prv)})
        self.assertTrue(te)
        self.assertFalse(tr)
        self.assertAlmostEqual(r, 1 - 0.002 * 99)

    def test_time_limit(self):
        e = AuditEnv(mode="train")
        e.reset(seed=self.seed, config={"preset": "smoke", "env": {"time_limit_s": 0.0}})
        o, r, te, tr, info = e.step({"action": "list_records", "table": "ledger"})
        self.assertTrue(tr)
        self.assertEqual(o["result"]["reason"], "time_limit")
        self.assertEqual(e.evaluation()["end"], "time_limit")

    def test_step_before_reset(self):
        with self.assertRaises(EnvError):
            AuditEnv().step({"action": "list_records", "table": "ledger"})

    def test_invalid_submission_terminates_with_minus_one_and_is_logged(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "grader.jsonl")
            e = AuditEnv(mode="train", grader_log=log)
            e.reset(seed=self.seed, config="smoke")
            o, r, te, tr, info = e.step({"action": "submit", "findings": {"episode_id": "wrong", "findings": []}})
            self.assertTrue(te)
            self.assertEqual(r, -1.0)
            rec = json.loads(_read(log).strip())
            self.assertFalse(rec["grade"]["valid"])
            self.assertTrue(rec["grade"]["error"])

    def test_eval_mode_hides_reward_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "grader.jsonl")
            e = AuditEnv(mode="eval", grader_log=log)
            e.reset(seed=self.seed, config="smoke")
            o, r, te, tr, info = e.step({"action": "submit", "findings": oracle_submission(self.pub, self.prv)})
            self.assertEqual(r, 0.0)
            self.assertNotIn("metrics", info)
            self.assertEqual(o["result"], {"status": "submitted"})
            self.assertEqual(json.loads(_read(log))["grade"]["reward"], 1.0)

    def test_submit_list_form(self):
        e, obs, _ = self.env()
        sub = oracle_submission(self.pub, self.prv)
        o, r, te, tr, info = e.step({"action": "submit", "episode_id": sub["episode_id"], "findings": sub["findings"]})
        self.assertEqual(r, 1.0)


if __name__ == "__main__":
    unittest.main()

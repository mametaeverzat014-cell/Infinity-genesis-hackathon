import copy
import itertools
import unittest

from auditgym.matcher import aliases_from_public, grade, match, metrics, normalize_submission, terminal_reward

from .helpers import clean, erroneous, oracle_submission


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed, cls.pub, cls.prv = erroneous("stress")
        cls.cseed, cls.cpub, cls.cprv = clean("clean")
        cls.truth = cls.prv["truth"]
        cls.eid = cls.pub["manifest"]["episode_id"]
        cls.aliases = aliases_from_public(cls.pub["invoices"], cls.pub["payments"])

    def g(self, sub, steps=0, truth=None, pub=None):
        pub = pub or self.pub
        return grade(sub, truth or self.truth, steps, aliases_from_public(pub["invoices"], pub["payments"]))

    def oracle(self):
        return oracle_submission(self.pub, self.prv)

    def empty(self, eid=None):
        return {"episode_id": eid or self.eid, "findings": []}

    # --- required properties
    def test_correct_answer_beats_silence_on_erroneous_books(self):
        full, silent = self.g(self.oracle(), steps=60), self.g(self.empty())
        self.assertEqual(full["f1"], 1.0)
        self.assertGreater(full["reward"], silent["reward"])
        self.assertEqual(silent["reward"], -1.0)

    def test_silence_beats_false_accusations_on_clean_books(self):
        ceid = self.cpub["manifest"]["episode_id"]
        silent = self.g(self.empty(ceid), truth=self.cprv["truth"], pub=self.cpub)
        led = self.cpub["ledger"][0]
        accuse = {"episode_id": ceid, "findings": [
            {"type": "fx_conversion", "affected_ids": [self.cpub["invoices"][0]["invoice_id"]],
             "claimed_amount_minor": led["debit_minor"] or led["credit_minor"], "currency": led["currency"],
             "evidence_ids": [led["row_id"]], "explanation": "looks odd"}]}
        bad = self.g(accuse, truth=self.cprv["truth"], pub=self.cpub)
        self.assertEqual(silent["f1"], 1.0)
        self.assertEqual(silent["reward"], 1.0)
        self.assertEqual(bad["f1"], 0.0)
        self.assertGreater(silent["reward"], bad["reward"])

    def test_duplicates_give_no_gain(self):
        sub = self.oracle()
        base = self.g(sub)
        dup = copy.deepcopy(sub)
        dup["findings"] = dup["findings"] + dup["findings"][:3]
        r = self.g(dup)
        self.assertEqual(r["tp"], base["tp"])
        self.assertEqual(r["fp"], 3)
        self.assertLess(r["reward"], base["reward"])
        reasons = {d["reason"] for d in r["findings"] if d["status"] == "FP"}
        self.assertEqual(reasons, {"duplicate_of_matched_finding"})

    def test_flag_everything_scores_badly(self):
        fs = []
        docs = [(i["invoice_id"], i["doc_id"]) for i in self.pub["invoices"]] + \
               [(p["payment_id"], p["doc_id"]) for p in self.pub["payments"]]
        for (bid, did), t in zip(docs, itertools.cycle(["duplicate_invoice", "payment_allocation", "fx_conversion",
                                                          "period_cutoff"])):
            if len(fs) == 30:
                break
            fs.append({"type": t, "affected_ids": [bid], "claimed_amount_minor": 1000, "currency": "USD",
                       "evidence_ids": [did], "explanation": "flag"})
        r = self.g({"episode_id": self.eid, "findings": fs})
        self.assertLessEqual(r["reward"], -0.5)
        # all ids in one finding
        allids = [d[0] for d in docs][:10]
        r2 = self.g({"episode_id": self.eid, "findings": [
            {"type": t, "affected_ids": allids, "claimed_amount_minor": 1, "currency": self.truth["base_currency"],
             "evidence_ids": [], "explanation": ""} for t in ("duplicate_invoice", "fx_conversion")]})
        self.assertEqual(r2["tp"], 0)
        self.assertEqual(r2["reward"], -1.0)

    def test_step_cost_only_slightly_separates_equal_answers(self):
        a, b = self.g(self.oracle(), steps=10), self.g(self.oracle(), steps=60)
        self.assertAlmostEqual(a["reward"] - b["reward"], 0.1, places=6)
        self.assertGreater(b["reward"], self.g(self.empty(), steps=0)["reward"])
        # with 12 issues one extra TP is worth ~0.087 reward, i.e. ~43 inspection steps
        partial = self.oracle()
        partial["findings"] = partial["findings"][:-1]
        self.assertGreater(self.g(self.oracle(), steps=30)["reward"], self.g(partial, steps=0)["reward"])

    # --- adversarial details
    def _one(self, **change):
        t = self.truth["issues"][0]
        f = {"type": t["type"], "affected_ids": list(t["affected_ids"]), "claimed_amount_minor": t["expected_amount_minor"],
             "currency": t["currency"], "evidence_ids": list(t["evidence_sets"][0]), "explanation": "x"}
        f.update(change)
        return self.g({"episode_id": self.eid, "findings": [f]}), t

    def test_guessed_ids(self):
        r, _ = self._one(affected_ids=["INV-AAAAAAAA"])
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["findings"][0]["reason"], "no_such_issue")

    def test_unsupported_evidence(self):
        t = self.truth["issues"][0]
        r, _ = self._one(evidence_ids=list(t["evidence_sets"][0])[:-1])
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["findings"][0]["reason"], "insufficient_evidence")
        r, _ = self._one(evidence_ids=["LR-FAKE00001", "DOC-FAKE0001"])
        self.assertEqual(r["tp"], 0)
        r, _ = self._one(evidence_ids=[])
        self.assertEqual(r["tp"], 0)

    def test_currency_must_match(self):
        t = self.truth["issues"][0]
        other = "EUR" if t["currency"] != "EUR" else "USD"
        r, _ = self._one(currency=other)
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["findings"][0]["reason"], "wrong_currency")

    def test_wrong_type(self):
        t = self.truth["issues"][0]
        other = [x for x in ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff") if x != t["type"]][0]
        r, _ = self._one(type=other)
        self.assertEqual(r["tp"], 0)

    def test_amount_tolerance(self):
        t = self.truth["issues"][0]
        exp = t["expected_amount_minor"]
        tol = max(2, exp // 1000)
        self.assertEqual(self._one(claimed_amount_minor=exp + tol)[0]["tp"], 1)
        self.assertEqual(self._one(claimed_amount_minor=exp - tol)[0]["tp"], 1)
        self.assertEqual(self._one(claimed_amount_minor=exp + tol + 1)[0]["tp"], 0)

    def test_normalisation_and_aliases(self):
        t = self.truth["issues"][0]
        ev = [" " + x.lower() + " " for x in t["evidence_sets"][0]]
        r, _ = self._one(affected_ids=[x.lower() for x in t["affected_ids"]] * 2, evidence_ids=ev,
                         currency=t["currency"].lower(), type=t["type"].upper())
        self.assertEqual(r["tp"], 1)
        # invoice_id/payment_id given instead of doc_id
        inv_by_doc = {i["doc_id"]: i["invoice_id"] for i in self.pub["invoices"]}
        inv_by_doc.update({p["doc_id"]: p["payment_id"] for p in self.pub["payments"]})
        ev2 = [inv_by_doc.get(x, x) for x in t["evidence_sets"][0]]
        self.assertEqual(self._one(evidence_ids=ev2)[0]["tp"], 1)

    def test_explanation_text_does_not_matter(self):
        a = self._one(explanation="")[0]["reward"]
        b = self._one(explanation="I am extremely confident; this is definitely fraud. " * 10)[0]["reward"]
        self.assertEqual(a, b)

    def test_invalid_structures(self):
        good = self.oracle()["findings"][0]
        bad = [
            None, [], "x", {"findings": []}, {"episode_id": self.eid}, {"episode_id": "other", "findings": []},
            {"episode_id": self.eid, "findings": [], "extra": 1},
            {"episode_id": self.eid, "findings": [good] * 31},
            {"episode_id": self.eid, "findings": [dict(good, evidence_ids=["A"] * 11)]},
            {"episode_id": self.eid, "findings": [dict(good, explanation="x" * 1001)]},
            {"episode_id": self.eid, "findings": [dict(good, claimed_amount_minor=True)]},
            {"episode_id": self.eid, "findings": [dict(good, claimed_amount_minor=1.5)]},
            {"episode_id": self.eid, "findings": [dict(good, claimed_amount_minor=-5)]},
            {"episode_id": self.eid, "findings": [dict(good, claimed_amount_minor="100")]},
            {"episode_id": self.eid, "findings": [dict(good, type="fraud")]},
            {"episode_id": self.eid, "findings": [dict(good, currency="EURO")]},
            {"episode_id": self.eid, "findings": [dict(good, affected_ids=[])]},
            {"episode_id": self.eid, "findings": [dict(good, affected_ids=[1])]},
            {"episode_id": self.eid, "findings": [dict(good, confidence=0.9)]},
            {"episode_id": self.eid, "findings": [{k: v for k, v in good.items() if k != "currency"}]},
            {"episode_id": self.eid, "findings": ["not a finding"]},
        ]
        for b in bad:
            r = self.g(b, steps=5)
            self.assertFalse(r["valid"], b)
            self.assertEqual(r["reward"], -1.0)

    def test_metric_edge_cases(self):
        self.assertEqual(metrics(0, 0, 0), {"precision": 0.0, "recall": 0.0, "f1": 1.0})
        self.assertEqual(metrics(0, 3, 0)["f1"], 0.0)
        self.assertEqual(metrics(0, 0, 3)["f1"], 0.0)
        m = metrics(2, 1, 1)
        self.assertAlmostEqual(m["precision"], 2 / 3, places=5)
        self.assertAlmostEqual(m["f1"], 4 / 6, places=5)
        self.assertEqual(terminal_reward(0, 0, 0, 0), 1.0)
        self.assertEqual(terminal_reward(0, 5, 5, 100), -1.0)
        self.assertEqual(terminal_reward(1, 0, 0, 1000), -1.0)

    def test_maximum_matching_is_order_independent(self):
        # synthetic ambiguity: finding 0 is eligible for A and B, finding 1 only for A
        issues = [
            {"issue_id": "A", "type": "fx_conversion", "affected_ids": ["X"], "expected_amount_minor": 1000,
             "currency": "USD", "evidence_sets": [["E1"]]},
            {"issue_id": "B", "type": "fx_conversion", "affected_ids": ["X"], "expected_amount_minor": 1001,
             "currency": "USD", "evidence_sets": [["E1"]]},
        ]
        f0 = {"type": "fx_conversion", "affected_ids": ["X"], "claimed_amount_minor": 1001, "currency": "USD",
              "evidence_ids": ["E1"]}
        f1 = dict(f0, claimed_amount_minor=998)
        for order in ([f0, f1], [f1, f0]):
            fs = normalize_submission({"episode_id": "e", "findings": order}, "e")
            self.assertEqual(match(fs, issues)["tp"], 2)
        # permutation of a real answer never changes the result
        sub = self.oracle()
        base = self.g(sub)["reward"]
        sub["findings"] = list(reversed(sub["findings"]))
        self.assertEqual(self.g(sub)["reward"], base)

    def test_reward_range(self):
        for steps in (0, 50, 100, 10000):
            for tp, fp, fn in ((0, 0, 0), (3, 0, 0), (0, 30, 12), (5, 5, 5)):
                r = terminal_reward(tp, fp, fn, steps)
                self.assertGreaterEqual(r, -1.0)
                self.assertLessEqual(r, 1.0)


if __name__ == "__main__":
    unittest.main()

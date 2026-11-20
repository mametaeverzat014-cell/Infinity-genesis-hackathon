import csv
import io
import os
import re
import tempfile
import unittest
from collections import Counter

from auditgym.config import ConfigError, load_config
from auditgym.constants import ISSUE_TYPES
from auditgym.episode import public_file_texts, read_private, read_public, write_episode
from auditgym.generator import generate
from auditgym.money import convert, parse_ts
from auditgym.verify import check_books, verify_episode

ISO = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$")


class DataTests(unittest.TestCase):
    def test_verifier_on_many_episodes(self):
        for cfg, n in (("train", 8), ("clean", 4), ("interaction", 6), ("stress", 6), ("eval_mixed", 4)):
            for s in range(n):
                pub, prv = generate(500 + s, cfg)
                rep = verify_episode(pub, prv)
                self.assertTrue(rep["ok"], (cfg, s, rep["checks"], rep["errors"][:3]))

    def test_sizes_and_limits(self):
        for s in range(6):
            pub, prv = generate(s, "train")
            c = pub["manifest"]["counts"]
            self.assertTrue(200 <= c["ledger"] <= 2000, c)
            self.assertTrue(20 <= c["documents"] <= 200, c)
            self.assertLessEqual(len(prv["truth"]["issues"]), 12)

    def test_each_type_independently_configurable(self):
        for t in ISSUE_TYPES:
            errs = {x: {"count": [0, 0]} for x in ISSUE_TYPES}
            errs[t] = {"count": [2, 2]}
            cfg = {"preset": "smoke", "errors": errs, "timezones": ["+05:00", "-05:00"],
                   "scenarios": {"partial_fx_payment_at_boundary": {"count": [0, 0]},
                                 "duplicate_with_misapplied_payment": {"count": [0, 0]}}}
            for s in range(3):
                pub, prv = generate(s, cfg)
                types = [i["type"] for i in prv["truth"]["issues"]]
                self.assertEqual(types, [t, t], (t, s, prv["provenance"]["skipped"]))
                self.assertTrue(verify_episode(pub, prv)["ok"])

    def test_variants_are_individually_selectable(self):
        variants = {"duplicate_invoice": ["duplicate_document", "double_posting"],
                    "payment_allocation": ["missing", "partial"],
                    "fx_conversion": ["utc_date", "wrong_date", "inverted", "digit_typo"],
                    "period_cutoff": ["tz_boundary", "backdated"]}
        for t, vs in variants.items():
            for v in vs:
                errs = {x: {"count": [0, 0]} for x in ISSUE_TYPES}
                errs[t] = {"count": [1, 1], "variants": [v]}
                cfg = {"preset": "smoke", "errors": errs, "timezones": ["+05:00", "-08:00"],
                       "foreign_customer_share": 0.7, "distractors": {"near_midnight_share": 0.3},
                       "scenarios": {"partial_fx_payment_at_boundary": {"count": [0, 0]},
                                     "duplicate_with_misapplied_payment": {"count": [0, 0]}}}
                got = Counter()
                for s in range(4):
                    pub, prv = generate(s, cfg)
                    for i in prv["truth"]["issues"]:
                        got[(i["type"], i["variant"])] += 1
                    self.assertTrue(verify_episode(pub, prv)["ok"])
                self.assertGreaterEqual(got[(t, v)], 3, (t, v, got))

    def test_clean_companies_have_distractors_and_no_issues(self):
        for s in range(4):
            pub, prv = generate(s, "clean")
            self.assertEqual(prv["truth"]["issues"], [])
            self.assertTrue(prv["truth"]["clean"])
            d = set(prv["provenance"]["distractors"])
            self.assertTrue({"recurring_same_amount", "split_payments", "overpayments", "advances"} <= d, d)
            self.assertEqual(check_books(pub, pub["ledger"], pub["invoices"], pub["payments"]), [])

    def test_interaction_scenarios_present(self):
        for s in range(5):
            pub, prv = generate(s, "interaction")
            sc = Counter(i["scenario"] for i in prv["truth"]["issues"] if i["scenario"])
            self.assertEqual(sc["partial_fx_payment_at_boundary"], 2, prv["provenance"]["skipped"])
            self.assertEqual(sc["duplicate_with_misapplied_payment"], 2)
            fx = [i for i in prv["truth"]["issues"] if i["scenario"] == "partial_fx_payment_at_boundary"]
            # both issues of the scenario concern the same payment
            pays = [set(x for x in i["affected_ids"] if x.startswith("PAY-")) for i in fx]
            self.assertEqual(pays[0], pays[1])

    def test_value_formats(self):
        pub, prv = generate(3, "train")
        m = pub["manifest"]
        self.assertTrue(ISO.match(m["period_start"]) and ISO.match(m["period_end"]))
        for r in pub["ledger"]:
            self.assertTrue(ISO.match(r["timestamp"]))
            self.assertIsInstance(r["debit_minor"], int)
            self.assertEqual(r["currency"], m["base_currency"])
        for d in pub["invoices"]:
            self.assertTrue(ISO.match(d["issued_at"]) and ISO.match(d["due_at"]))
            self.assertIsInstance(d["total_minor"], int)
        for f in pub["fx"]:
            self.assertIsInstance(f["rate"], str)
            self.assertRegex(f["rate"], r"^\d+(\.\d+)?$")
        self.assertTrue(any(p["invoice_refs"] is None for p in pub["payments"]))
        texts = public_file_texts(pub)
        self.assertIn(",null,", texts["ledger.csv"])
        self.assertIn('"invoice_refs":null', texts["payments.jsonl"])

    def test_variety(self):
        seen = Counter()
        for s in range(12):
            pub, _ = generate(s, "train")
            m = pub["manifest"]
            seen[("base", m["base_currency"])] += 1
            seen[("tz", m["timezone"])] += 1
        self.assertGreaterEqual(len([k for k in seen if k[0] == "base"]), 3)
        self.assertGreaterEqual(len([k for k in seen if k[0] == "tz"]), 4)

    def test_minor_units_and_conversion(self):
        from decimal import Decimal
        self.assertEqual(convert(1000, "EUR", "USD", Decimal("1.0842")), 1084)
        self.assertEqual(convert(1005, "EUR", "USD", Decimal("1")), 1005)
        self.assertEqual(convert(10000, "USD", "JPY", Decimal("151.515")), 15152)   # 100.00 USD -> 15152 JPY (half up)
        self.assertEqual(convert(1000, "JPY", "KWD", Decimal("0.00203")), 2030)     # exponent 0 -> 3
        self.assertEqual(convert(5, "USD", "USD", Decimal("1")), 5)

    def test_file_roundtrip(self):
        pub, prv = generate(11, "smoke")
        with tempfile.TemporaryDirectory() as d:
            write_episode(d, pub, prv)
            self.assertEqual(read_public(d), pub)
            self.assertEqual(read_private(d)["truth"], prv["truth"])
            self.assertEqual(sorted(os.listdir(os.path.join(d, "public"))),
                             sorted(["manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
                                     "fx.csv"]))

    def test_config_validation(self):
        for bad in ({"bogus": 1}, {"errors": {"made_up": {"count": [0, 1]}}}, {"max_issues": 13},
                    {"n_invoices": [5, 500]}, {"base_currencies": ["XXX"]}, {"env": {"max_steps": 101}}):
            with self.assertRaises(ConfigError):
                load_config(bad)
        with self.assertRaises(ConfigError):
            load_config("no_such_preset")

    def test_open_episode_files_regenerate(self):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "episodes")
        for split in ("train", "dev"):
            dirs = sorted(os.listdir(os.path.join(root, split)))
            self.assertEqual(len(dirs), 10)
        d = os.path.join(root, "dev", dirs[0])
        prv = read_private(d)
        pub2, _ = generate(int(prv["provenance"]["seed"]), prv["provenance"]["config"], secret="")
        for name, text in public_file_texts(pub2).items():
            with open(os.path.join(d, "public", name), encoding="utf-8") as fh:
                self.assertEqual(fh.read(), text, name)


if __name__ == "__main__":
    unittest.main()

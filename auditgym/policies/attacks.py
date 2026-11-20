"""Adversarial and random baselines: random explorer and the flag-everything attack."""

from __future__ import annotations

from ..constants import ISSUE_TYPES, TABLES
from ..generator import stream
from .base import Policy, TableReader


class RandomExplorerPolicy(Policy):
    """Random tool calls with a fixed seed, then a random (small) submission built from seen ids."""
    name = "random"

    def __init__(self, seed=0):
        self.seed = seed

    def reset(self, observation, info):
        super().reset(observation, info)
        self.rng = stream("random-explorer", self.seed, self.episode_id)
        self.budget = self.rng.randint(3, 30)
        self.seen = {"row": [], "doc": [], "inv": [], "pay": [], "fx": [], "amount": []}
        self.cursors = {}

    def _observe(self, obs):
        res = obs.get("result") or {}
        recs = list(res.get("rows") or [])
        if res.get("record"):
            recs.append(res["record"])
        if res.get("document"):
            recs.append(res["document"])
        if res.get("table") and "next_cursor" in res:
            self.cursors[res["table"]] = res["next_cursor"]
        for r in recs:
            for f, key in (("row_id", "row"), ("doc_id", "doc"), ("invoice_id", "inv"), ("payment_id", "pay"),
                           ("fx_id", "fx")):
                if r.get(f):
                    self.seen[key].append(r[f])
            for f in ("debit_minor", "credit_minor", "total_minor", "amount_minor"):
                if r.get(f):
                    self.seen["amount"].append(r[f])

    def act(self, observation, info):
        if observation.get("kind") != "reset":
            self._observe(observation)
        rng = self.rng
        if self.budget <= 0 or observation["steps_remaining"] <= 1:
            return self._submit()
        self.budget -= 1
        k = rng.randrange(5)
        if k <= 1:
            t = rng.choice(TABLES)
            return {"action": "list_records", "table": t, "cursor": self.cursors.get(t) if rng.random() < 0.5 else None,
                    "limit": rng.randint(1, 50)}
        if k == 2:
            t = rng.choice(TABLES)
            pool = self.seen["row"] + self.seen["inv"] + self.seen["pay"] + self.seen["fx"]
            rid = rng.choice(pool) if pool and rng.random() < 0.7 else "XX-%06d" % rng.randint(0, 999999)
            return {"action": "get_record", "table": t, "id": rid}
        if k == 3:
            did = rng.choice(self.seen["doc"]) if self.seen["doc"] and rng.random() < 0.8 else "DOC-NOPE"
            return {"action": "get_document", "doc_id": did}
        return {"action": "calculator", "expression": "%d * %d / %d" % (rng.randint(1, 9999), rng.randint(1, 99),
                                                                          rng.randint(1, 99))}

    def _submit(self):
        rng = self.rng
        fs = []
        ids = self.seen["inv"] + self.seen["pay"]
        for _ in range(rng.randint(0, 3) if ids else 0):
            ev_pool = self.seen["row"] + self.seen["doc"] + self.seen["fx"]
            fs.append({"type": rng.choice(ISSUE_TYPES),
                       "affected_ids": [rng.choice(ids) for _ in range(rng.randint(1, 2))],
                       "claimed_amount_minor": int(rng.choice(self.seen["amount"] or [0])),
                       "currency": self.manifest["base_currency"],
                       "evidence_ids": [rng.choice(ev_pool) for _ in range(rng.randint(0, 4))] if ev_pool else [],
                       "explanation": "random guess"})
        return self.submit(fs)


class FlagAllPolicy(Policy):
    """Attack: read cheaply, then flag as many documents x types as the 30-finding cap allows,
    each with the document's own ledger rows as 'evidence'."""
    name = "flag_all"

    def reset(self, observation, info):
        super().reset(observation, info)
        self.reader = TableReader(["invoices", "payments", "ledger"])

    def act(self, observation, info):
        if observation.get("kind") != "reset":
            self.reader.consume(observation)
        if self.reader.done() or observation["steps_remaining"] <= 1:
            return self.submit(self._flag())
        return self.reader.next_action()

    def _flag(self):
        rows = self.reader.rows
        by_doc = {}
        for r in rows["ledger"]:
            key = r["payment_id"] or r["invoice_id"]
            if key:
                by_doc.setdefault(key, []).append(r)
        docs = [(i["invoice_id"], i["doc_id"]) for i in rows["invoices"]] + \
               [(p["payment_id"], p["doc_id"]) for p in rows["payments"]]
        docs = [d for d in docs if d[0] in by_doc]
        out = []
        i = 0
        while len(out) < 30 and docs:
            bid, did = docs[i % len(docs)]
            t = ISSUE_TYPES[(i // len(docs) + i) % len(ISSUE_TYPES)]
            rs = by_doc[bid]
            amt = max(r["debit_minor"] or r["credit_minor"] for r in rs)
            fx = [f["fx_id"] for f in rows.get("fx", [])][:1]
            out.append({"type": t, "affected_ids": [bid], "claimed_amount_minor": int(amt),
                        "currency": self.manifest["base_currency"],
                        "evidence_ids": ([did] + [r["row_id"] for r in rs] + fx)[:10],
                        "explanation": "flagged"})
            i += 1
            if i > 10000:
                break
        return out


class OraclePolicy(Policy):
    """Evaluator-only: submits the truth with its first evidence set in one step.
    Measures the maximum attainable reward, not agent ability."""
    name = "oracle"
    privileged = True

    def __init__(self, truth=None):
        self.truth = truth

    def set_truth(self, truth):
        self.truth = truth

    def act(self, observation, info):
        fs = [{"type": t["type"], "affected_ids": list(t["affected_ids"]),
               "claimed_amount_minor": int(t["expected_amount_minor"]), "currency": t["currency"],
               "evidence_ids": list(t["evidence_sets"][0]), "explanation": "oracle"}
              for t in self.truth["issues"]]
        return self.submit(fs)


class SilentPolicy(Policy):
    """Submits an empty answer immediately (used in reward tests)."""
    name = "silent"

    def act(self, observation, info):
        return self.submit([])



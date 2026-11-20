"""Simple reconciliation baseline (deliberately naive, like a quick spreadsheet check).

* period and FX rate date taken from the UTC calendar date (ignores the company timezone);
* duplicates = same entity, currency and total (ignores PO / line items);
* allocations compared at the payment-date rate (ignores historical invoice rates), 1% threshold;
* no source documents are opened.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timezone
from decimal import Decimal

from ..constants import ACC_AR, ACC_CASH, ACC_UNAPPLIED
from ..money import RateBook, convert, parse_ts
from .base import Policy, TableReader


class ReconcilerPolicy(Policy):
    name = "reconciler"

    def reset(self, observation, info):
        super().reset(observation, info)
        self.reader = TableReader(["ledger", "invoices", "payments", "fx"])

    def act(self, observation, info):
        if observation.get("kind") != "reset":
            self.reader.consume(observation)
        if observation["steps_remaining"] <= 1 or self.reader.done():
            return self.submit(self._findings()[:30])
        return self.reader.next_action()

    def _findings(self):
        rows = self.reader.rows
        base = self.manifest["base_currency"]
        p0 = date.fromisoformat(self.manifest["period_start"][:10])
        p1 = date.fromisoformat(self.manifest["period_end"][:10])
        rb = RateBook([(date.fromisoformat(r["date"]), r["source_currency"], r["target_currency"],
                        Decimal(r["rate"]), r["fx_id"]) for r in rows["fx"]])

        def utc_date(ts):
            return parse_ts(ts).astimezone(timezone.utc).date()

        def in_period(ts):
            return p0 <= utc_date(ts) < p1

        def rate(cur, ts):
            hit = rb.lookup(cur, base, utc_date(ts))
            return (hit[0], hit[1]) if hit else (None, None)

        by_inv, by_pay = defaultdict(list), defaultdict(list)
        for r in rows["ledger"]:
            if r["payment_id"]:
                by_pay[r["payment_id"]].append(r)
            elif r["invoice_id"]:
                by_inv[r["invoice_id"]].append(r)
        out = []

        def add(t, aff, amt, ev):
            out.append({"type": t, "affected_ids": aff, "claimed_amount_minor": int(abs(amt)), "currency": base,
                        "evidence_ids": [e for e in ev if e][:10], "explanation": "reconciliation difference"})

        invs = {i["invoice_id"]: i for i in rows["invoices"]}
        seen = defaultdict(list)
        for iid, inv in sorted(invs.items(), key=lambda kv: (kv[1]["issued_at"], kv[0])):
            ar = [r for r in by_inv[iid] if r["account"] == ACC_AR and r["debit_minor"] > 0]
            if not ar:
                continue
            if not in_period(inv["issued_at"]):
                add("period_cutoff", [iid], ar[0]["debit_minor"], [inv["doc_id"], ar[0]["row_id"]])
                continue
            if len(ar) > 1:
                add("duplicate_invoice", [iid], ar[1]["debit_minor"], [inv["doc_id"], ar[0]["row_id"], ar[1]["row_id"]])
                continue
            key = (inv["entity_id"], inv["currency"], inv["total_minor"])
            if seen[key]:
                o = seen[key][0]
                add("duplicate_invoice", [o["invoice_id"], iid], ar[0]["debit_minor"], [o["doc_id"], inv["doc_id"], ar[0]["row_id"]])
            seen[key].append(inv)
            rt, fx_id = rate(inv["currency"], inv["issued_at"])
            if rt is not None:
                exp = convert(inv["total_minor"], inv["currency"], base, rt)
                if exp != ar[0]["debit_minor"]:
                    add("fx_conversion", [iid], exp - ar[0]["debit_minor"], [inv["doc_id"], ar[0]["row_id"], fx_id])
        for p in rows["payments"]:
            rs = by_pay[p["payment_id"]]
            cash = [r for r in rs if r["account"] == ACC_CASH and r["debit_minor"] > 0]
            if not cash:
                continue
            if not in_period(p["timestamp"]):
                add("period_cutoff", [p["payment_id"]], cash[0]["debit_minor"], [p["doc_id"], cash[0]["row_id"]])
                continue
            rt, fx_id = rate(p["currency"], p["timestamp"])
            if rt is None:
                continue
            exp = convert(p["amount_minor"], p["currency"], base, rt)
            if exp != cash[0]["debit_minor"]:
                add("fx_conversion", [p["payment_id"]], exp - cash[0]["debit_minor"], [p["doc_id"], cash[0]["row_id"], fx_id])
            for ref in p.get("invoice_refs") or []:
                inv = invs.get(ref["invoice_id"])
                want = convert(ref["applied_minor"], p["currency"], base, rt)
                got = sum(r["credit_minor"] for r in rs if r["account"] == ACC_AR and r["invoice_id"] == ref["invoice_id"])
                if abs(want - got) * 100 > max(1, want):
                    un = [r for r in rs if r["account"] == ACC_UNAPPLIED]
                    row = (un or cash)[0]
                    add("payment_allocation", [p["payment_id"], ref["invoice_id"]], want - got,
                        [p["doc_id"], inv["doc_id"] if inv else None, row["row_id"]])
        return out

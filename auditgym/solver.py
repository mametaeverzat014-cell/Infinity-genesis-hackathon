"""Reference auditor that uses ONLY public records and the published policy.

It is used (a) by the scripted ``expert`` policy, which reads the records through
the tool interface, and (b) by ``verify`` to prove that every declared issue is
derivable from public data and that no undeclared discrepancy exists.
It never touches truth.json, provenance.json or the seed.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal

from .constants import ACC_AR, ACC_CASH, ACC_REVENUE, ACC_UNAPPLIED
from .money import RateBook, convert, parse_offset, parse_ts


class PublicBooks:
    def __init__(self, manifest, ledger, invoices, payments, fx):
        self.manifest = manifest
        self.base = manifest["base_currency"]
        self.tz = parse_offset(manifest["timezone"])
        self.start = parse_ts(manifest["period_start"])
        self.end = parse_ts(manifest["period_end"])
        self.ledger = ledger
        self.invoices = {i["invoice_id"]: i for i in invoices}
        self.payments = {p["payment_id"]: p for p in payments}
        self.fx = fx
        self.rates = RateBook([(date.fromisoformat(r["date"]), r["source_currency"], r["target_currency"],
                                Decimal(r["rate"]), r["fx_id"]) for r in fx])
        self.by_inv = defaultdict(list)
        self.by_pay = defaultdict(list)
        for r in ledger:
            if r["payment_id"] is not None:
                self.by_pay[r["payment_id"]].append(r)
            elif r["invoice_id"] is not None:
                self.by_inv[r["invoice_id"]].append(r)

    def in_period(self, ts_text: str) -> bool:
        return self.start <= parse_ts(ts_text) < self.end

    def rate_at(self, currency: str, ts_text: str):
        d = parse_ts(ts_text).astimezone(self.tz).date()
        hit = self.rates.lookup(currency, self.base, d)
        if hit is None:
            return None, None
        return hit[0], hit[1]


def _line_signature(doc):
    return tuple(sorted((l["sku"], l["quantity"], l["unit_price_minor"], l["service_period"])
                        for l in doc.get("line_items") or []))


def duplicate_candidate_groups(books: PublicBooks):
    """Groups of booked in-period invoices sharing (entity, currency, total): need document detail."""
    groups = defaultdict(list)
    for iid, inv in books.invoices.items():
        if not books.in_period(inv["issued_at"]):
            continue
        ar = [r for r in books.by_inv[iid] if r["account"] == ACC_AR and r["debit_minor"] > 0]
        if len(ar) == 1:
            groups[(inv["entity_id"], inv["currency"], inv["total_minor"])].append(iid)
    return [sorted(v) for k, v in sorted(groups.items()) if len(v) > 1]


def audit(books: PublicBooks, documents: dict | None = None) -> list:
    """Return findings (findings.json 'findings' list). ``documents``: doc_id -> full document."""
    documents = documents or {}
    base = books.base
    out = []

    def finding(t, affected, amount, evidence, why):
        out.append({"type": t, "affected_ids": sorted(affected), "claimed_amount_minor": int(amount),
                    "currency": base, "evidence_ids": evidence, "explanation": why})

    # ---- invoices: cut-off, double posting, FX
    for iid in sorted(books.invoices):
        inv = books.invoices[iid]
        rows = books.by_inv[iid]
        ar = [r for r in rows if r["account"] == ACC_AR and r["debit_minor"] > 0]
        if not books.in_period(inv["issued_at"]):
            if ar:
                finding("period_cutoff", [iid], ar[0]["debit_minor"], [inv["doc_id"], ar[0]["row_id"]],
                        "Invoice dated outside the period (company timezone) is posted in this period.")
            continue
        if len(ar) >= 2:
            ar = sorted(ar, key=lambda r: (r["timestamp"], r["row_id"]))
            finding("duplicate_invoice", [iid], ar[1]["debit_minor"],
                    [inv["doc_id"], ar[0]["row_id"], ar[1]["row_id"]], "Invoice posted to AR twice.")
            continue
        if len(ar) == 1:
            rate, fx_id = books.rate_at(inv["currency"], inv["issued_at"])
            if rate is None:
                continue
            expected = convert(inv["total_minor"], inv["currency"], base, rate)
            booked = ar[0]["debit_minor"]
            if booked != expected:
                ev = [inv["doc_id"], ar[0]["row_id"]] + ([fx_id] if fx_id else [])
                finding("fx_conversion", [iid], abs(booked - expected), ev,
                        "AR debit %d differs from policy conversion %d." % (booked, expected))

    # ---- duplicates across documents (needs line items / po_ref)
    for group in duplicate_candidate_groups(books):
        docs = []
        for iid in group:
            inv = books.invoices[iid]
            d = documents.get(inv["doc_id"]) or (inv if "line_items" in inv else None)
            if d is not None:
                docs.append(d)
        buckets = defaultdict(list)
        for d in docs:
            buckets[(d.get("po_ref"), _line_signature(d))].append(d)
        for key, ds in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            if len(ds) < 2 or key[0] is None:
                continue
            ds = sorted(ds, key=lambda d: (parse_ts(d["issued_at"]), d["invoice_id"]))
            orig = ds[0]
            for dup in ds[1:]:
                ar = [r for r in books.by_inv[dup["invoice_id"]] if r["account"] == ACC_AR and r["debit_minor"] > 0]
                finding("duplicate_invoice", [orig["invoice_id"], dup["invoice_id"]], ar[0]["debit_minor"],
                        [orig["doc_id"], dup["doc_id"], ar[0]["row_id"]],
                        "Same entity, PO and line items invoiced twice.")

    # ---- payments: cut-off, FX, allocation
    for pid in sorted(books.payments):
        p = books.payments[pid]
        rows = books.by_pay[pid]
        if not books.in_period(p["timestamp"]):
            cash = [r for r in rows if r["account"] == ACC_CASH and r["debit_minor"] > 0]
            if cash:
                finding("period_cutoff", [pid], cash[0]["debit_minor"], [p["doc_id"], cash[0]["row_id"]],
                        "Payment dated outside the period (company timezone) is posted in this period.")
            continue
        if p.get("kind", "receipt") != "receipt" or not rows:
            continue
        cash = [r for r in rows if r["account"] == ACC_CASH and r["debit_minor"] > 0]
        rate, fx_id = books.rate_at(p["currency"], p["timestamp"])
        if cash and rate is not None:
            expected = convert(p["amount_minor"], p["currency"], base, rate)
            if cash[0]["debit_minor"] != expected:
                ev = [p["doc_id"], cash[0]["row_id"]] + ([fx_id] if fx_id else [])
                finding("fx_conversion", [pid], abs(cash[0]["debit_minor"] - expected), ev,
                        "Cash debit differs from policy conversion at the payment date.")
        for ref in p.get("invoice_refs") or []:
            iid = ref["invoice_id"]
            inv = books.invoices.get(iid)
            if inv is None:
                continue
            hr, hfx = books.rate_at(inv["currency"], inv["issued_at"])
            if hr is None:
                continue
            expected = convert(ref["applied_minor"], p["currency"], base, hr)
            ar_rows = [r for r in rows if r["account"] == ACC_AR and r["invoice_id"] == iid]
            booked = sum(r["credit_minor"] - r["debit_minor"] for r in ar_rows)
            if booked != expected:
                if ar_rows:
                    row = ar_rows[0]
                else:
                    un = [r for r in rows if r["account"] == ACC_UNAPPLIED]
                    row = un[0] if un else (cash[0] if cash else rows[0])
                ev = [p["doc_id"], inv["doc_id"], row["row_id"]] + ([hfx] if hfx else [])
                finding("payment_allocation", [pid, iid], abs(expected - booked), ev,
                        "Remittance applies %d to the invoice but AR was credited %d." % (expected, booked))
    return out

"""Independent data and truth verification (evaluator-side).

For one episode it checks:
  1. pre-injection books: revert every logged change (provenance) and check the accounting
     identities of the clean ledger against the documents and policy;
  2. post-injection: every declared issue is re-derived from the records of each of its
     evidence sets alone (plus the published policy; "absence" facts such as "no AR credit
     exists for this pair" are checked on the full table);
  3. completeness: the public-only reference auditor finds exactly the declared issues
     (no undeclared discrepancy, no unsolvable issue);
  4. schema/limits: ids exist, evidence sets <= 10 ids, amounts positive, counts in manifest.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from datetime import date
from decimal import Decimal

from .constants import ACC_AR, ACC_CASH, ACC_FX, ACC_REVENUE, ACC_UNAPPLIED, MAX_EVIDENCE
from .matcher import aliases_from_public, grade
from .money import RateBook, convert, parse_offset, parse_ts
from .solver import PublicBooks, audit


class Ctx:
    def __init__(self, public, ledger=None, invoices=None, payments=None):
        m = public["manifest"]
        self.m = m
        self.base = m["base_currency"]
        self.tz = parse_offset(m["timezone"])
        self.start = parse_ts(m["period_start"])
        self.end = parse_ts(m["period_end"])
        self.ledger = public["ledger"] if ledger is None else ledger
        self.invoices = {d["invoice_id"]: d for d in (public["invoices"] if invoices is None else invoices)}
        self.payments = {d["payment_id"]: d for d in (public["payments"] if payments is None else payments)}
        self.docs = {d["doc_id"]: d for d in list(self.invoices.values()) + list(self.payments.values())}
        self.rows = {r["row_id"]: r for r in self.ledger}
        self.fx = {r["fx_id"]: r for r in public["fx"]}
        self.rb = RateBook([(date.fromisoformat(r["date"]), r["source_currency"], r["target_currency"],
                             Decimal(r["rate"]), r["fx_id"]) for r in public["fx"]])

    def inp(self, ts):
        return self.start <= parse_ts(ts) < self.end

    def rate(self, cur, ts):
        hit = self.rb.lookup(cur, self.base, parse_ts(ts).astimezone(self.tz).date())
        return (hit[0], hit[1]) if hit else (None, None)

    def conv(self, amt, cur, rate):
        return convert(amt, cur, self.base, rate)


# ------------------------------------------------------------------ revert
def revert(public, provenance):
    ledger = {r["row_id"]: dict(r) for r in public["ledger"]}
    removed_docs = set()
    for iss in reversed(provenance["issues"]):
        for op in reversed(iss["changes"]):
            o = op["op"]
            if o == "insert_row":
                ledger.pop(op["row"]["row_id"], None)
            elif o == "delete_row":
                ledger[op["row"]["row_id"]] = dict(op["row"])
            elif o == "update_row":
                ledger[op["before"]["row_id"]] = dict(op["before"])
            elif o == "insert_document":
                removed_docs.add(op["doc_id"])
    invoices = [d for d in public["invoices"] if d["doc_id"] not in removed_docs]
    payments = [d for d in public["payments"] if d["doc_id"] not in removed_docs]
    return list(ledger.values()), invoices, payments


def check_books(public, ledger, invoices, payments) -> list:
    """Accounting identities that hold for correct books. Returns a list of violations."""
    c = Ctx(public, ledger, invoices, payments)
    v = []
    journals = defaultdict(list)
    for r in c.ledger:
        journals[r["journal_id"]].append(r)
        if r["currency"] != c.base:
            v.append("row %s not in base currency" % r["row_id"])
        if (r["debit_minor"] > 0) == (r["credit_minor"] > 0) or r["debit_minor"] < 0 or r["credit_minor"] < 0:
            v.append("row %s must have exactly one positive side" % r["row_id"])
        if not c.inp(r["timestamp"]):
            v.append("row %s outside period" % r["row_id"])
        for f, table in (("invoice_id", c.invoices), ("payment_id", c.payments)):
            if r[f] is not None and r[f] not in table:
                v.append("row %s references unknown %s" % (r["row_id"], f))
    for j, rs in journals.items():
        if sum(r["debit_minor"] for r in rs) != sum(r["credit_minor"] for r in rs):
            v.append("journal %s does not balance" % j)
    by_inv, by_pay = defaultdict(list), defaultdict(list)
    for r in c.ledger:
        if r["payment_id"] is not None:
            by_pay[r["payment_id"]].append(r)
        elif r["invoice_id"] is not None:
            by_inv[r["invoice_id"]].append(r)
    for iid, d in c.invoices.items():
        lines = d["line_items"]
        if any(l["amount_minor"] != l["quantity"] * l["unit_price_minor"] for l in lines):
            v.append("invoice %s line amount mismatch" % iid)
        if sum(l["amount_minor"] for l in lines) != d["total_minor"]:
            v.append("invoice %s total mismatch" % iid)
        rs = by_inv[iid]
        if not c.inp(d["issued_at"]):
            if rs:
                v.append("out-of-period invoice %s posted" % iid)
            continue
        rate, _ = c.rate(d["currency"], d["issued_at"])
        want = c.conv(d["total_minor"], d["currency"], rate)
        ar = [r for r in rs if r["account"] == ACC_AR]
        rev = [r for r in rs if r["account"] == ACC_REVENUE]
        if len(rs) != 2 or len(ar) != 1 or len(rev) != 1 or ar[0]["debit_minor"] != want or rev[0]["credit_minor"] != want:
            v.append("invoice %s posting mismatch" % iid)
        if any(parse_ts(r["timestamp"]) != parse_ts(d["issued_at"]) for r in rs):
            v.append("invoice %s posting timestamp differs from document" % iid)
    for pid, p in c.payments.items():
        rs = by_pay[pid]
        if p["invoice_refs"] is not None:
            if sum(x["applied_minor"] for x in p["invoice_refs"]) > p["amount_minor"]:
                v.append("payment %s over-allocated" % pid)
            for x in p["invoice_refs"]:
                inv = c.invoices.get(x["invoice_id"])
                if inv is None or inv["currency"] != p["currency"] or inv["entity_id"] != p["entity_id"]:
                    v.append("payment %s references inconsistent invoice" % pid)
        if not c.inp(p["timestamp"]):
            if rs:
                v.append("out-of-period payment %s posted" % pid)
            continue
        if any(parse_ts(r["timestamp"]) != parse_ts(p["timestamp"]) for r in rs):
            v.append("payment %s posting timestamp differs from document" % pid)
        rate, _ = c.rate(p["currency"], p["timestamp"])
        amt = c.conv(p["amount_minor"], p["currency"], rate)
        if p["kind"] == "refund":
            ok = (len(rs) == 2 and any(r["account"] == ACC_UNAPPLIED and r["debit_minor"] == amt for r in rs)
                  and any(r["account"] == ACC_CASH and r["credit_minor"] == amt for r in rs))
            if not ok:
                v.append("refund %s posting mismatch" % pid)
            continue
        cash = [r for r in rs if r["account"] == ACC_CASH]
        if len(cash) != 1 or cash[0]["debit_minor"] != amt:
            v.append("payment %s cash mismatch" % pid)
        applied = 0
        for x in p["invoice_refs"] or []:
            inv = c.invoices[x["invoice_id"]]
            hr, _ = c.rate(inv["currency"], inv["issued_at"])
            want = c.conv(x["applied_minor"], p["currency"], hr)
            got = [r for r in rs if r["account"] == ACC_AR and r["invoice_id"] == x["invoice_id"]]
            if len(got) != 1 or got[0]["credit_minor"] != want:
                v.append("payment %s allocation to %s mismatch" % (pid, x["invoice_id"]))
            applied += x["applied_minor"]
        refs = {x["invoice_id"] for x in p["invoice_refs"] or []}
        if any(r["account"] == ACC_AR and r["invoice_id"] not in refs for r in rs):
            v.append("payment %s credits an unreferenced invoice" % pid)
        rem = p["amount_minor"] - applied
        un = [r for r in rs if r["account"] == ACC_UNAPPLIED]
        want_un = c.conv(rem, p["currency"], rate) if rem > 0 else 0
        if (rem > 0) != bool(un) or (un and un[0]["credit_minor"] != want_un):
            v.append("payment %s unapplied mismatch" % pid)
        if any(r["account"] not in (ACC_CASH, ACC_AR, ACC_UNAPPLIED, ACC_FX) for r in rs):
            v.append("payment %s unexpected account" % pid)
    return v


# ------------------------------------------------------------- declared issues
def _primary(c, row, doc):
    if "invoice_id" in doc and "payment_id" not in doc:
        ok = row["invoice_id"] == doc["invoice_id"] and row["payment_id"] is None and (
            (row["account"] == ACC_AR and row["debit_minor"] > 0) or (row["account"] == ACC_REVENUE and row["credit_minor"] > 0))
        return ok, row["debit_minor"] or row["credit_minor"]
    ok = row["payment_id"] == doc["payment_id"] and row["account"] == ACC_CASH and row["debit_minor"] > 0
    return ok, row["debit_minor"]


def _biz_id(doc):
    return doc["payment_id"] if "payment_id" in doc else doc["invoice_id"]


def check_issue(c: Ctx, t: dict) -> list:
    """Re-derive the declared discrepancy from each evidence set. Returns violations."""
    v = []
    amount = t["expected_amount_minor"]
    if amount <= 0 or t["currency"] != c.base:
        v.append("%s: amount/currency invalid" % t["issue_id"])
    for es in t["evidence_sets"]:
        if len(es) > MAX_EVIDENCE:
            v.append("%s: evidence set too large" % t["issue_id"])
        unknown = [x for x in es if x not in c.rows and x not in c.docs and x not in c.fx]
        if unknown:
            v.append("%s: evidence ids not in public records" % t["issue_id"])
            continue
        docs = [c.docs[x] for x in es if x in c.docs]
        rows = [c.rows[x] for x in es if x in c.rows]
        fxs = [c.fx[x] for x in es if x in c.fx]
        typ = t["type"]
        try:
            if typ == "duplicate_invoice":
                ok = _check_dup(c, t, docs, rows)
            elif typ == "payment_allocation":
                ok = _check_alloc(c, t, docs, rows, fxs)
            elif typ == "fx_conversion":
                ok = _check_fx(c, t, docs, rows, fxs)
            else:
                ok = _check_cutoff(c, t, docs, rows)
        except Exception as exc:  # pragma: no cover - reported as violation
            ok = "exception %s" % type(exc).__name__
        if ok is not True:
            v.append("%s (%s): evidence set does not prove the issue: %s" % (t["issue_id"], typ, ok))
    return v


def _sig(d):
    return sorted((l["sku"], l["quantity"], l["unit_price_minor"], l["service_period"]) for l in d["line_items"])


def _check_dup(c, t, docs, rows):
    amt = t["expected_amount_minor"]
    if len(docs) == 2:
        a, b = sorted(docs, key=lambda d: parse_ts(d["issued_at"]))
        if not (a["entity_id"] == b["entity_id"] and a["currency"] == b["currency"] and a["po_ref"] == b["po_ref"]
                and a["total_minor"] == b["total_minor"] and _sig(a) == _sig(b) and a["invoice_id"] != b["invoice_id"]):
            return "documents are not duplicates"
        if not (c.inp(a["issued_at"]) and c.inp(b["issued_at"])):
            return "documents not in period"
        if sorted(t["affected_ids"]) != sorted([a["invoice_id"], b["invoice_id"]]):
            return "affected ids"
        if len(rows) != 1:
            return "expects one ledger row"
        ok, val = _primary(c, rows[0], b)
        return True if ok and val == amt else "row is not the duplicate posting or amount differs"
    if len(docs) == 1:
        d = docs[0]
        if len(rows) != 2 or rows[0]["journal_id"] == rows[1]["journal_id"]:
            return "expects two postings in different journals"
        vals = []
        for r in rows:
            ok, val = _primary(c, r, d)
            if not ok:
                return "row does not post this invoice"
            vals.append(val)
        if rows[0]["account"] != rows[1]["account"] or vals[0] != vals[1] or vals[1] != amt:
            return "postings differ"
        return True if t["affected_ids"] == [d["invoice_id"]] else "affected ids"
    return "expects one or two documents"


def _check_alloc(c, t, docs, rows, fxs):
    pays = [d for d in docs if "payment_id" in d]
    invs = [d for d in docs if "payment_id" not in d]
    if len(pays) != 1 or len(invs) != 1 or len(rows) != 1:
        return "expects payment doc, invoice doc and one ledger row"
    p, inv = pays[0], invs[0]
    if sorted(t["affected_ids"]) != sorted([p["payment_id"], inv["invoice_id"]]):
        return "affected ids"
    refs = {x["invoice_id"]: x["applied_minor"] for x in p["invoice_refs"] or []}
    if inv["invoice_id"] not in refs:
        return "payment does not reference invoice"
    if not c.inp(p["timestamp"]):
        return "payment not in period"
    if inv["currency"] == c.base:
        rate = Decimal(1)
        if fxs:
            return "unexpected fx row"
    else:
        if len(fxs) != 1:
            return "expects the invoice-date fx row"
        f = fxs[0]
        _, want_id = c.rate(inv["currency"], inv["issued_at"])
        if f["fx_id"] != want_id:
            return "fx row is not the policy rate for the invoice date"
        rate = Decimal(f["rate"])
    expected = c.conv(refs[inv["invoice_id"]], p["currency"], rate)
    booked = sum(r["credit_minor"] - r["debit_minor"] for r in c.ledger
                 if r["payment_id"] == p["payment_id"] and r["account"] == ACC_AR and r["invoice_id"] == inv["invoice_id"])
    if rows[0]["payment_id"] != p["payment_id"]:
        return "row is not part of the payment posting"
    return True if expected - booked == t["expected_amount_minor"] else "amount %d != %d" % (expected - booked, t["expected_amount_minor"])


def _check_fx(c, t, docs, rows, fxs):
    if len(docs) != 1 or len(rows) != 1 or len(fxs) != 1:
        return "expects document, row and fx rate"
    d, r, f = docs[0], rows[0], fxs[0]
    ts = d.get("issued_at") or d.get("timestamp")
    if not c.inp(ts):
        return "document not in period"
    _, want_id = c.rate(d["currency"], ts)
    if f["fx_id"] != want_id:
        return "fx row is not the policy rate"
    ok, booked = _primary(c, r, d)
    if not ok:
        return "row is not the primary line"
    amt_doc = d["total_minor"] if "payment_id" not in d else d["amount_minor"]
    correct = c.conv(amt_doc, d["currency"], Decimal(f["rate"]))
    if t["affected_ids"] != [_biz_id(d)]:
        return "affected ids"
    return True if abs(booked - correct) == t["expected_amount_minor"] else "amount"


def _check_cutoff(c, t, docs, rows):
    if len(docs) != 1 or len(rows) != 1:
        return "expects document and row"
    d, r = docs[0], rows[0]
    ts = d.get("issued_at") or d.get("timestamp")
    if c.inp(ts):
        return "document is in period"
    ok, booked = _primary(c, r, d)
    # the row is part of this period's ledger; its timestamp is either the document instant
    # (tz_boundary: outside the period in company time) or a back-dated in-period time
    if not ok:
        return "row is not a posting of the document"
    if t["affected_ids"] != [_biz_id(d)]:
        return "affected ids"
    return True if booked == t["expected_amount_minor"] else "amount"


# ---------------------------------------------------------------- episode
def verify_episode(public: dict, private: dict) -> dict:
    truth, prov = private["truth"], private["provenance"]
    checks = {}
    errors = []
    m = public["manifest"]
    counts_ok = (m["counts"]["ledger"] == len(public["ledger"]) and m["counts"]["invoices"] == len(public["invoices"])
                 and m["counts"]["payments"] == len(public["payments"]) and m["counts"]["fx"] == len(public["fx"]))
    checks["manifest_counts"] = counts_ok
    if not counts_ok:
        errors.append("manifest counts differ from files")
    ids = [r["row_id"] for r in public["ledger"]] + [d["doc_id"] for d in public["invoices"] + public["payments"]] + \
          [d["invoice_id"] for d in public["invoices"]] + [d["payment_id"] for d in public["payments"]] + \
          [f["fx_id"] for f in public["fx"]]
    checks["unique_ids"] = len(ids) == len(set(ids))
    if not checks["unique_ids"]:
        errors.append("duplicate public ids")
    # 1. pre-injection identities
    if prov is not None:
        led, inv, pay = revert(public, prov)
        pre = check_books(public, led, inv, pay)
        checks["pre_injection_identities"] = not pre
        errors += ["pre: " + x for x in pre[:20]]
    # 2. declared discrepancies, from evidence alone
    c = Ctx(public)
    decl = []
    for t in truth["issues"]:
        decl += check_issue(c, t)
    checks["declared_issues_proven_by_evidence"] = not decl
    errors += decl[:20]
    keys = [(t["type"], tuple(sorted(t["affected_ids"]))) for t in truth["issues"]]
    checks["issue_keys_unique"] = len(keys) == len(set(keys))
    checks["issue_count_le_12"] = len(truth["issues"]) <= 12
    # 3. completeness: public-only auditor == truth
    books = PublicBooks(m, public["ledger"], public["invoices"], public["payments"], public["fx"])
    found = audit(books)
    g = grade({"episode_id": m["episode_id"], "findings": found[:30]}, truth, 0,
              aliases_from_public(public["invoices"], public["payments"]))
    checks["reference_auditor_matches_truth"] = g["f1"] == 1.0 and g["fp"] == 0 and g["fn"] == 0
    if not checks["reference_auditor_matches_truth"]:
        errors.append("reference auditor tp=%d fp=%d fn=%d" % (g["tp"], g["fp"], g["fn"]))
    # 4. books after injection: only the declared issues break identities (clean => none)
    if truth["clean"]:
        post = check_books(public, public["ledger"], public["invoices"], public["payments"])
        checks["clean_books_have_no_violations"] = not post
        errors += ["clean: " + x for x in post[:20]]
    ok = all(checks.values())
    return {"episode_id": m["episode_id"], "ok": ok, "checks": checks, "errors": errors,
            "n_issues": len(truth["issues"]), "clean": truth["clean"]}

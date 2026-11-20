"""Shared, published matcher and terminal reward.

Normalisation
  * IDs and currency codes: ``str.strip().upper()``; affected_ids and evidence_ids are sets
    (order and repeats inside one finding are irrelevant).
  * Evidence aliases: an invoice_id / payment_id given as evidence is replaced by its doc_id.
  * ``type`` is matched case-insensitively against the four published types.

A finding is *eligible* for a truth issue iff ALL hold:
  type equal; affected_ids set equal to the canonical set; currency equal;
  |claimed - expected| <= max(abs_minor, floor(rel * expected)); and the evidence set
  contains at least one complete acceptable evidence set.

Matching is one-to-one maximum bipartite matching (Kuhn's augmenting paths). Ties are
resolved deterministically: findings in submission order, issues sorted by issue_id.
The TP count is the matching size, which is independent of tie-breaking and of the
order of findings. Unmatched findings (including repeats of a matched one) are FP;
unmatched truth issues are FN.

Reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1); invalid structure => -1.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_EVEN, Decimal

from .constants import (
    AMOUNT_TOL_ABS_MINOR, AMOUNT_TOL_REL, ISSUE_TYPES, MAX_AFFECTED, MAX_EVIDENCE, MAX_EXPLANATION,
    MAX_FINDINGS, MAX_ID_LEN, STEP_COST,
)
from .money import amount_tolerance

FINDING_KEYS_REQUIRED = {"type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids"}
FINDING_KEYS_OPTIONAL = {"explanation"}
TOP_KEYS = {"episode_id", "findings"}
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
MAX_AMOUNT = 10 ** 15


class SubmissionError(ValueError):
    pass


def _norm_id(x, what):
    if not isinstance(x, str):
        raise SubmissionError("%s must be strings" % what)
    s = x.strip().upper()
    if not s or len(s) > MAX_ID_LEN:
        raise SubmissionError("%s must be non-empty and at most %d chars" % (what, MAX_ID_LEN))
    return s


def normalize_submission(sub, episode_id: str, aliases: dict | None = None) -> list:
    """Validate findings.json and return normalised findings. Raises SubmissionError."""
    aliases = aliases or {}
    if not isinstance(sub, dict):
        raise SubmissionError("submission must be an object")
    extra = set(sub) - TOP_KEYS
    if extra:
        raise SubmissionError("unknown top-level keys")
    if sub.get("episode_id") != episode_id:
        raise SubmissionError("episode_id missing or does not match")
    fs = sub.get("findings")
    if not isinstance(fs, list):
        raise SubmissionError("findings must be a list")
    if len(fs) > MAX_FINDINGS:
        raise SubmissionError("at most %d findings" % MAX_FINDINGS)
    out = []
    for i, f in enumerate(fs):
        if not isinstance(f, dict):
            raise SubmissionError("finding %d must be an object" % i)
        keys = set(f)
        if not FINDING_KEYS_REQUIRED <= keys or keys - FINDING_KEYS_REQUIRED - FINDING_KEYS_OPTIONAL:
            raise SubmissionError("finding %d has missing or unknown keys" % i)
        t = f["type"]
        if not isinstance(t, str) or t.strip().lower() not in ISSUE_TYPES:
            raise SubmissionError("finding %d has unknown type" % i)
        aff = f["affected_ids"]
        if not isinstance(aff, list) or not (1 <= len(aff) <= MAX_AFFECTED):
            raise SubmissionError("finding %d: affected_ids must be a list of 1..%d ids" % (i, MAX_AFFECTED))
        ev = f["evidence_ids"]
        if not isinstance(ev, list) or len(ev) > MAX_EVIDENCE:
            raise SubmissionError("finding %d: evidence_ids must be a list of at most %d ids" % (i, MAX_EVIDENCE))
        amt = f["claimed_amount_minor"]
        if isinstance(amt, bool) or not isinstance(amt, int) or not (0 <= amt <= MAX_AMOUNT):
            raise SubmissionError("finding %d: claimed_amount_minor must be a non-negative integer" % i)
        cur = f["currency"]
        if not isinstance(cur, str) or not CURRENCY_RE.match(cur.strip().upper()):
            raise SubmissionError("finding %d: currency must be a 3-letter code" % i)
        expl = f.get("explanation", "")
        if not isinstance(expl, str) or len(expl) > MAX_EXPLANATION:
            raise SubmissionError("finding %d: explanation must be a string of at most %d chars" % (i, MAX_EXPLANATION))
        ev_ids = [_norm_id(x, "evidence_ids") for x in ev]
        out.append({
            "index": i,
            "type": t.strip().lower(),
            "affected": frozenset(_norm_id(x, "affected_ids") for x in aff),
            "amount": int(amt),
            "currency": cur.strip().upper(),
            "evidence": frozenset(aliases.get(x, x) for x in ev_ids),
        })
    return out


def tolerance(expected: int) -> int:
    return amount_tolerance(expected, AMOUNT_TOL_ABS_MINOR, AMOUNT_TOL_REL)


def _check(f, t):
    """Return None if eligible, else the first failing criterion."""
    if f["type"] != t["type"]:
        return "wrong_type"
    if f["affected"] != frozenset(t["affected_ids"]):
        return "wrong_affected_ids"
    if f["currency"] != t["currency"]:
        return "wrong_currency"
    if abs(f["amount"] - int(t["expected_amount_minor"])) > tolerance(int(t["expected_amount_minor"])):
        return "amount_out_of_tolerance"
    if not any(frozenset(s) <= f["evidence"] for s in t["evidence_sets"]):
        return "insufficient_evidence"
    return None


def match(findings: list, issues: list, observed: set | None = None) -> dict:
    issues = sorted(issues, key=lambda t: t["issue_id"])
    if observed is not None:
        findings = [dict(f, evidence=frozenset(e for e in f["evidence"] if e in observed)) for f in findings]
    elig = [[j for j, t in enumerate(issues) if _check(f, t) is None] for f in findings]
    owner = [-1] * len(issues)

    def augment(i, seen):
        for j in elig[i]:
            if j in seen:
                continue
            seen.add(j)
            if owner[j] == -1 or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(findings)):
        augment(i, set())
    matched_f = {owner[j]: j for j in range(len(issues)) if owner[j] != -1}
    diags = []
    for i, f in enumerate(findings):
        if i in matched_f:
            diags.append({"finding": f["index"], "status": "TP", "issue_id": issues[matched_f[i]]["issue_id"]})
            continue
        if elig[i]:
            reason = "duplicate_of_matched_finding"
        else:
            same = [t for t in issues if frozenset(t["affected_ids"]) == f["affected"]]
            if same:
                reason = min((_check(f, t) for t in same), key=lambda r: [
                    "insufficient_evidence", "amount_out_of_tolerance", "wrong_currency", "wrong_type",
                    "wrong_affected_ids"].index(r))
            else:
                overlap = [t for t in issues if frozenset(t["affected_ids"]) & f["affected"]]
                reason = "wrong_affected_ids" if overlap else "no_such_issue"
        diags.append({"finding": f["index"], "status": "FP", "type": f["type"], "reason": reason})
    tp = len(matched_f)
    missed = [{"issue_id": t["issue_id"], "type": t["type"], "status": "FN"}
              for j, t in enumerate(issues) if owner[j] == -1]
    return {"tp": tp, "fp": len(findings) - tp, "fn": len(issues) - tp, "findings": diags, "missed": missed}


def metrics(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if tp + fp + fn == 0:
        f1 = 1.0
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6)}


def terminal_reward(tp: int, fp: int, fn: int, inspection_steps: int) -> float:
    if tp + fp + fn == 0:
        f1 = Decimal(1)
    else:
        f1 = Decimal(2 * tp) / Decimal(2 * tp + fp + fn)
    raw = 2 * f1 - 1 - STEP_COST * int(inspection_steps)
    raw = max(Decimal(-1), min(Decimal(1), raw))
    return float(raw.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN))


def grade(submission, truth: dict, inspection_steps: int, aliases: dict | None = None,
          observed: set | None = None) -> dict:
    """Grade a findings.json object against truth.json. Pure function, no I/O."""
    issues = truth["issues"]
    try:
        findings = normalize_submission(submission, truth["episode_id"], aliases)
    except SubmissionError as exc:
        return {"valid": False, "error": str(exc), "tp": 0, "fp": 0, "fn": len(issues),
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "reward": -1.0,
                "inspection_steps": int(inspection_steps), "findings": [], "missed": []}
    m = match(findings, issues, observed)
    res = {"valid": True, "error": None, "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]}
    res.update(metrics(m["tp"], m["fp"], m["fn"]))
    res["reward"] = terminal_reward(m["tp"], m["fp"], m["fn"], inspection_steps)
    res["inspection_steps"] = int(inspection_steps)
    res["findings"] = m["findings"]
    res["missed"] = m["missed"]
    return res


def aliases_from_public(invoices, payments) -> dict:
    a = {}
    for i in invoices:
        a[i["invoice_id"].upper()] = i["doc_id"].upper()
    for p in payments:
        a[p["payment_id"].upper()] = p["doc_id"].upper()
    return a

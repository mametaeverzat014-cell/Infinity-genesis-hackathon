"""AuditEnv: JSON-compatible Gymnasium-style environment.

    obs, info = env.reset(seed=..., config=...)
    obs, reward, terminated, truncated, info = env.step(action)

Actions (JSON objects):
    {"action": "list_records", "table": "ledger|invoices|payments|fx", "cursor": null, "limit": 50}
    {"action": "get_record", "table": "...", "id": "..."}
    {"action": "get_document", "doc_id": "DOC-..."}
    {"action": "calculator", "expression": "(1250 * 1.0842) - 1355"}
    {"action": "submit", "findings": {"episode_id": "...", "findings": [...]}}

Modes:
    train - open training mode: the terminal reward (and aggregate metrics) are returned.
    eval  - closed evaluation: reward is always 0.0 to the policy; reward and diagnostics
            are written only to the grader log, which the agent cannot read.
"""

from __future__ import annotations

import copy
import json
import time

from . import calculator
from .config import load_config
from .constants import (
    ENV_VERSION, MAX_ERROR_MESSAGE, MAX_STEPS, PAGE_LIMIT, SCHEMA_VERSION, TABLES, TIME_LIMIT_S,
)
from .episode import read_private, read_public
from .generator import generate
from .matcher import aliases_from_public, grade

TASK_TEXT = (
    "You are auditing one company for one reporting period. Records are read-only and reachable only "
    "through the tools. Find misstatements of four types (duplicate_invoice, payment_allocation, "
    "fx_conversion, period_cutoff) and submit findings with evidence ids. Amounts are ledger misstatements "
    "in base-currency minor units as defined in policy.findings. Legitimate look-alikes exist (recurring "
    "amounts, split or short payments, overpayments, refunds, documents near midnight). Submitting nothing "
    "is correct for clean books. Every action, including errors, uses one step; submit ends the episode."
)

TOOLS = {
    "list_records": {"args": {"table": list(TABLES), "cursor": "string|null", "limit": "1..%d" % PAGE_LIMIT},
                     "returns": "rows, next_cursor (null at end), total"},
    "get_record": {"args": {"table": list(TABLES), "id": "row_id | invoice_id | payment_id | doc_id | fx_id"}},
    "get_document": {"args": {"doc_id": "doc_id"}, "returns": "full source document (line items, po_ref, ...)"},
    "calculator": {"args": {"expression": "numbers, + - * / and parentheses only"}},
    "submit": {"args": {"findings": "findings.json object {episode_id, findings:[...]}"}},
}

SUBMISSION_FORMAT = {
    "episode_id": "string (from manifest)",
    "findings": [{"type": "one of issue_types", "affected_ids": ["..."], "claimed_amount_minor": 0,
                  "currency": "base currency code", "evidence_ids": ["... (max 10)"],
                  "explanation": "short justification (max 1000 chars)"}],
    "limits": {"max_findings": 30, "max_evidence_ids": 10, "max_explanation_chars": 1000},
}

ID_FIELDS = ("row_id", "journal_id", "doc_id", "invoice_id", "payment_id", "fx_id", "entity_id")


class EnvError(RuntimeError):
    pass


def _err(code: str, message: str) -> dict:
    return {"code": code, "message": message[:MAX_ERROR_MESSAGE]}


class AuditEnv:
    def __init__(self, mode: str = "train", grader_log: str | None = None, default_config="train"):
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode = mode
        self.grader_log = grader_log
        self.default_config = default_config
        self._ready = False
        self.last_evaluation = None

    # ------------------------------------------------------------ lifecycle
    def reset(self, seed=None, config=None, options=None):
        options = dict(options or {})
        if config is None:
            config = options.pop("config", None)
        episode_dir = options.pop("episode_dir", None)
        if episode_dir is not None:
            public = read_public(episode_dir)
            private = read_private(episode_dir)
            env_cfg = load_config(None)["env"]
            source = {"episode_dir": True}
        else:
            if seed is None:
                raise EnvError("reset requires a seed (or options.episode_dir)")
            cfg = load_config(config if config is not None else self.default_config)
            public, private = generate(seed, cfg)
            env_cfg = cfg["env"]
            source = {"seed": str(seed), "config": cfg["name"]}
        self._load(public, private, env_cfg, source)
        obs = {
            "kind": "reset",
            "task": TASK_TEXT,
            "episode_id": self.episode_id,
            "manifest": copy.deepcopy(public["manifest"]),
            "policy": copy.deepcopy(public["policy"]),
            "tools": TOOLS,
            "submission_format": SUBMISSION_FORMAT,
            "steps_remaining": self.max_steps,
        }
        return obs, self._info()

    def _load(self, public, private, env_cfg, source):
        self.public = public
        self.truth = private["truth"]
        self.episode_id = public["manifest"]["episode_id"]
        self.max_steps = min(MAX_STEPS, int(env_cfg.get("max_steps", MAX_STEPS)))
        self.time_limit = float(env_cfg.get("time_limit_s", TIME_LIMIT_S))
        self.strict_evidence = bool(env_cfg.get("strict_evidence", False))
        self._source = source
        self.aliases = aliases_from_public(public["invoices"], public["payments"])
        # indices (fresh per episode: reset fully isolates state)
        self._tables = {
            "ledger": [dict(r) for r in public["ledger"]],
            "invoices": [self._invoice_index(d) for d in public["invoices"]],
            "payments": [self._payment_index(d) for d in public["payments"]],
            "fx": [dict(r) for r in public["fx"]],
        }
        self._by_id = {t: {} for t in TABLES}
        for r in self._tables["ledger"]:
            self._by_id["ledger"][r["row_id"]] = r
        for r in self._tables["invoices"]:
            self._by_id["invoices"][r["invoice_id"]] = r
            self._by_id["invoices"][r["doc_id"]] = r
        for r in self._tables["payments"]:
            self._by_id["payments"][r["payment_id"]] = r
            self._by_id["payments"][r["doc_id"]] = r
        for r in self._tables["fx"]:
            self._by_id["fx"][r["fx_id"]] = r
        self._docs = {}
        for d in public["invoices"]:
            self._docs[d["doc_id"]] = dict(copy.deepcopy(d), doc_type="invoice")
        for d in public["payments"]:
            self._docs[d["doc_id"]] = dict(copy.deepcopy(d), doc_type="payment")
        self.steps_used = 0
        self.terminated = False
        self.truncated = False
        self.observed = set()
        self._t0 = time.monotonic()
        self._ready = True
        self.last_evaluation = None

    @staticmethod
    def _invoice_index(d):
        r = {k: v for k, v in d.items() if k not in ("line_items", "po_ref")}
        r["line_count"] = len(d.get("line_items") or [])
        return r

    @staticmethod
    def _payment_index(d):
        return {k: copy.deepcopy(v) for k, v in d.items() if k != "bank_reference"}

    def _info(self, extra=None):
        info = {"episode_id": self.episode_id, "mode": self.mode, "steps_used": self.steps_used,
                "steps_remaining": self.max_steps - self.steps_used, "env_version": ENV_VERSION,
                "schema_version": SCHEMA_VERSION}
        if extra:
            info.update(extra)
        return info

    def _see(self, records):
        for r in records:
            for f in ID_FIELDS:
                v = r.get(f)
                if isinstance(v, str):
                    self.observed.add(v.upper())
            for ref in r.get("invoice_refs") or []:
                self.observed.add(ref["invoice_id"].upper())

    # ---------------------------------------------------------------- step
    def step(self, action):
        if not self._ready:
            raise EnvError("call reset() before step()")
        if self.terminated or self.truncated:
            obs = {"kind": "step", "step": self.steps_used, "action": None, "result": None,
                   "error": _err("episode_finished", "episode already finished; call reset()"),
                   "steps_remaining": self.max_steps - self.steps_used}
            return obs, 0.0, self.terminated, self.truncated, self._info()
        self.steps_used += 1
        name = action.get("action") if isinstance(action, dict) else None
        if time.monotonic() - self._t0 > self.time_limit:
            return self._finish(None, "time_limit", name)
        result, error = None, None
        if not isinstance(action, dict) or not isinstance(name, str):
            error = _err("invalid_action", "action must be an object with a string 'action' field")
        elif name == "submit":
            return self._finish(action, "submitted", name)
        elif name == "list_records":
            result, error = self._list(action)
        elif name == "get_record":
            result, error = self._get(action)
        elif name == "get_document":
            result, error = self._get_doc(action)
        elif name == "calculator":
            try:
                result = {"value": calculator.evaluate(action.get("expression"))}
            except calculator.CalcError as exc:
                error = _err("calc_error", str(exc))
        else:
            error = _err("unknown_action", "unknown action; use list_records, get_record, get_document, calculator, submit")
        if self.steps_used >= self.max_steps:
            return self._finish(None, "step_limit", name if isinstance(name, str) and len(name) < 32 else None,
                                result=result, error=error)
        obs = {"kind": "step", "step": self.steps_used, "action": name if name in TOOLS else None,
               "result": result, "error": error, "steps_remaining": self.max_steps - self.steps_used}
        return obs, 0.0, False, False, self._info()

    def _table_arg(self, action):
        t = action.get("table")
        if not isinstance(t, str) or t not in TABLES:
            return None, _err("unknown_table", "table must be one of ledger, invoices, payments, fx")
        return t, None

    def _list(self, action):
        t, e = self._table_arg(action)
        if e:
            return None, e
        limit = action.get("limit", PAGE_LIMIT)
        if limit is None:
            limit = PAGE_LIMIT
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return None, _err("invalid_argument", "limit must be a positive integer (max %d)" % PAGE_LIMIT)
        limit = min(limit, PAGE_LIMIT)
        cursor = action.get("cursor")
        offset = 0
        if cursor is not None:
            ok = isinstance(cursor, str) and cursor.startswith("c1." + t + ".")
            if ok:
                tail = cursor[len("c1." + t + "."):]
                ok = tail.isdigit() and len(tail) < 7 and int(tail) <= len(self._tables[t])
            if not ok:
                return None, _err("bad_cursor", "cursor is not valid for this table")
            offset = int(tail)
        rows = copy.deepcopy(self._tables[t][offset: offset + limit])
        nxt = offset + len(rows)
        self._see(rows)
        return {"table": t, "rows": rows, "next_cursor": ("c1.%s.%d" % (t, nxt)) if nxt < len(self._tables[t]) else None,
                "total": len(self._tables[t]), "limit": limit}, None

    def _get(self, action):
        t, e = self._table_arg(action)
        if e:
            return None, e
        rid = action.get("id")
        if not isinstance(rid, str) or len(rid) > 64:
            return None, _err("invalid_argument", "id must be a string")
        rec = self._by_id[t].get(rid.strip().upper())
        if rec is None:
            return None, _err("not_found", "no record with that id in this table")
        rec = copy.deepcopy(rec)
        self._see([rec])
        return {"table": t, "record": rec}, None

    def _get_doc(self, action):
        did = action.get("doc_id")
        if not isinstance(did, str) or len(did) > 64:
            return None, _err("invalid_argument", "doc_id must be a string")
        doc = self._docs.get(did.strip().upper())
        if doc is None:
            return None, _err("not_found", "no document with that doc_id")
        doc = copy.deepcopy(doc)
        self._see([doc])
        return {"document": doc}, None

    # ------------------------------------------------------------ terminal
    def _submission_from(self, action):
        if action is None:
            return {"episode_id": self.episode_id, "findings": []}
        f = action.get("findings")
        if isinstance(f, list):
            return {"episode_id": action.get("episode_id"), "findings": f}
        return f

    def _finish(self, action, reason, name, result=None, error=None):
        submitted = reason == "submitted"
        inspection = self.steps_used - 1 if submitted else self.steps_used
        if submitted and set(action) - {"action", "findings", "episode_id"}:
            submission = None  # unknown keys in the submit action -> invalid structure
        else:
            submission = self._submission_from(action)
        g = grade(submission, self.truth, inspection, self.aliases,
                  self.observed if self.strict_evidence else None)
        self.terminated = submitted
        self.truncated = not submitted
        elapsed = round(time.monotonic() - self._t0, 4)
        record = {"episode_id": self.episode_id, "mode": self.mode, "end": reason,
                  "terminated": self.terminated, "truncated": self.truncated,
                  "steps_used": self.steps_used, "elapsed_s": elapsed, "grade": g}
        self.last_evaluation = record
        if self.grader_log:
            with open(self.grader_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        status = {"status": "submitted" if submitted else "truncated"}
        if not submitted:
            status["reason"] = reason
        if result is not None:
            status["last_result"] = result
        obs = {"kind": "terminal", "step": self.steps_used, "action": name if name in TOOLS else None,
               "result": status, "error": error, "steps_remaining": self.max_steps - self.steps_used}
        if self.mode == "train":
            extra = {"metrics": {k: g[k] for k in ("valid", "tp", "fp", "fn", "precision", "recall", "f1",
                                                   "inspection_steps")}}
            if not g["valid"]:
                extra["submission_error"] = g["error"]
            return obs, g["reward"], self.terminated, self.truncated, self._info(extra)
        return obs, 0.0, self.terminated, self.truncated, self._info()

    def evaluation(self):
        """Evaluator-side accessor (never exposed through the agent protocol)."""
        return copy.deepcopy(self.last_evaluation)

    def close(self):
        self._ready = False

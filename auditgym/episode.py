"""Episode files: public records (agent-visible through tools) and private grader files."""

from __future__ import annotations

import csv
import io
import json
import os

LEDGER_COLUMNS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                  "entity_id", "invoice_id", "payment_id", "journal_id"]
FX_COLUMNS = ["fx_id", "date", "source_currency", "target_currency", "rate"]
INT_COLUMNS = {"debit_minor", "credit_minor"}
NULL = "null"

PUBLIC_FILES = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv")
PRIVATE_FILES = ("truth.json", "provenance.json")


def _dump_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def _csv_text(rows, columns) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow([NULL if r[c] is None else r[c] for c in columns])
    return buf.getvalue()


def _jsonl_text(rows) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)


def public_file_texts(public: dict) -> dict:
    return {
        "manifest.json": _dump_json(public["manifest"]),
        "policy.json": _dump_json(public["policy"]),
        "ledger.csv": _csv_text(public["ledger"], LEDGER_COLUMNS),
        "invoices.jsonl": _jsonl_text(public["invoices"]),
        "payments.jsonl": _jsonl_text(public["payments"]),
        "fx.csv": _csv_text(public["fx"], FX_COLUMNS),
    }


def write_episode(directory: str, public: dict, private: dict | None = None) -> None:
    pub = os.path.join(directory, "public")
    os.makedirs(pub, exist_ok=True)
    for name, text in public_file_texts(public).items():
        with open(os.path.join(pub, name), "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    if private is not None:
        prv = os.path.join(directory, "private")
        os.makedirs(prv, exist_ok=True)
        with open(os.path.join(prv, "truth.json"), "w", encoding="utf-8") as fh:
            fh.write(_dump_json(private["truth"]))
        with open(os.path.join(prv, "provenance.json"), "w", encoding="utf-8") as fh:
            fh.write(_dump_json(private["provenance"]))


def _read_csv(path, columns):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        if header != columns:
            raise ValueError("unexpected header in %s" % os.path.basename(path))
        out = []
        for line in rd:
            rec = {}
            for c, v in zip(columns, line):
                if v == NULL:
                    rec[c] = None
                elif c in INT_COLUMNS:
                    rec[c] = int(v)
                else:
                    rec[c] = v
            out.append(rec)
        return out


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_public(directory: str) -> dict:
    pub = os.path.join(directory, "public")
    with open(os.path.join(pub, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(os.path.join(pub, "policy.json"), encoding="utf-8") as fh:
        policy = json.load(fh)
    return {
        "manifest": manifest,
        "policy": policy,
        "ledger": _read_csv(os.path.join(pub, "ledger.csv"), LEDGER_COLUMNS),
        "invoices": _read_jsonl(os.path.join(pub, "invoices.jsonl")),
        "payments": _read_jsonl(os.path.join(pub, "payments.jsonl")),
        "fx": _read_csv(os.path.join(pub, "fx.csv"), FX_COLUMNS),
    }


def read_private(directory: str) -> dict:
    prv = os.path.join(directory, "private")
    with open(os.path.join(prv, "truth.json"), encoding="utf-8") as fh:
        truth = json.load(fh)
    prov = None
    p = os.path.join(prv, "provenance.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            prov = json.load(fh)
    return {"truth": truth, "provenance": prov}

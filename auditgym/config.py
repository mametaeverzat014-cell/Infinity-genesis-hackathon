"""Generator configuration: defaults, presets, validation and hashing.

Every knob that changes generated data lives here, so (seed, config, ENV_VERSION)
fully determines an episode. Configs are versioned via ``config_version``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any

from .constants import CURRENCY_EXPONENTS, ISSUE_TYPES, SCENARIOS

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

DEFAULT_CONFIG: dict[str, Any] = {
    "config_version": "1.0.0",
    "name": "default",
    # company / period
    "base_currencies": ["USD", "EUR", "KZT", "GBP", "JPY"],
    "foreign_currencies": ["EUR", "USD", "GBP", "CHF", "CNY", "JPY", "KWD", "KZT"],
    "n_foreign_currencies": [1, 3],
    "timezones": ["-08:00", "-05:00", "-03:00", "+00:00", "+01:00", "+03:00",
                  "+05:00", "+05:30", "+06:00", "+09:00"],
    "years": [2025, 2026],
    # sizes
    "n_customers": [8, 20],
    "n_vendors": [4, 10],
    "n_invoices": [30, 75],           # in-period invoices
    "prior_invoice_share": 0.25,      # open invoices issued in the previous period
    "n_next_period_invoices": [2, 5],
    "foreign_customer_share": 0.4,
    "ledger_rows": [300, 1200],       # target size incl. expense noise
    # distractors (legitimate, correctly booked look-alikes)
    "distractors": {
        "recurring_same_amount": [1, 3],   # same lines/total, different PO
        "partial_deliveries": [1, 3],      # same PO, different lines
        "split_payments": [1, 4],          # one invoice settled by two payments
        "multi_invoice_payments": [1, 4],
        "short_payments": [1, 3],          # legit partial settlement, invoice stays open
        "overpayments": [1, 3],            # remainder legitimately unapplied
        "advances": [1, 3],                # payment without invoice reference (null)
        "refunds": [0, 2],
        "tz_boundary_docs": [1, 3],        # UTC date and local date fall in different periods
        "near_midnight_share": 0.12,       # docs whose UTC date != local date
        "next_period_payments": [1, 3],
    },
    # errors: each type is independently configurable
    "clean_probability": 0.0,
    "errors": {
        "duplicate_invoice": {"count": [0, 2], "variants": ["duplicate_document", "double_posting"]},
        "payment_allocation": {"count": [0, 2], "variants": ["missing", "partial"]},
        "fx_conversion": {"count": [0, 2], "variants": ["utc_date", "wrong_date", "inverted", "digit_typo"]},
        "period_cutoff": {"count": [0, 2], "variants": ["tz_boundary", "backdated"]},
    },
    "scenarios": {
        "partial_fx_payment_at_boundary": {"count": [0, 1]},
        "duplicate_with_misapplied_payment": {"count": [0, 1]},
    },
    "min_issues": 0,
    "max_issues": 12,
    # harmless transformations (paired invariance checks)
    "id_salt": "",
    "order_salt": "",
    "shuffle_records": False,
    # episode runtime
    "env": {"max_steps": 100, "time_limit_s": 60.0, "strict_evidence": False},
}

PRESETS = ("train", "dev", "clean", "interaction", "eval_mixed", "eval_clean", "eval_interaction", "smoke", "stress")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ConfigError(ValueError):
    pass


def _range(v, name, lo=0, hi=10**6):
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v)):
        raise ConfigError("%s must be [min, max] integers" % name)
    if not (lo <= v[0] <= v[1] <= hi):
        raise ConfigError("%s out of range" % name)


def validate(cfg: dict) -> dict:
    unknown = set(cfg) - set(DEFAULT_CONFIG) - {"description"}
    if unknown:
        raise ConfigError("unknown config keys: %s" % sorted(unknown))
    for c in cfg["base_currencies"] + cfg["foreign_currencies"]:
        if c not in CURRENCY_EXPONENTS:
            raise ConfigError("unsupported currency %s" % c)
    for tz in cfg["timezones"]:
        if not (isinstance(tz, str) and len(tz) == 6 and tz[0] in "+-" and tz[3] == ":"):
            raise ConfigError("timezone must look like +05:00")
    for name in ("n_customers", "n_vendors", "n_invoices", "n_next_period_invoices", "ledger_rows", "n_foreign_currencies", "years"):
        _range(cfg[name], name, 0, 10**5)
    if cfg["n_invoices"][1] > 160:
        raise ConfigError("n_invoices max is 160 (document budget)")
    if cfg["ledger_rows"][1] > 2000:
        raise ConfigError("ledger_rows max is 2000")
    for k, v in cfg["distractors"].items():
        if k == "near_midnight_share":
            if not (0 <= v <= 0.5):
                raise ConfigError("near_midnight_share in [0, 0.5]")
        else:
            _range(v, "distractors." + k, 0, 20)
    if set(cfg["errors"]) - set(ISSUE_TYPES):
        raise ConfigError("unknown error type")
    for t, spec in cfg["errors"].items():
        _range(spec["count"], "errors.%s.count" % t, 0, 12)
    if set(cfg["scenarios"]) - set(SCENARIOS):
        raise ConfigError("unknown scenario")
    for s, spec in cfg["scenarios"].items():
        _range(spec["count"], "scenarios.%s.count" % s, 0, 6)
    if not (0 <= cfg["max_issues"] <= 12):
        raise ConfigError("max_issues must be within 0..12")
    if not (0.0 <= float(cfg["clean_probability"]) <= 1.0):
        raise ConfigError("clean_probability in [0,1]")
    env = cfg["env"]
    if not (1 <= int(env["max_steps"]) <= 100):
        raise ConfigError("env.max_steps within 1..100")
    return cfg


def load_config(spec: Any = None) -> dict:
    """Accepts None, a preset name, a path to a JSON file or a dict (merged over defaults)."""
    if spec is None:
        override = {}
    elif isinstance(spec, dict):
        override = spec
        if "preset" in override:
            base = load_config(override["preset"])
            rest = {k: v for k, v in override.items() if k != "preset"}
            return validate(_deep_merge(base, rest))
    elif isinstance(spec, str):
        path = spec if spec.endswith(".json") else os.path.join(CONFIG_DIR, spec + ".json")
        if not spec.endswith(".json") and spec not in PRESETS:
            raise ConfigError("unknown preset %r" % spec)
        with open(path, "r", encoding="utf-8") as fh:
            override = json.load(fh)
    else:
        raise ConfigError("config must be None, preset name, path or dict")
    return validate(_deep_merge(DEFAULT_CONFIG, override))


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(cfg: dict) -> str:
    """Hash of everything that influences generated data (runtime ``env`` excluded)."""
    gen = {k: v for k, v in cfg.items() if k not in ("env", "name", "description")}
    return hashlib.sha256(canonical_json(gen).encode()).hexdigest()

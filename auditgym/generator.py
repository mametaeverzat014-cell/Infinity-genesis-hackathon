"""Synthetic company generator.

Pipeline (each stage uses its own RNG stream derived from the seed):

1. ``history``  - hidden, internally consistent company history: customers, vendors,
   invoices, payments (with legitimate look-alike distractors), FX table.
2. ``posting``  - a clean double-entry ledger derived from the history by the
   published accounting policy (see ``policy.json``).
3. ``inject``   - errors are injected as *causal changes* (row updates / inserts /
   deletes and document inserts). Every change is logged in provenance and every
   injected discrepancy gets a truth record with its expected amount and the
   evidence sets that prove it from public records alone.
4. ``render``   - public IDs are drawn from an independent ID stream (``id_salt``)
   and records are ordered by an independent order stream (``order_salt``), so
   ID/order transformations never change the underlying history.

No floating point arithmetic is used for data: only integer randomness and Decimal.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, localcontext

from .config import canonical_json, config_hash, load_config
from .constants import (
    ACC_AR, ACC_CASH, ACC_FX, ACC_REVENUE, ACC_UNAPPLIED, ACCOUNTS, AMOUNT_TOL_ABS_MINOR,
    AMOUNT_TOL_REL, CURRENCY_EXPONENTS, ENV_VERSION, EXPENSE_ACCOUNTS, ISSUE_TYPES,
    POLICY_VERSION, SCHEMA_VERSION, USD_VALUE,
)
from .money import RateBook, amount_tolerance, convert, fmt_rate, fmt_ts, offset_minutes

import random

ID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
TRANSFORM_KEYS = ("id_salt", "order_salt", "shuffle_records")

NAMES_A = ["Altyn", "Borealis", "Cedar", "Delta", "Ember", "Fjord", "Granite", "Helix", "Iris",
           "Juniper", "Kestrel", "Lumen", "Meridian", "Nimbus", "Orion", "Pioneer", "Quartz",
           "Riverside", "Summit", "Tundra", "Umbra", "Vertex", "Willow", "Zephyr", "Saryarka",
           "Baikonur", "Steppe", "Aral"]
NAMES_B = ["Logistics", "Components", "Analytics", "Foods", "Textiles", "Energy", "Systems",
           "Retail", "Pharma", "Media", "Robotics", "Agro", "Mining", "Freight"]
NAMES_C = ["LLP", "GmbH", "Ltd", "Inc", "SA", "TOO", "BV", "KK"]

CATALOG = [
    ("SRV-CONS", "Consulting hours", 90, 260),
    ("SRV-SUPP", "Support subscription", 300, 3500),
    ("SRV-IMPL", "Implementation milestone", 1500, 9000),
    ("HW-SENS", "Sensor module", 25, 400),
    ("HW-GATE", "Edge gateway", 400, 2200),
    ("LIC-ENT", "Enterprise licence seat", 60, 900),
    ("LOG-FRT", "Freight handling", 120, 1800),
    ("TRN-WS", "Training workshop", 700, 4200),
    ("MAT-STL", "Steel fittings batch", 80, 1300),
    ("SRV-AUD", "Data quality review", 500, 5000),
]


def stream(*labels) -> random.Random:
    h = hashlib.sha256(canonical_json([str(x) for x in labels]).encode()).digest()
    return random.Random(int.from_bytes(h[:16], "big"))


def _q6(d: Decimal) -> Decimal:
    """Quantise to 6 significant digits."""
    with localcontext() as ctx:
        ctx.prec = 40
        return d.quantize(Decimal(10) ** (d.adjusted() - 5))


class GenerationError(RuntimeError):
    pass


SECRET_ENV = "AUDITGYM_SEED_SECRET"


class Generator:
    def __init__(self, seed, cfg: dict, secret: str = ""):
        """``secret``: optional evaluator-only key mixed into every RNG stream. With it, knowing the
        public generator and even a small seed is not enough to regenerate an episode."""
        self.seed = seed
        self.cfg = cfg
        self.keyed = bool(secret)
        sk = hashlib.sha256(("auditgym-secret:" + secret).encode()).hexdigest() if secret else ""
        data_cfg = {k: v for k, v in cfg.items() if k not in TRANSFORM_KEYS}
        self.data_hash = config_hash(data_cfg)
        self.full_hash = config_hash(cfg)
        key = (sk,) if sk else ()
        self.r = stream("history", seed, self.data_hash, ENV_VERSION, *key)
        self.ri = stream("inject", seed, self.data_hash, ENV_VERSION, *key)
        self.rid = stream("ids", seed, self.data_hash, cfg["id_salt"], ENV_VERSION, *key)
        self.rord = stream("order", seed, self.data_hash, cfg["order_salt"], ENV_VERSION, *key)
        self.episode_id = "ep-" + hashlib.sha256(
            canonical_json(["episode", str(seed), self.full_hash, ENV_VERSION, *key]).encode()).hexdigest()[:16]
        self._k = 0
        self.entities = {}
        self.invoices = {}
        self.payments = {}
        self.rows = {}           # lid -> row (live ledger)
        self.all_rows = {}       # lid -> row snapshot for every row ever created
        self.journals = {}       # jid -> [lid]
        self.doc_journal = {}    # ("inv"|"pay", k) -> [jid]
        self.booked_pay_rate = {}
        self.used = set()        # ("inv"|"pay", k) used as error target
        self.issues = []
        self.skipped = []
        self.distractor_log = []
        self._po_counter = 0
        self._lid = 0
        self._jid = 0

    # ------------------------------------------------------------------ utils
    def nk(self) -> int:
        self._k += 1
        return self._k

    def at_local(self, d: date, minute_of_day: int, second: int = 0) -> datetime:
        return datetime.combine(d, time(0, 0), tzinfo=self.tz) + timedelta(minutes=minute_of_day, seconds=second)

    def ldate(self, dt: datetime) -> date:
        return dt.astimezone(self.tz).date()

    def in_period(self, dt: datetime) -> bool:
        return self.start <= dt < self.end

    def rate(self, currency: str, d: date) -> Decimal:
        hit = self.ratebook.lookup(currency, self.base, d)
        if hit is None:
            raise GenerationError("no FX rate for %s on %s" % (currency, d))
        return hit[0]

    def rate_key(self, currency: str, d: date):
        hit = self.ratebook.lookup(currency, self.base, d)
        return None if hit is None else hit[1]

    def conv(self, amount: int, currency: str, rate: Decimal) -> int:
        return convert(amount, currency, self.base, rate)

    def sample_minute(self, rng, near_midnight: bool) -> int:
        """Minute-of-day in local time. near_midnight => UTC date differs from local date."""
        if near_midnight and self.tz_min != 0:
            if self.tz_min > 0:
                return rng.randint(5, self.tz_min - 5)
            return rng.randint(1440 + self.tz_min + 5, 1435)
        if rng.random() < 0.8:
            return rng.randint(8 * 60, 19 * 60)
        return rng.randint(0, 1439)

    def random_instant(self, rng, d0: date, d1: date, near_share: float = None) -> datetime:
        days = (d1 - d0).days
        d = d0 + timedelta(days=rng.randint(0, max(0, days)))
        share = self.cfg["distractors"]["near_midnight_share"] if near_share is None else near_share
        near = rng.random() < share
        return self.at_local(d, self.sample_minute(rng, near), rng.randint(0, 59))

    def doc_offset(self, rng, kind: str) -> int:
        """Rendering offset of a document timestamp (documents come from several systems)."""
        p_utc = 0.3 if kind == "inv" else 0.5
        return 0 if rng.random() < p_utc else self.tz_min

    # --------------------------------------------------------------- history
    def build_company(self):
        r, cfg = self.r, self.cfg
        self.base = r.choice(cfg["base_currencies"])
        pool = [c for c in cfg["foreign_currencies"] if c != self.base]
        lo, hi = cfg["n_foreign_currencies"]
        n_f = min(len(pool), r.randint(lo, hi))
        self.foreign = sorted(r.sample(pool, n_f)) if cfg["foreign_customer_share"] > 0 else []
        self.tz_str = r.choice(cfg["timezones"])
        self.tz_min = offset_minutes(self.tz_str)
        self.tz = timezone(timedelta(minutes=self.tz_min))
        year = r.randint(cfg["years"][0], cfg["years"][1])
        month = r.randint(1, 12)
        self.start_date = date(year, month, 1)
        self.end_date = date(year + (month == 12), month % 12 + 1, 1)
        self.start = self.at_local(self.start_date, 0)
        self.end = self.at_local(self.end_date, 0)
        self.company_name = "%s %s %s" % (r.choice(NAMES_A), r.choice(NAMES_B), r.choice(NAMES_C))
        self.ledger_offset = r.choice([0, self.tz_min])

    def build_fx(self):
        r = self.r
        self.fx_rows = []
        d0 = self.start_date - timedelta(days=60)
        d1 = self.end_date + timedelta(days=14)
        for cur in self.foreign:
            with localcontext() as ctx:
                ctx.prec = 40
                level = USD_VALUE[cur] / USD_VALUE[self.base] * (Decimal(1) + Decimal(r.randint(-300, 300)) / 10000)
            prev = None
            d = d0
            while d <= d1:
                if d.weekday() < 5:
                    with localcontext() as ctx:
                        ctx.prec = 40
                        level = level * (Decimal(1) + Decimal(r.randint(-70, 70)) / 10000)
                    q = _q6(level)
                    if prev is not None and q == prev:
                        q = _q6(q * Decimal("1.0003"))
                    prev = q
                    self.fx_rows.append({"k": self.nk(), "date": d, "src": cur, "tgt": self.base, "rate": q})
                d += timedelta(days=1)
        self.ratebook = RateBook([(x["date"], x["src"], x["tgt"], x["rate"], x["k"]) for x in self.fx_rows])
        self.fx_by_k = {x["k"]: x for x in self.fx_rows}

    def build_entities(self):
        r, cfg = self.r, self.cfg
        used_names = set()

        def name():
            for _ in range(200):
                n = "%s %s %s" % (r.choice(NAMES_A), r.choice(NAMES_B), r.choice(NAMES_C))
                if n not in used_names and n != self.company_name:
                    used_names.add(n)
                    return n
            raise GenerationError("names exhausted")

        n_c = r.randint(*cfg["n_customers"])
        self.customers = []
        for i in range(n_c):
            if self.foreign and i < len(self.foreign):
                cur = self.foreign[i]          # at least one customer per foreign currency
            elif self.foreign and r.random() < cfg["foreign_customer_share"]:
                cur = r.choice(self.foreign)
            else:
                cur = self.base
            k = self.nk()
            self.entities[k] = {"k": k, "kind": "customer", "name": name(), "currency": cur}
            self.customers.append(k)
        self.vendors = []
        for _ in range(r.randint(*cfg["n_vendors"])):
            k = self.nk()
            self.entities[k] = {"k": k, "kind": "vendor", "name": name(), "currency": self.base}
            self.vendors.append(k)

    def new_po(self, rng) -> str:
        self._po_counter += 1
        return "PO-%d%04d" % (rng.randint(10, 99), self._po_counter)

    def make_lines(self, rng, currency: str, issued: datetime, n_lines=None):
        exp = CURRENCY_EXPONENTS[currency]
        lines = []
        skus = rng.sample(CATALOG, n_lines or rng.randint(1, 4))
        period = self.ldate(issued).strftime("%Y-%m")
        for i, (sku, desc, lo, hi) in enumerate(skus):
            usd = Decimal(rng.randint(lo * 100, hi * 100)) / 100
            with localcontext() as ctx:
                ctx.prec = 40
                unit = int((usd / USD_VALUE[currency] * (Decimal(10) ** exp)).to_integral_value())
            unit = max(1, unit)
            qty = rng.randint(1, 12)
            lines.append({"line_no": i + 1, "sku": sku, "description": desc, "quantity": qty,
                          "unit_price_minor": unit, "amount_minor": unit * qty, "service_period": period})
        return lines

    def make_invoice(self, rng, entity_k: int, issued: datetime, lines=None, po_ref=None, kind_tag="normal"):
        ent = self.entities[entity_k]
        cur = ent["currency"]
        if lines is None:
            lines = self.make_lines(rng, cur, issued)
        k = self.nk()
        terms = rng.choice([14, 30, 30, 45])
        inv = {"k": k, "doc_k": self.nk(), "entity": entity_k, "currency": cur, "issued": issued,
               "due": issued + timedelta(days=terms), "lines": copy.deepcopy(lines),
               "po_ref": po_ref or self.new_po(rng), "total": sum(l["amount_minor"] for l in lines),
               "off_issued": self.doc_offset(rng, "inv"), "tag": kind_tag}
        self.invoices[k] = inv
        return inv

    def make_payment(self, rng, entity_k: int, ts: datetime, amount: int, refs, kind="receipt", refund_of=None,
                     tag="normal"):
        ent = self.entities[entity_k]
        k = self.nk()
        pay = {"k": k, "doc_k": self.nk(), "entity": entity_k, "currency": ent["currency"], "ts": ts,
               "amount": int(amount), "refs": None if refs is None else [(a, int(b)) for a, b in refs],
               "kind": kind, "refund_of": refund_of, "off": self.doc_offset(rng, "pay"),
               "bank_ref": "BR%010d" % rng.randint(0, 10**10 - 1), "tag": tag}
        self.payments[k] = pay
        return pay

    def build_invoices(self):
        r, cfg, dc = self.r, self.cfg, self.cfg["distractors"]
        n_main = r.randint(*cfg["n_invoices"])
        n_prior = max(1, int(Decimal(n_main) * Decimal(str(cfg["prior_invoice_share"]))))
        last = self.end_date - timedelta(days=1)
        for _ in range(n_prior):
            ts = self.random_instant(r, self.start_date - timedelta(days=40), self.start_date - timedelta(days=1))
            self.make_invoice(r, r.choice(self.customers), ts)
        base_main = n_main
        # recurring same-amount invoices (legit: different PO / order)
        for _ in range(r.randint(*dc["recurring_same_amount"])):
            e = r.choice(self.customers)
            t1 = self.random_instant(r, self.start_date, last)
            inv1 = self.make_invoice(r, e, t1, kind_tag="recurring")
            t2 = self.random_instant(r, self.start_date, last)
            self.make_invoice(r, e, t2, lines=inv1["lines"], kind_tag="recurring")
            self.distractor_log.append("recurring_same_amount")
            base_main -= 2
        # partial deliveries against one PO (legit: same PO, different lines)
        for _ in range(r.randint(*dc["partial_deliveries"])):
            e = r.choice(self.customers)
            t1 = self.random_instant(r, self.start_date, last)
            inv1 = self.make_invoice(r, e, t1, kind_tag="partial_delivery")
            lines2 = copy.deepcopy(inv1["lines"])
            for ln in lines2:
                ln["quantity"] = ln["quantity"] + r.randint(1, 5)
                ln["amount_minor"] = ln["quantity"] * ln["unit_price_minor"]
            t2 = self.random_instant(r, self.start_date, last)
            self.make_invoice(r, e, t2, lines=lines2, po_ref=inv1["po_ref"], kind_tag="partial_delivery")
            self.distractor_log.append("partial_deliveries")
            base_main -= 2
        for _ in range(max(5, base_main)):
            ts = self.random_instant(r, self.start_date, last)
            self.make_invoice(r, r.choice(self.customers), ts)
        # documents dated just outside the period (correctly NOT booked)
        for _ in range(r.randint(*cfg["n_next_period_invoices"])):
            ts = self.random_instant(r, self.end_date, self.end_date + timedelta(days=4))
            self.make_invoice(r, r.choice(self.customers), ts)
        # time-zone boundary look-alikes
        if self.tz_min != 0:
            for _ in range(r.randint(*dc["tz_boundary_docs"])):
                e = r.choice(self.customers)
                if self.tz_min > 0:
                    inside = self.at_local(self.start_date, r.randint(5, self.tz_min - 5), r.randint(0, 59))
                    outside = self.at_local(self.end_date, r.randint(5, self.tz_min - 5), r.randint(0, 59))
                else:
                    inside = self.at_local(last, r.randint(1440 + self.tz_min + 5, 1435), r.randint(0, 59))
                    outside = self.at_local(self.start_date - timedelta(days=1),
                                            r.randint(1440 + self.tz_min + 5, 1435), r.randint(0, 59))
                # inside: in period although its UTC date is not; outside: the reverse (must stay unposted)
                self.make_invoice(r, e, inside, kind_tag="tz_boundary_inside")
                self.make_invoice(r, r.choice(self.customers), outside, kind_tag="tz_boundary_outside")
                self.distractor_log.append("tz_boundary_docs")

    def build_payments(self):
        r, dc = self.r, self.cfg["distractors"]
        last_ok = self.end - timedelta(hours=2)
        payable = sorted([inv for inv in self.invoices.values() if inv["issued"] < self.end - timedelta(days=1)],
                         key=lambda x: x["k"])
        r.shuffle(payable)
        self.outstanding = {inv["k"]: inv["total"] for inv in self.invoices.values()}

        def pay_time(lo: datetime):
            lo = max(lo, self.start)
            if lo >= last_ok:
                return None
            for _ in range(20):
                ts = self.random_instant(r, self.ldate(lo), self.ldate(last_ok))
                if lo <= ts < last_ok:
                    return ts
            return None

        def take(pred):
            for inv in payable:
                if self.outstanding[inv["k"]] == inv["total"] and pred(inv):
                    return inv
            return None

        # multi-invoice payments
        for _ in range(r.randint(*dc["multi_invoice_payments"])):
            first = take(lambda i: True)
            if not first:
                break
            group = [first] + [i for i in payable if i["entity"] == first["entity"] and i["k"] != first["k"]
                               and self.outstanding[i["k"]] == i["total"]][: r.randint(1, 2)]
            if len(group) < 2:
                continue
            ts = pay_time(max(g["issued"] for g in group) + timedelta(hours=6))
            if ts is None:
                continue
            refs = [(g["k"], g["total"]) for g in group]
            self.make_payment(r, first["entity"], ts, sum(a for _, a in refs), refs, tag="multi_invoice")
            for g in group:
                self.outstanding[g["k"]] = 0
            self.distractor_log.append("multi_invoice_payments")
        # split payments: one invoice settled by two payments
        for _ in range(r.randint(*dc["split_payments"])):
            inv = take(lambda i: True)
            if not inv:
                break
            t1 = pay_time(inv["issued"] + timedelta(hours=6))
            if t1 is None:
                continue
            t2 = pay_time(t1 + timedelta(hours=3))
            if t2 is None:
                continue
            a1 = max(1, inv["total"] * r.randint(30, 70) // 100)
            self.make_payment(r, inv["entity"], t1, a1, [(inv["k"], a1)], tag="split")
            self.make_payment(r, inv["entity"], t2, inv["total"] - a1, [(inv["k"], inv["total"] - a1)], tag="split")
            self.outstanding[inv["k"]] = 0
            self.distractor_log.append("split_payments")
        # short payments (legit partial settlement; invoice stays open)
        for _ in range(r.randint(*dc["short_payments"])):
            inv = take(lambda i: True)
            if not inv:
                break
            ts = pay_time(inv["issued"] + timedelta(hours=6))
            if ts is None:
                continue
            a = max(1, inv["total"] * r.randint(40, 90) // 100)
            self.make_payment(r, inv["entity"], ts, a, [(inv["k"], a)], tag="short")
            self.outstanding[inv["k"]] -= a
            self.distractor_log.append("short_payments")
        # overpayments (remainder legitimately unapplied)
        unapplied_pays = []
        for _ in range(r.randint(*dc["overpayments"])):
            inv = take(lambda i: True)
            if not inv:
                break
            ts = pay_time(inv["issued"] + timedelta(hours=6))
            if ts is None:
                continue
            extra = max(1, inv["total"] * r.randint(1, 8) // 100)
            p = self.make_payment(r, inv["entity"], ts, inv["total"] + extra, [(inv["k"], inv["total"])], tag="overpayment")
            self.outstanding[inv["k"]] = 0
            unapplied_pays.append((p, extra))
            self.distractor_log.append("overpayments")
        # advances (no invoice reference -> explicit null)
        for _ in range(r.randint(*dc["advances"])):
            e = r.choice(self.customers)
            cur = self.entities[e]["currency"]
            ts = pay_time(self.start + timedelta(hours=1))
            if ts is None:
                continue
            amt = r.randint(200, 20000) * (10 ** CURRENCY_EXPONENTS[cur])
            with localcontext() as ctx:
                ctx.prec = 40
                amt = max(1, int(Decimal(amt) / USD_VALUE[cur] / 10))
            p = self.make_payment(r, e, ts, amt, None, tag="advance")
            unapplied_pays.append((p, amt))
            self.distractor_log.append("advances")
        # refunds of unapplied money
        for _ in range(r.randint(*dc["refunds"])):
            if not unapplied_pays:
                break
            p, avail = unapplied_pays.pop(r.randrange(len(unapplied_pays)))
            ts = pay_time(p["ts"] + timedelta(hours=12))
            if ts is None:
                continue
            amt = max(1, avail * r.randint(50, 100) // 100)
            self.make_payment(r, p["entity"], ts, amt, None, kind="refund", refund_of=p["k"], tag="refund")
            self.distractor_log.append("refunds")
        # ordinary full settlements
        for inv in payable:
            if self.outstanding[inv["k"]] != inv["total"] or r.random() > 0.7:
                continue
            ts = pay_time(inv["issued"] + timedelta(hours=6))
            if ts is None:
                continue
            self.make_payment(r, inv["entity"], ts, inv["total"], [(inv["k"], inv["total"])])
            self.outstanding[inv["k"]] = 0
        # payments received after period end (correctly NOT booked)
        for _ in range(r.randint(*dc["next_period_payments"])):
            inv = take(lambda i: True)
            if not inv:
                break
            ts = self.random_instant(r, self.end_date, self.end_date + timedelta(days=3))
            self.make_payment(r, inv["entity"], ts, inv["total"], [(inv["k"], inv["total"])], tag="next_period")
            self.outstanding[inv["k"]] = 0

    # --------------------------------------------------------------- posting
    def spec_invoice(self, inv, rate=None, ts=None):
        rate = self.rate(inv["currency"], self.ldate(inv["issued"])) if rate is None else rate
        amt = self.conv(inv["total"], inv["currency"], rate)
        ts = inv["issued"] if ts is None else ts
        return [
            {"ts": ts, "account": ACC_AR, "dr": amt, "cr": 0, "entity": inv["entity"], "inv": inv["k"], "pay": None},
            {"ts": ts, "account": ACC_REVENUE, "dr": 0, "cr": amt, "entity": inv["entity"], "inv": inv["k"], "pay": None},
        ]

    def spec_payment(self, pay, rate=None, ts=None, ar=None):
        cur = pay["currency"]
        rate = self.rate(cur, self.ldate(pay["ts"])) if rate is None else rate
        ts = pay["ts"] if ts is None else ts
        e, P = pay["entity"], pay["k"]
        amt = self.conv(pay["amount"], cur, rate)
        if pay["kind"] == "refund":
            return [
                {"ts": ts, "account": ACC_UNAPPLIED, "dr": amt, "cr": 0, "entity": e, "inv": None, "pay": P},
                {"ts": ts, "account": ACC_CASH, "dr": 0, "cr": amt, "entity": e, "inv": None, "pay": P},
            ]
        ar = (pay["refs"] or []) if ar is None else ar
        rows = [{"ts": ts, "account": ACC_CASH, "dr": amt, "cr": 0, "entity": e, "inv": None, "pay": P}]
        credited_doc = 0
        credits = 0
        for inv_k, applied in ar:
            if applied <= 0:
                continue
            inv = self.invoices[inv_k]
            hr = self.rate(inv["currency"], self.ldate(inv["issued"]))
            c = self.conv(applied, cur, hr)
            rows.append({"ts": ts, "account": ACC_AR, "dr": 0, "cr": c, "entity": e, "inv": inv_k, "pay": P})
            credited_doc += applied
            credits += c
        rem = pay["amount"] - credited_doc
        if rem > 0:
            c = self.conv(rem, cur, rate)
            rows.append({"ts": ts, "account": ACC_UNAPPLIED, "dr": 0, "cr": c, "entity": e, "inv": None, "pay": P})
            credits += c
        diff = amt - credits
        if diff > 0:
            rows.append({"ts": ts, "account": ACC_FX, "dr": 0, "cr": diff, "entity": e, "inv": None, "pay": P})
        elif diff < 0:
            rows.append({"ts": ts, "account": ACC_FX, "dr": -diff, "cr": 0, "entity": e, "inv": None, "pay": P})
        return rows

    def commit(self, specs, doc_key=None, log=None) -> int:
        self._jid += 1
        jid = self._jid
        self.journals[jid] = []
        for s in specs:
            self._lid += 1
            row = dict(s, k=self._lid, j=jid)
            self.rows[row["k"]] = row
            self.all_rows[row["k"]] = dict(row)
            self.journals[jid].append(row["k"])
            if log is not None:
                log.append({"op": "insert_row", "row": dict(row)})
        if doc_key is not None:
            self.doc_journal.setdefault(doc_key, []).append(jid)
        return jid

    def replace_journal(self, jid, specs, log):
        """Rewrite a journal; rows are matched by role so unchanged rows keep their identity."""
        def role(x):
            return (x["account"], x["inv"], x["pay"])
        old = {role(self.rows[l]): l for l in self.journals[jid]}
        new_lids = []
        for s in specs:
            key = role(s)
            if key in old:
                lid = old.pop(key)
                before = self.rows[lid]
                after = dict(s, k=lid, j=jid)
                if any(before[f] != after[f] for f in ("ts", "dr", "cr", "entity")):
                    log.append({"op": "update_row", "before": dict(before), "after": dict(after)})
                self.rows[lid] = after
                self.all_rows[lid] = dict(after)
                new_lids.append(lid)
            else:
                self._lid += 1
                row = dict(s, k=self._lid, j=jid)
                self.rows[row["k"]] = row
                self.all_rows[row["k"]] = dict(row)
                new_lids.append(row["k"])
                log.append({"op": "insert_row", "row": dict(row)})
        for key, lid in sorted(old.items(), key=lambda kv: kv[1]):
            log.append({"op": "delete_row", "row": dict(self.rows[lid])})
            del self.rows[lid]
        self.journals[jid] = new_lids

    def post_all(self):
        for inv in sorted(self.invoices.values(), key=lambda x: x["k"]):
            if self.in_period(inv["issued"]):
                self.commit(self.spec_invoice(inv), ("inv", inv["k"]))
        for pay in sorted(self.payments.values(), key=lambda x: x["k"]):
            if self.in_period(pay["ts"]):
                self.commit(self.spec_payment(pay), ("pay", pay["k"]))
        r = self.r
        doc_rows = len(self.rows)
        target = r.randint(*self.cfg["ledger_rows"])
        n_exp = max(5, (target - doc_rows) // 2)
        for _ in range(n_exp):
            v = r.choice(self.vendors)
            ts = self.random_instant(r, self.start_date, self.end_date - timedelta(days=1), near_share=0.0)
            acc = r.choice(EXPENSE_ACCOUNTS)
            with localcontext() as ctx:
                ctx.prec = 40
                amt = int(Decimal(r.randint(5000, 2000000)) / USD_VALUE[self.base]
                          * (Decimal(10) ** CURRENCY_EXPONENTS[self.base]) / 100)
            amt = max(1, amt)
            self.commit([
                {"ts": ts, "account": acc, "dr": amt, "cr": 0, "entity": v, "inv": None, "pay": None},
                {"ts": ts, "account": ACC_CASH, "dr": 0, "cr": amt, "entity": v, "inv": None, "pay": None},
            ])

    # ------------------------------------------------------------- injection
    _STATE = ("invoices", "payments", "rows", "all_rows", "journals", "doc_journal", "booked_pay_rate", "used",
              "issues", "outstanding", "_k", "_lid", "_jid", "_po_counter")

    def _snapshot(self):
        return {a: copy.deepcopy(getattr(self, a)) for a in self._STATE}

    def _restore(self, snap):
        for a, v in snap.items():
            setattr(self, a, copy.deepcopy(v))

    def jrows(self, doc_key):
        out = []
        for jid in self.doc_journal.get(doc_key, []):
            out.extend(self.rows[l] for l in self.journals[jid] if l in self.rows)
        return out

    def find_row(self, doc_key, account, inv=None, pay=None, jid=None):
        jids = [jid] if jid else self.doc_journal.get(doc_key, [])
        for j in jids:
            for l in self.journals[j]:
                x = self.rows.get(l)
                if x and x["account"] == account and (inv is None or x["inv"] == inv) and (pay is None or x["pay"] == pay):
                    return x["k"]
        return None

    def booked_in_period_invoices(self):
        return [inv for inv in sorted(self.invoices.values(), key=lambda x: x["k"])
                if self.in_period(inv["issued"]) and ("inv", inv["k"]) in self.doc_journal
                and ("inv", inv["k"]) not in self.used and len(self.doc_journal[("inv", inv["k"])]) == 1]

    def booked_receipts(self):
        return [p for p in sorted(self.payments.values(), key=lambda x: x["k"])
                if p["kind"] == "receipt" and self.in_period(p["ts"]) and ("pay", p["k"]) in self.doc_journal
                and ("pay", p["k"]) not in self.used]

    def min_diff(self) -> int:
        return max(20, 10 ** CURRENCY_EXPONENTS[self.base])

    def add_issue(self, itype, variant, affected, amount, evidence_sets, setup, changes, notes, scenario=None):
        exp_tol = amount_tolerance(amount, AMOUNT_TOL_ABS_MINOR, AMOUNT_TOL_REL)
        if amount < max(self.min_diff(), 5 * exp_tol):
            raise GenerationError("discrepancy too small")
        self.issues.append({"type": itype, "variant": variant, "affected": affected, "amount": int(amount),
                            "evidence_sets": evidence_sets, "setup": setup, "changes": changes,
                            "notes": notes, "scenario": scenario})

    # -- duplicate invoice
    def inj_duplicate(self, variant, scenario=None):
        ri = self.ri
        cands = self.booked_in_period_invoices()
        ri.shuffle(cands)
        for inv in cands:
            setup, changes = [], []
            if variant == "duplicate_document":
                gap = timedelta(minutes=ri.randint(90, 7 * 1440))
                issued = inv["issued"] + gap
                if not self.in_period(issued):
                    continue
                dup = self.make_invoice(ri, inv["entity"], issued, lines=inv["lines"], po_ref=inv["po_ref"],
                                        kind_tag="injected_duplicate")
                changes.append({"op": "insert_document", "table": "invoices", "inv": dup["k"]})
                jid = self.commit(self.spec_invoice(dup), ("inv", dup["k"]), changes)
                ar = self.find_row(None, ACC_AR, jid=jid)
                rev = self.find_row(None, ACC_REVENUE, jid=jid)
                amount = self.rows[ar]["dr"]
                ev = [[("doc_inv", inv["k"]), ("doc_inv", dup["k"]), ("row", ar)],
                      [("doc_inv", inv["k"]), ("doc_inv", dup["k"]), ("row", rev)]]
                self.used.update({("inv", inv["k"]), ("inv", dup["k"])})
                self.add_issue("duplicate_invoice", variant, [("inv", inv["k"]), ("inv", dup["k"])], amount, ev,
                               setup, changes, {"gap_minutes": int(gap.total_seconds() // 60)}, scenario)
                return dup
            else:
                specs = self.spec_invoice(inv)
                jid0 = self.doc_journal[("inv", inv["k"])][0]
                jid = self.commit(specs, ("inv", inv["k"]), changes)
                ar1, ar2 = self.find_row(None, ACC_AR, jid=jid0), self.find_row(None, ACC_AR, jid=jid)
                rv1, rv2 = self.find_row(None, ACC_REVENUE, jid=jid0), self.find_row(None, ACC_REVENUE, jid=jid)
                amount = self.rows[ar2]["dr"]
                ev = [[("doc_inv", inv["k"]), ("row", ar1), ("row", ar2)],
                      [("doc_inv", inv["k"]), ("row", rv1), ("row", rv2)]]
                self.used.add(("inv", inv["k"]))
                self.add_issue("duplicate_invoice", variant, [("inv", inv["k"])], amount, ev, setup, changes, {}, scenario)
                return inv
        return None

    # -- payment allocation
    def alloc_evidence(self, pay, inv_k, extra_rows=()):
        base = [("doc_pay", pay["k"]), ("doc_inv", inv_k)]
        inv = self.invoices[inv_k]
        if inv["currency"] != self.base:
            base.append(("fx", self.rate_key(inv["currency"], self.ldate(inv["issued"]))))
        jid = self.doc_journal[("pay", pay["k"])][0]
        rows = []
        for l in self.journals[jid]:
            x = self.rows[l]
            if x["account"] == ACC_CASH or x["account"] == ACC_UNAPPLIED or (x["account"] == ACC_AR and x["inv"] == inv_k):
                rows.append(l)
        rows.extend(extra_rows)
        return [base + [("row", l)] for l in sorted(set(rows))]

    def apply_allocation_error(self, pay, inv_k, variant, changes, redirect_to=None):
        jid = self.doc_journal[("pay", pay["k"])][0]
        inv = self.invoices[inv_k]
        applied = dict(pay["refs"])[inv_k]
        hr = self.rate(inv["currency"], self.ldate(inv["issued"]))
        expected = self.conv(applied, pay["currency"], hr)
        ar = []
        for k, a in pay["refs"]:
            if k != inv_k:
                ar.append((k, a))
            elif variant == "partial":
                u = applied * self.ri.randint(20, 70) // 100
                ar.append((k, applied - u))
            elif redirect_to is not None:
                ar.append((redirect_to, a))
        rate = self.booked_pay_rate.get(pay["k"])
        self.replace_journal(jid, self.spec_payment(pay, rate=rate, ar=ar), changes)
        booked = sum(self.rows[l]["cr"] for l in self.journals[jid]
                     if self.rows[l]["account"] == ACC_AR and self.rows[l]["inv"] == inv_k)
        return expected - booked

    def inj_allocation(self, variant, scenario=None):
        ri = self.ri
        cands = [p for p in self.booked_receipts() if p["refs"]]
        ri.shuffle(cands)
        for pay in cands:
            refs = [k for k, a in pay["refs"] if ("inv", k) not in self.used and a > 0]
            if not refs:
                continue
            inv_k = ri.choice(refs)
            changes = []
            snap = self._snapshot()
            amount = self.apply_allocation_error(pay, inv_k, variant, changes)
            try:
                self.add_issue("payment_allocation", variant, [("pay", pay["k"]), ("inv", inv_k)], amount,
                               self.alloc_evidence(pay, inv_k), [], changes, {}, scenario)
            except GenerationError:
                self._restore(snap)
                continue
            self.used.update({("pay", pay["k"]), ("inv", inv_k)})
            return pay
        return None

    # -- FX conversion
    def wrong_rate(self, variant, currency, ts, extra_date=None):
        ri = self.ri
        d_loc = self.ldate(ts)
        correct = self.rate(currency, d_loc)
        if variant == "utc_date":
            d_utc = ts.astimezone(timezone.utc).date()
            if d_utc == d_loc:
                return None
            w = self.rate(currency, d_utc)
        elif variant == "wrong_date":
            options = [d_loc + timedelta(days=x) for x in (-7, -5, -3, -2, -1, 1, 2, 3)]
            if extra_date is not None:
                options.append(extra_date)
            d = ri.choice(options)
            if self.ratebook.lookup(currency, self.base, d) is None or d > self.end_date + timedelta(days=13):
                return None
            w = self.rate(currency, d)
        elif variant == "inverted":
            with localcontext() as ctx:
                ctx.prec = 40
                w = _q6(Decimal(1) / correct)
        else:  # digit_typo: two adjacent significant digits transposed
            digits = list(fmt_rate(correct))
            pos = [i for i in range(len(digits) - 1) if digits[i].isdigit() and digits[i + 1].isdigit()
                   and digits[i] != digits[i + 1] and not (i == 0 and digits[i + 1] == "0")]
            if not pos:
                return None
            i = ri.choice(pos)
            digits[i], digits[i + 1] = digits[i + 1], digits[i]
            w = Decimal("".join(digits))
        if w == correct:
            return None
        return w

    def inj_fx(self, variant, scenario=None):
        ri = self.ri
        cands = [("inv", x) for x in self.booked_in_period_invoices() if x["currency"] != self.base]
        cands += [("pay", x) for x in self.booked_receipts() if x["currency"] != self.base]
        ri.shuffle(cands)
        for kind, doc in cands:
            ts = doc["issued"] if kind == "inv" else doc["ts"]
            amount_doc = doc["total"] if kind == "inv" else doc["amount"]
            w = self.wrong_rate(variant, doc["currency"], ts, doc["due"].astimezone(self.tz).date() if kind == "inv" else None)
            if w is None:
                continue
            correct_rate = self.rate(doc["currency"], self.ldate(ts))
            correct = self.conv(amount_doc, doc["currency"], correct_rate)
            booked = self.conv(amount_doc, doc["currency"], w)
            if abs(booked - correct) < self.min_diff() * 5:
                continue
            changes = []
            jid = self.doc_journal[(kind, doc["k"])][0]
            if kind == "inv":
                self.replace_journal(jid, self.spec_invoice(doc, rate=w), changes)
                prim = [self.find_row(None, ACC_AR, jid=jid), self.find_row(None, ACC_REVENUE, jid=jid)]
                dk = ("doc_inv", doc["k"])
            else:
                self.replace_journal(jid, self.spec_payment(doc, rate=w), changes)
                self.booked_pay_rate[doc["k"]] = w
                prim = [self.find_row(None, ACC_CASH, jid=jid)]
                dk = ("doc_pay", doc["k"])
            fxk = ("fx", self.rate_key(doc["currency"], self.ldate(ts)))
            ev = [[dk, ("row", p), fxk] for p in prim]
            self.used.add((kind, doc["k"]))
            self.add_issue("fx_conversion", variant, [(kind, doc["k"])], abs(booked - correct), ev, [], changes,
                           {"correct_rate": fmt_rate(correct_rate), "booked_rate": fmt_rate(w)}, scenario)
            return doc
        return None

    # -- period cut-off
    def inj_cutoff(self, variant, scenario=None):
        ri = self.ri
        if variant == "tz_boundary" and self.tz_min == 0:
            return None
        kind = "inv" if ri.random() < 0.65 else "pay"
        late = ri.random() < 0.5
        if variant == "tz_boundary":
            if self.tz_min > 0:
                late = True
                doc_ts = self.at_local(self.end_date, ri.randint(5, self.tz_min - 5), ri.randint(0, 59))
            else:
                late = False
                doc_ts = self.at_local(self.start_date - timedelta(days=1),
                                       ri.randint(1440 + self.tz_min + 5, 1435), ri.randint(0, 59))
            post_ts = doc_ts
        else:
            if late:
                doc_ts = self.random_instant(ri, self.end_date + timedelta(days=1), self.end_date + timedelta(days=5), 0.0)
                post_ts = self.end - timedelta(minutes=ri.randint(30, 30 * 60), seconds=ri.randint(0, 59))
            else:
                doc_ts = self.random_instant(ri, self.start_date - timedelta(days=6), self.start_date - timedelta(days=2), 0.0)
                post_ts = self.start + timedelta(minutes=ri.randint(30, 30 * 60), seconds=ri.randint(0, 59))
        setup, changes = [], []
        e = ri.choice(self.customers)
        if kind == "inv":
            doc = self.make_invoice(ri, e, doc_ts)
            setup.append({"op": "insert_document", "table": "invoices", "inv": doc["k"]})
            jid = self.commit(self.spec_invoice(doc, ts=post_ts), ("inv", doc["k"]), changes)
            prim = [self.find_row(None, ACC_AR, jid=jid), self.find_row(None, ACC_REVENUE, jid=jid)]
            dk = ("doc_inv", doc["k"])
        else:
            refs = None
            open_inv = [i for i in sorted(self.invoices.values(), key=lambda x: x["k"])
                        if i["entity"] == e and i["issued"] < doc_ts - timedelta(hours=2)
                        and self.outstanding.get(i["k"], 0) == i["total"] and ("inv", i["k"]) not in self.used]
            if open_inv:
                tgt = ri.choice(open_inv)
                refs = [(tgt["k"], tgt["total"])]
                amount = tgt["total"]
                self.outstanding[tgt["k"]] = 0
            else:
                cur = self.entities[e]["currency"]
                with localcontext() as ctx:
                    ctx.prec = 40
                    amount = max(1, int(Decimal(ri.randint(500, 30000)) / USD_VALUE[cur]
                                        * (Decimal(10) ** CURRENCY_EXPONENTS[cur])))
            doc = self.make_payment(ri, e, doc_ts, amount, refs)
            setup.append({"op": "insert_document", "table": "payments", "pay": doc["k"]})
            jid = self.commit(self.spec_payment(doc, ts=post_ts), ("pay", doc["k"]), changes)
            prim = [self.find_row(None, ACC_CASH, jid=jid)]
            dk = ("doc_pay", doc["k"])
        amount = self.rows[prim[0]]["dr"]
        self.used.add((kind, doc["k"]))
        self.add_issue("period_cutoff", variant, [(kind, doc["k"])], amount, [[dk, ("row", p)] for p in prim],
                       setup, changes, {"direction": "late" if late else "early"}, scenario)
        return doc

    # -- interaction scenario 1: partial payment in another currency at the period boundary
    def scen_partial_fx_boundary(self):
        ri = self.ri
        fcust = [c for c in self.customers if self.entities[c]["currency"] != self.base]
        if not fcust or self.tz_min == 0:
            return False
        e = ri.choice(fcust)
        cur = self.entities[e]["currency"]
        # candidate payment instants: period boundary first, then any day boundary in the period
        cands = []
        if self.tz_min > 0:
            cands.append(("period_boundary", self.at_local(self.start_date, ri.randint(5, self.tz_min - 5),
                                                           ri.randint(0, 59))))
        else:
            cands.append(("period_boundary", self.at_local(self.end_date - timedelta(days=1),
                                                           ri.randint(1440 + self.tz_min + 5, 1435),
                                                           ri.randint(0, 59))))
        days = list(range(3, (self.end_date - self.start_date).days - 1))
        ri.shuffle(days)
        for dd in days[:10]:
            d = self.start_date + timedelta(days=dd)
            m = ri.randint(5, self.tz_min - 5) if self.tz_min > 0 else ri.randint(1440 + self.tz_min + 5, 1435)
            cands.append(("day_boundary", self.at_local(d, m, ri.randint(0, 59))))
        for where, ts in cands:
            d_loc, d_utc = self.ldate(ts), ts.astimezone(timezone.utc).date()
            if self.rate(cur, d_loc) == self.rate(cur, d_utc):
                continue
            snap = self._snapshot()
            issued = self.random_instant(ri, self.start_date - timedelta(days=25), self.start_date - timedelta(days=3), 0.0)
            inv = self.make_invoice(ri, e, issued)
            pay = self.make_payment(ri, e, ts, inv["total"], [(inv["k"], inv["total"])])
            self.outstanding[inv["k"]] = 0
            setup = [{"op": "insert_document", "table": "invoices", "inv": inv["k"]},
                     {"op": "insert_document", "table": "payments", "pay": pay["k"]}]
            jid = self.commit(self.spec_payment(pay), ("pay", pay["k"]), setup)
            w = self.rate(cur, d_utc)
            correct_cash = self.conv(pay["amount"], cur, self.rate(cur, d_loc))
            booked_cash = self.conv(pay["amount"], cur, w)
            if abs(booked_cash - correct_cash) < self.min_diff() * 5:
                self._restore(snap)
                continue
            fx_changes, al_changes = [], []
            self.replace_journal(jid, self.spec_payment(pay, rate=w), fx_changes)
            self.booked_pay_rate[pay["k"]] = w
            amount_alloc = self.apply_allocation_error(pay, inv["k"], "partial", al_changes)
            fxk = ("fx", self.rate_key(cur, d_loc))
            cash = self.find_row(None, ACC_CASH, jid=jid)
            notes = {"where": where, "correct_rate": fmt_rate(self.rate(cur, d_loc)), "booked_rate": fmt_rate(w)}
            n0 = len(self.issues)
            try:
                self.add_issue("fx_conversion", "utc_date", [("pay", pay["k"])], abs(booked_cash - correct_cash),
                               [[("doc_pay", pay["k"]), ("row", cash), fxk]], setup, fx_changes, notes,
                               "partial_fx_payment_at_boundary")
                self.add_issue("payment_allocation", "partial", [("pay", pay["k"]), ("inv", inv["k"])], amount_alloc,
                               self.alloc_evidence(pay, inv["k"]), [], al_changes, {"where": where},
                               "partial_fx_payment_at_boundary")
            except GenerationError:
                self._restore(snap)
                continue
            self.used.update({("pay", pay["k"]), ("inv", inv["k"])})
            return True
        return False

    # -- interaction scenario 2: duplicate invoice + payment applied to the duplicate
    def scen_duplicate_misapplied(self):
        ri = self.ri
        for _ in range(10):
            e = ri.choice(self.customers)
            i_ts = self.random_instant(ri, self.start_date, self.start_date + timedelta(days=12), 0.0)
            d_ts = i_ts + timedelta(minutes=ri.randint(1440, 5 * 1440))
            p_ts = d_ts + timedelta(minutes=ri.randint(600, 8 * 1440))
            if not (self.in_period(d_ts) and self.in_period(p_ts)):
                continue
            setup, dchanges, achanges = [], [], []
            inv = self.make_invoice(ri, e, i_ts)
            setup.append({"op": "insert_document", "table": "invoices", "inv": inv["k"]})
            self.commit(self.spec_invoice(inv), ("inv", inv["k"]), setup)
            dup = self.make_invoice(ri, e, d_ts, lines=inv["lines"], po_ref=inv["po_ref"], kind_tag="injected_duplicate")
            dchanges.append({"op": "insert_document", "table": "invoices", "inv": dup["k"]})
            djid = self.commit(self.spec_invoice(dup), ("inv", dup["k"]), dchanges)
            pay = self.make_payment(ri, e, p_ts, inv["total"], [(inv["k"], inv["total"])])
            self.outstanding[inv["k"]] = 0
            setup.append({"op": "insert_document", "table": "payments", "pay": pay["k"]})
            self.commit(self.spec_payment(pay), ("pay", pay["k"]), setup)
            amount_alloc = self.apply_allocation_error(pay, inv["k"], "missing", achanges, redirect_to=dup["k"])
            ar_d = self.find_row(None, ACC_AR, jid=djid)
            rv_d = self.find_row(None, ACC_REVENUE, jid=djid)
            redirected = self.find_row(("pay", pay["k"]), ACC_AR, inv=dup["k"])
            self.add_issue("duplicate_invoice", "duplicate_document", [("inv", inv["k"]), ("inv", dup["k"])],
                           self.rows[ar_d]["dr"],
                           [[("doc_inv", inv["k"]), ("doc_inv", dup["k"]), ("row", ar_d)],
                            [("doc_inv", inv["k"]), ("doc_inv", dup["k"]), ("row", rv_d)]],
                           setup, dchanges, {}, "duplicate_with_misapplied_payment")
            self.add_issue("payment_allocation", "missing", [("pay", pay["k"]), ("inv", inv["k"])], amount_alloc,
                           self.alloc_evidence(pay, inv["k"], extra_rows=[redirected]), [], achanges,
                           {"redirected_to_duplicate": True}, "duplicate_with_misapplied_payment")
            self.used.update({("inv", inv["k"]), ("inv", dup["k"]), ("pay", pay["k"])})
            return True
        return False

    def plan_and_inject(self):
        ri, cfg = self.ri, self.cfg
        clean = ri.random() < float(cfg["clean_probability"])
        plan = []
        for s in sorted(cfg["scenarios"]):
            n = 0 if clean else ri.randint(*cfg["scenarios"][s]["count"])
            plan += [("scenario", s)] * n
        for t in ISSUE_TYPES:
            spec = cfg["errors"].get(t, {"count": [0, 0]})
            n = 0 if clean else ri.randint(*spec["count"])
            for _ in range(n):
                plan.append(("type", t))
        # respect max/min issue counts (scenarios produce two issues each)
        def size(p):
            return sum(2 if kind == "scenario" else 1 for kind, _ in p)
        while size(plan) > cfg["max_issues"]:
            plan.pop(ri.randrange(len(plan)))
        while not clean and size(plan) < cfg["min_issues"]:
            enabled = [t for t in ISSUE_TYPES if cfg["errors"].get(t, {"count": [0, 0]})["count"][1] > 0]
            if not enabled:
                break
            plan.append(("type", ri.choice(enabled)))
        self.clean = clean
        self.plan = list(plan)
        order = {"scenario": 0, "period_cutoff": 1, "duplicate_invoice": 2, "payment_allocation": 3, "fx_conversion": 4}
        plan.sort(key=lambda p: order["scenario" if p[0] == "scenario" else p[1]])
        for kind, name in plan:
            if kind == "scenario":
                snap = self._snapshot()
                try:
                    ok = (self.scen_partial_fx_boundary() if name == "partial_fx_payment_at_boundary"
                          else self.scen_duplicate_misapplied())
                except GenerationError:
                    ok = False
                if not ok:
                    self._restore(snap)
                    self.skipped.append({"scenario": name, "reason": "infeasible"})
                continue
            variants = list(cfg["errors"][name]["variants"])
            ri.shuffle(variants)
            done = False
            for v in variants:
                fn = {"duplicate_invoice": self.inj_duplicate, "payment_allocation": self.inj_allocation,
                      "fx_conversion": self.inj_fx, "period_cutoff": self.inj_cutoff}[name]
                snap = self._snapshot()
                try:
                    if fn(v) is not None:
                        done = True
                        break
                except GenerationError:
                    pass
                self._restore(snap)
            if not done:
                self.skipped.append({"type": name, "reason": "no feasible target"})

    # ---------------------------------------------------------------- render
    def assign_ids(self):
        taken = set()
        rid = self.rid

        def new(prefix, n=8):
            while True:
                s = prefix + "".join(rid.choice(ID_ALPHABET) for _ in range(n))
                if s not in taken:
                    taken.add(s)
                    return s

        self.ent_id = {}
        for k in sorted(self.entities):
            self.ent_id[k] = new("CUS-" if self.entities[k]["kind"] == "customer" else "VEN-", 6)
        self.inv_id, self.pay_id, self.doc_id = {}, {}, {}
        for k in sorted(self.invoices):
            self.inv_id[k] = new("INV-")
            self.doc_id[("inv", k)] = new("DOC-")
        for k in sorted(self.payments):
            self.pay_id[k] = new("PAY-")
            self.doc_id[("pay", k)] = new("DOC-")
        self.row_id = {k: new("LR-", 9) for k in sorted(self.all_rows)}
        self.jr_id = {}
        for k in sorted(self.all_rows):
            j = self.all_rows[k]["j"]
            if j not in self.jr_id:
                self.jr_id[j] = new("JE-")
        self.fx_id = {x["k"]: new("FX-") for x in self.fx_rows}

    def render_row(self, x):
        return {
            "row_id": self.row_id[x["k"]],
            "timestamp": fmt_ts(x["ts"], self.ledger_offset),
            "account": x["account"],
            "debit_minor": int(x["dr"]),
            "credit_minor": int(x["cr"]),
            "currency": self.base,
            "entity_id": self.ent_id[x["entity"]] if x["entity"] is not None else None,
            "invoice_id": self.inv_id[x["inv"]] if x["inv"] is not None else None,
            "payment_id": self.pay_id[x["pay"]] if x["pay"] is not None else None,
            "journal_id": self.jr_id[x["j"]],
        }

    def render_invoice(self, inv):
        return {
            "doc_id": self.doc_id[("inv", inv["k"])],
            "invoice_id": self.inv_id[inv["k"]],
            "entity_id": self.ent_id[inv["entity"]],
            "customer_name": self.entities[inv["entity"]]["name"],
            "issued_at": fmt_ts(inv["issued"], inv["off_issued"]),
            "due_at": fmt_ts(inv["due"], inv["off_issued"]),
            "currency": inv["currency"],
            "total_minor": int(inv["total"]),
            "po_ref": inv["po_ref"],
            "line_items": copy.deepcopy(inv["lines"]),
        }

    def render_payment(self, p):
        return {
            "doc_id": self.doc_id[("pay", p["k"])],
            "payment_id": self.pay_id[p["k"]],
            "entity_id": self.ent_id[p["entity"]],
            "kind": p["kind"],
            "timestamp": fmt_ts(p["ts"], p["off"]),
            "currency": p["currency"],
            "amount_minor": int(p["amount"]),
            "invoice_refs": None if p["refs"] is None else
            [{"invoice_id": self.inv_id[k], "applied_minor": int(a)} for k, a in p["refs"]],
            "refund_of": self.pay_id[p["refund_of"]] if p["refund_of"] is not None else None,
            "bank_reference": p["bank_ref"],
        }

    def ref_id(self, ref):
        kind, k = ref
        if kind == "inv":
            return self.inv_id[k]
        if kind == "pay":
            return self.pay_id[k]
        if kind == "doc_inv":
            return self.doc_id[("inv", k)]
        if kind == "doc_pay":
            return self.doc_id[("pay", k)]
        if kind == "row":
            return self.row_id[k]
        if kind == "fx":
            return self.fx_id[k]
        raise KeyError(kind)

    def render_op(self, op):
        o = op["op"]
        if o == "insert_document":
            if op["table"] == "invoices":
                return {"op": o, "table": "invoices", "doc_id": self.doc_id[("inv", op["inv"])],
                        "invoice_id": self.inv_id[op["inv"]]}
            return {"op": o, "table": "payments", "doc_id": self.doc_id[("pay", op["pay"])],
                    "payment_id": self.pay_id[op["pay"]]}
        if o in ("insert_row", "delete_row"):
            return {"op": o, "row": self.render_row(op["row"])}
        return {"op": o, "before": self.render_row(op["before"]), "after": self.render_row(op["after"])}

    def order(self, items, key):
        items = sorted(items, key=key)
        if self.cfg["shuffle_records"]:
            self.rord.shuffle(items)
        return items

    def render(self):
        self.assign_ids()
        ledger = [self.render_row(x) for x in self.rows.values()]
        acc_rank = {ACC_AR: 0, ACC_CASH: 0, ACC_UNAPPLIED: 1, ACC_REVENUE: 2, ACC_FX: 3}
        ledger_k = {self.row_id[x["k"]]: x for x in self.rows.values()}

        def lkey(r):
            x = ledger_k[r["row_id"]]
            return (x["ts"], r["journal_id"], 0 if x["dr"] > 0 else 1, acc_rank.get(x["account"], 0), r["row_id"])
        ledger = self.order(ledger, lkey)
        inv_k = {self.inv_id[k]: v for k, v in self.invoices.items()}
        invoices = self.order([self.render_invoice(v) for v in self.invoices.values()],
                              lambda d: (inv_k[d["invoice_id"]]["issued"], d["invoice_id"]))
        pay_k = {self.pay_id[k]: v for k, v in self.payments.items()}
        payments = self.order([self.render_payment(v) for v in self.payments.values()],
                              lambda d: (pay_k[d["payment_id"]]["ts"], d["payment_id"]))
        fx = self.order([{"fx_id": self.fx_id[x["k"]], "date": x["date"].isoformat(), "source_currency": x["src"],
                          "target_currency": x["tgt"], "rate": fmt_rate(x["rate"])} for x in self.fx_rows],
                        lambda d: (d["date"], d["source_currency"], d["fx_id"]))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "env_version": ENV_VERSION,
            "config_version": self.cfg["config_version"],
            "config_name": self.cfg["name"],
            "config_sha256": self.full_hash,
            "episode_id": self.episode_id,
            "company_name": self.company_name,
            "base_currency": self.base,
            "currency_exponents": {c: CURRENCY_EXPONENTS[c] for c in sorted(set([self.base] + self.foreign))},
            "period_start": fmt_ts(self.start, self.tz_min),
            "period_end": fmt_ts(self.end, self.tz_min),
            "period_end_exclusive": True,
            "timezone": self.tz_str,
            "counts": {"ledger": len(ledger), "invoices": len(invoices), "payments": len(payments),
                       "fx": len(fx), "documents": len(invoices) + len(payments)},
            "rounding": {"mode": "ROUND_HALF_UP", "unit": "base currency minor unit", "scope": "per posting line"},
            "issue_types": list(ISSUE_TYPES),
            "files": ["manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv"],
        }
        policy = build_policy(self.base, self.tz_str, manifest["period_start"], manifest["period_end"])
        truth_issues = []
        provenance_issues = []
        for i, iss in enumerate(self.issues):
            iid = "ISS-%02d" % (i + 1)
            truth_issues.append({
                "issue_id": iid,
                "type": iss["type"],
                "variant": iss["variant"],
                "scenario": iss["scenario"],
                "affected_ids": sorted(self.ref_id(a) for a in iss["affected"]),
                "expected_amount_minor": iss["amount"],
                "currency": self.base,
                "evidence_sets": [sorted(self.ref_id(e) for e in s) for s in iss["evidence_sets"]],
            })
            provenance_issues.append({
                "issue_id": iid, "type": iss["type"], "variant": iss["variant"], "scenario": iss["scenario"],
                "notes": iss["notes"],
                "setup": [self.render_op(o) for o in iss["setup"]],
                "changes": [self.render_op(o) for o in iss["changes"]],
            })
        truth = {"schema_version": SCHEMA_VERSION, "episode_id": self.episode_id, "base_currency": self.base,
                 "clean": len(truth_issues) == 0, "issues": truth_issues}
        provenance = {
            "schema_version": SCHEMA_VERSION, "env_version": ENV_VERSION, "episode_id": self.episode_id,
            "seed": str(self.seed), "seed_secret_used": self.keyed, "config": self.cfg,
            "config_sha256": self.full_hash,
            "clean_draw": self.clean, "plan": [list(p) for p in self.plan], "skipped": self.skipped,
            "distractors": sorted(set(self.distractor_log)),
            "stages": ["history", "posting", "injection", "render"],
            "issues": provenance_issues,
        }
        public = {"manifest": manifest, "policy": policy, "ledger": ledger, "invoices": invoices,
                  "payments": payments, "fx": fx}
        return public, {"truth": truth, "provenance": provenance}

    def run(self):
        self.build_company()
        self.build_fx()
        self.build_entities()
        self.build_invoices()
        self.build_payments()
        self.post_all()
        self.plan_and_inject()
        return self.render()


def build_policy(base, tz, start, end) -> dict:
    return {
        "policy_version": POLICY_VERSION,
        "base_currency": base,
        "timezone": tz,
        "ledger": {
            "currency": "All ledger rows are kept in the base currency (minor units).",
            "accounts": ACCOUNTS,
            "nulls": "Missing references are explicit null; never infer links that are not recorded.",
            "timestamp": "Ledger timestamp of a document posting equals the document timestamp (same instant).",
        },
        "period": {
            "start_inclusive": start, "end_exclusive": end,
            "basis": "An invoice belongs to the period of its issued_at instant, a payment to the period of its "
                     "timestamp, both judged in the company timezone. Only in-period documents may be posted.",
        },
        "fx": {
            "rate_meaning": "1 unit of source_currency = rate units of target_currency (major units)",
            "rate_date": "Local calendar date (company timezone) of the document timestamp; if no rate exists for "
                         "that date use the most recent earlier date in fx.csv.",
            "invoice_posting": "AR debit and revenue credit = convert(total_minor, rate at invoice date).",
            "payment_posting": "Cash debit = convert(amount_minor, rate at payment date); AR credit per applied "
                               "invoice = convert(applied_minor, rate at that invoice's date); unapplied remainder to "
                               "2300 at the payment-date rate; any difference to 7900.",
            "rounding": "ROUND_HALF_UP to the base minor unit, per posting line.",
        },
        "allocation": "A receipt's invoice_refs are the authoritative allocation. A refund debits 2300.",
        "duplicates": "Two invoices of the same entity with the same po_ref and identical line items "
                      "(sku, quantity, unit price, service_period) are duplicates. Equal totals alone are not. "
                      "An invoice posted twice to AR is also a duplicate.",
        "findings": {
            "currency": "base currency",
            "amount": {
                "duplicate_invoice": "AR debit of the duplicate posting",
                "payment_allocation": "expected AR credit for the (payment, invoice) pair minus the booked AR credit",
                "fx_conversion": "|booked - correct| of the primary line (AR debit for invoices, cash debit for payments)",
                "period_cutoff": "booked primary line (AR debit for invoices, cash debit for payments)",
            },
            "affected_ids": {
                "duplicate_invoice": "both invoice_ids (duplicate document) or the single invoice_id (double posting)",
                "payment_allocation": "payment_id and invoice_id",
                "fx_conversion": "invoice_id or payment_id",
                "period_cutoff": "invoice_id or payment_id",
            },
            "evidence": {
                "duplicate_invoice": "both doc_ids + one ledger row of the duplicate posting; or doc_id + both AR "
                                     "(or both revenue) rows for a double posting",
                "payment_allocation": "payment doc_id + invoice doc_id + one row of the payment's posting (cash, "
                                      "2300, or AR row it credited) + fx_id of the invoice-date rate if the invoice "
                                      "is not in base currency",
                "fx_conversion": "doc_id + primary ledger row (AR or revenue / cash) + fx_id of the correct rate",
                "period_cutoff": "doc_id + primary ledger row (AR or revenue / cash)",
            },
            "evidence_aliases": "invoice_id / payment_id given as evidence are normalised to their doc_id",
            "tolerance": {"abs_minor": AMOUNT_TOL_ABS_MINOR, "rel": str(AMOUNT_TOL_REL),
                          "rule": "|claimed - expected| <= max(abs_minor, floor(rel * expected))"},
        },
    }


def generate(seed, config=None, secret=None):
    """Return (public, private) for (seed, config[, secret]). Deterministic.

    ``secret=None`` reads the evaluator-only environment variable AUDITGYM_SEED_SECRET (unset => "").
    """
    import os
    cfg = load_config(config)
    if secret is None:
        secret = os.environ.get(SECRET_ENV, "")
    gen = Generator(seed, cfg, secret)
    return gen.run()

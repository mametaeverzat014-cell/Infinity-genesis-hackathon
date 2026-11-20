"""Money, FX and time helpers. Money is always an integer number of minor units."""

from __future__ import annotations

import bisect
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, localcontext

from .constants import CURRENCY_EXPONENTS


def exponent(currency: str) -> int:
    return CURRENCY_EXPONENTS[currency]


def convert(amount_minor: int, src: str, base: str, rate: Decimal) -> int:
    """Convert minor units of ``src`` into minor units of ``base``.

    ``rate`` is the number of major ``base`` units per one major ``src`` unit.
    Rounding is ROUND_HALF_UP to the base minor unit, applied per posting line.
    """
    if src == base:
        return int(amount_minor)
    with localcontext() as ctx:
        ctx.prec = 60
        scale = Decimal(10) ** (exponent(base) - exponent(src))
        value = Decimal(int(amount_minor)) * Decimal(rate) * scale
        return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def amount_tolerance(expected_minor: int, abs_minor: int, rel: Decimal) -> int:
    rel_part = int((Decimal(abs(int(expected_minor))) * rel).to_integral_value(rounding="ROUND_FLOOR"))
    return max(int(abs_minor), rel_part)


def fmt_rate(rate: Decimal) -> str:
    s = format(rate, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ---------------------------------------------------------------- time helpers

def parse_offset(text: str) -> timezone:
    sign = 1 if text[0] == "+" else -1
    hh, mm = text[1:].split(":")
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


def offset_minutes(text: str) -> int:
    sign = 1 if text[0] == "+" else -1
    hh, mm = text[1:].split(":")
    return sign * (int(hh) * 60 + int(mm))


def offset_str(minutes: int) -> str:
    sign = "+" if minutes >= 0 else "-"
    m = abs(minutes)
    return "%s%02d:%02d" % (sign, m // 60, m % 60)


def fmt_ts(dt: datetime, offset_min: int) -> str:
    tz = timezone(timedelta(minutes=offset_min))
    return dt.astimezone(tz).replace(microsecond=0).isoformat()


def parse_ts(text: str) -> datetime:
    if not isinstance(text, str):
        raise ValueError("timestamp must be a string")
    t = text.strip()
    if t.endswith("Z") or t.endswith("z"):
        t = t[:-1] + "+00:00"
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        raise ValueError("timestamp without offset")
    return dt


def local_date(dt: datetime, tz: timezone) -> date:
    return dt.astimezone(tz).date()


class RateBook:
    """FX rates source->target by date with the policy's 'most recent prior date' fallback."""

    def __init__(self, rows):
        # rows: iterable of (date, src, tgt, Decimal rate, key)
        self._by_pair = {}
        for d, src, tgt, rate, key in rows:
            self._by_pair.setdefault((src, tgt), []).append((d, rate, key))
        for lst in self._by_pair.values():
            lst.sort(key=lambda x: x[0])
        self._dates = {p: [x[0] for x in lst] for p, lst in self._by_pair.items()}

    def lookup(self, src: str, tgt: str, on: date):
        """Return (rate, key, rate_date) or None. Same currency -> (1, None, on)."""
        if src == tgt:
            return Decimal(1), None, on
        lst = self._by_pair.get((src, tgt))
        if not lst:
            return None
        i = bisect.bisect_right(self._dates[(src, tgt)], on) - 1
        if i < 0:
            return None
        d, rate, key = lst[i]
        return rate, key, d

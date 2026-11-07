"""Type coercion shared by every extraction path.

Output contract (contract/README.md): numbers are JSON numbers (no $ or thousands separators),
dates are YYYY-MM-DD, percentages stay exactly as the document prints them (5.26 stays 5.26),
null means "not in the document".
"""
from __future__ import annotations

import calendar
import re
from datetime import date, datetime
from typing import Any

_NULLISH = {"", "n/a", "na", "none", "null", "-", "--", "—", "nan"}

_DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y",
                 "%d %b %Y", "%d %B %Y", "%m-%d-%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"]
_MONTH_FORMATS = ["%b %Y", "%B %Y", "%b-%Y", "%B-%Y", "%Y-%m", "%m/%Y", "%b %y", "%b-%y"]


def is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in _NULLISH)


def to_number(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (datetime, date)):
        return None
    s = str(v).strip()
    if s.lower() in _NULLISH:
        return None
    neg = (s.startswith("(") and s.endswith(")")) or s.endswith("-") and s[:-1].replace(",", "").replace(".", "").isdigit()
    s = s.strip("()").rstrip("-")
    s = re.sub(r"[,$€£₹%\s]", "", s)
    try:
        x = float(s)
    except ValueError:
        return None
    return -abs(x) if neg else x


def _parse_date(v: Any) -> date | None:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    for fmt in _MONTH_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().replace(day=1)
        except ValueError:
            pass
    return None


def to_date(v: Any, mode: str = "date") -> str | None:
    """mode: date | month_start | month_end | year_month"""
    if is_blank(v):
        return None
    d = _parse_date(v)
    if d is None:
        return None
    if mode == "month_start":
        d = d.replace(day=1)
    elif mode == "month_end":
        d = d.replace(day=calendar.monthrange(d.year, d.month)[1])
    elif mode == "year_month":
        return f"{d.year:04d}-{d.month:02d}"
    return d.isoformat()


def to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1"):
        return True
    if s in ("false", "no", "n", "0"):
        return False
    return None


def to_string(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (datetime, date)):
        return _parse_date(v).isoformat()          # type: ignore[union-attr]
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    return s if s != "" else None


def coerce(v: Any, as_type: str | None) -> Any:
    """Coerce a raw value. Returns None when it cannot be interpreted as `as_type`."""
    t = (as_type or "string").lower()
    if t in ("number", "float", "decimal", "money", "percent", "percentage"):
        return to_number(v)
    if t in ("integer", "int"):
        x = to_number(v)
        return None if x is None else int(round(x))
    if t in ("date", "month_start", "month_end", "year_month"):
        return to_date(v, "date" if t == "date" else t)
    if t in ("boolean", "bool"):
        return to_bool(v)
    return to_string(v)


def clean_number(x: float | None) -> float | int | None:
    """Stable JSON form: integral numbers stay floats like 2842.0 (validator treats them as equal)."""
    if x is None:
        return None
    return round(float(x), 6)

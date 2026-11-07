"""Cross-field arithmetic checks for 'calculated fields are still fields'.

Each rule fires only when the schema actually contains the keys it needs, so unknown field lists are
unaffected. A rule recomputes a value from its inputs and, if the model's value disagrees, replaces it
(the inputs are verified quotes; the arithmetic is exact). Every replacement is logged in `notes`.
"""
from __future__ import annotations

import calendar
from datetime import date
from typing import Any


def _d(v: Any) -> date | None:
    try:
        return date.fromisoformat(v) if isinstance(v, str) else None
    except ValueError:
        return None


def _month_end(d: date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def months_inclusive(start: date, end: date) -> int | None:
    """Whole months from `start` through `end` inclusive, when start is a 1st and end a month-end."""
    if start.day != 1 or not _month_end(end) or end < start:
        return None
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _close(a: float, b: float, tol: float = 0.005) -> bool:
    return abs(a - b) <= max(0.01, abs(b) * tol)


def apply_rules(values: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    """`values` maps field key -> python value. Returns {key: new_value} for fields to overwrite."""
    out: dict[str, Any] = {}
    g = values.get

    # lease term in months from the two dates
    if "lease_term_months" in values and "lease_commencement_date" in values and "lease_expiration_date" in values:
        s, e = _d(g("lease_commencement_date")), _d(g("lease_expiration_date"))
        if s and e:
            m = months_inclusive(s, e)
            cur = _num(g("lease_term_months"))
            if m is not None and (cur is None or int(cur) != m):
                out["lease_term_months"] = m
                notes.append(f"lease_term_months: {cur} -> {m} (from commencement/expiration dates)")

    # rent schedule: per-period months, annual $/RSF and total base rent
    sd, ed, mr = g("rent_schedule_start_dates"), g("rent_schedule_end_dates"), g("rent_schedule_monthly_rent")
    if all(isinstance(x, list) and x for x in (sd, ed, mr)) and len(sd) == len(ed) == len(mr):
        months = []
        for a, b in zip(sd, ed):
            da, db = _d(a), _d(b)
            months.append(months_inclusive(da, db) if da and db else None)
        rents = [_num(x) for x in mr]
        if all(m is not None for m in months) and all(r is not None for r in rents):
            total = float(sum(r * m for r, m in zip(rents, months)))
            if "total_base_rent" in values:
                cur = _num(g("total_base_rent"))
                if cur is None or not _close(cur, total):
                    out["total_base_rent"] = round(total, 2)
                    notes.append(f"total_base_rent: {cur} -> {total:.2f} (sum of monthly rent x months per period)")
        rsf = _num(g("rentable_square_footage"))
        if rsf and "rent_schedule_annual_rent_per_rsf" in values and all(r is not None for r in rents):
            calc = [round(r * 12 / rsf, 2) for r in rents]          # type: ignore[operator]
            cur_l = g("rent_schedule_annual_rent_per_rsf")
            if not (isinstance(cur_l, list) and len(cur_l) == len(calc) and all(
                    _num(c) is not None and _close(_num(c), k, 0.01) for c, k in zip(cur_l, calc))):
                out["rent_schedule_annual_rent_per_rsf"] = calc
                notes.append("rent_schedule_annual_rent_per_rsf recomputed as monthly x 12 / RSF")

    # offering memorandum arithmetic
    price, units = _num(g("asking_price")), _num(g("units"))
    if "price_per_unit" in values and price and units:
        calc = round(price / units)
        cur = _num(g("price_per_unit"))
        if cur is None or not _close(cur, calc, 0.002):
            out["price_per_unit"] = calc
            notes.append(f"price_per_unit: {cur} -> {calc}")
    rsf2 = _num(g("rentable_sf"))
    if "average_unit_sf" in values and g("average_unit_sf") is None and rsf2 and units:
        out["average_unit_sf"] = round(rsf2 / units)
        notes.append("average_unit_sf computed as rentable_sf / units")
    return out

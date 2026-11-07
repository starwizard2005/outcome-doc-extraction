"""Turn a schema (name + free-text description per field) into typed 'field cards'.

Type inference is deterministic (keyword rules over the description) so it is free and stable;
descriptions themselves reach the model verbatim plus the domain conventions in prompts.py.
Nothing here knows any particular field list.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class FieldCard:
    key: str
    name: str
    description: str
    type: str                 # string | number | integer | date | boolean | list
    item_type: str = "string"  # element type when type == "list"
    primary: bool = False
    nullable_hint: str = ""   # what to do when the document is silent


_LIST = re.compile(r"\blist of\b|\bone (?:list )?item per\b|\bin the order shown\b|\baligned item by item\b", re.I)
_BOOL = re.compile(r"^\s*(?:true if|true or false)\b|\btrue if\b.*\bfalse\b|\bboolean\b", re.I)
_DATE = re.compile(r"yyyy-mm-dd|\bdate\b", re.I)
_INT = re.compile(r"\bwhole (?:number|months|years)\b|\bfour-digit\b|\binteger\b|\bcount of\b|\bnumber of (?:units|buildings|stor|parking|months|spaces|options|model|down|admin)", re.I)
_NUM = re.compile(r"\bas a number\b|\bnumbers? only\b|\bas a (?:percentage )?number\b|\bdollar|\bpercentage number\b|\bsum of\b|\bsquare f|\bnumeric\b|\bamount\b", re.I)
_STR_FORCE = re.compile(r"\bas a string\b|\bkeep the .{0,30}wording\b|\bexactly as (?:written|shown|stated)\b|\bshort phrase\b"
                        r"|\bas the (?:lease|document|om|report|contract) (?:states?|says|words?|prints?)(?: it)?\b|\be\.g\.,? '[^']*[a-z]{3,}[^']*'", re.I)
_COLUMN_NUM = re.compile(r"\bcolumn\b.*\b(?:exactly as printed|as printed)\b|\b(?:exactly as printed|as printed)\b.*\bcolumn\b", re.I)
_NUMERIC_KEY = re.compile(r"percent|pct|trend|avail|(?:^|_)occ(?:_|$)|units|vacant|rented|model|down|admin|sq_?ft|rent|avg|count", re.I)


def infer_type(key: str, description: str) -> tuple[str, str]:
    d = description or ""
    if _LIST.search(d):
        # element type: dates > numbers > strings, judged from the same text
        if re.search(r"yyyy-mm-dd|start dates?|end dates?|\bdates?\b", d, re.I) and not re.search(r"\bmonths?\b.*\bnotice\b|\bin months\b", d, re.I):
            return "list", "date"
        if re.search(r"\bnumbers? only\b|\bmonthly\b|\bin months\b|\bper rentable\b|\bamounts?\b", d, re.I):
            return "list", "number"
        return "list", "string"
    if _STR_FORCE.search(d) and not re.search(r"\bas a (?:percentage )?number\b|^\s*true if", d, re.I) and not (
            re.search(r"yyyy-mm-dd", d, re.I) is None and _COLUMN_NUM.search(d)):
        pass_str = True
    else:
        pass_str = False
    if not pass_str and _BOOL.search(d):
        return "boolean", "string"
    if _COLUMN_NUM.search(d) and _NUMERIC_KEY.search(key):
        return "number", "string"
    if pass_str:
        # 'as written/shown/short phrase' => text, unless it is clearly a date/number field
        if _DATE.search(d) and re.search(r"yyyy-mm-dd", d, re.I):
            return "date", "string"
        return "string", "string"
    if re.search(r"yyyy-mm-dd", d, re.I) or (_DATE.search(d) and not _NUM.search(d) and not re.search(r"\bname\b", d, re.I)
                                             and re.search(r"\bdate\b", d, re.I) and not re.search(r"\bas a (?:four-digit )?(?:year|number)\b", d, re.I)):
        return "date", "string"
    if _INT.search(d):
        return "integer", "string"
    if _NUM.search(d) or re.search(r"(?:^|_)(?:pct|rate|price|rent|fee|amount|sf|sqft|count|total|ratio|noi|acres)(?:_|$)", key):
        return "number", "string"
    return "string", "string"


def build_cards(schema: dict) -> list[FieldCard]:
    cards: list[FieldCard] = []
    for key, f in (schema.get("fields") or {}).items():
        f = f if isinstance(f, dict) else {"description": str(f)}
        desc = f.get("description") or f.get("name") or key
        t, it = infer_type(key, desc)
        cards.append(FieldCard(key=key, name=f.get("name") or key, description=desc, type=t, item_type=it,
                               primary=bool(f.get("primary"))))
    return cards


def card_lines(cards: list[FieldCard]) -> str:
    """Compact field list for prompts."""
    out = []
    for c in cards:
        tt = f"list of {c.item_type}" if c.type == "list" else c.type
        out.append(f'- "{c.key}" ({tt}): {c.description}')
    return "\n".join(out)

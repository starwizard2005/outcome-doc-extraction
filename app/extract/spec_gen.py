"""LLM -> spreadsheet extraction spec, validated by actually running it.

The model sees a *skeleton* of the workbook (not the data): header rows, the distinct row shapes with
counts and examples, title/section rows, and the last rows. Long cells are truncated, so a sheet that
contains a 10,000-token parameter list costs the same as a tidy one. The spec it writes is executed by
sheet_exec (deterministic); structural errors, implausible results and an independent audit of sample
rows are fed back for up to two repairs before giving up (the caller then falls back to chunked
LLM extraction).
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter

from ..llm import LLMError, chat_json
from ..parse.types import ParsedDoc, Sheet, col_letter, render_rows
from .ctx import JobContext
from .fieldspec import FieldCard, card_lines
from .sheet_exec import SheetResult, SpecError, check_result, run_spec
from .normalize import to_number

log = logging.getLogger("spec_gen")


class SpecFailed(Exception):
    pass


# ------------------------------------------------------------------ skeleton

def _sig(sheet: Sheet, row: int) -> str:
    parts = []
    for c, v in sorted(sheet.row_values(row).items()):
        if hasattr(v, "year"):
            t = "d"
        elif to_number(v) is not None and not isinstance(v, str):
            t = "n"
        elif isinstance(v, str) and to_number(v) is not None:
            t = "n"
        else:
            t = "t"
        parts.append(f"{col_letter(c)}{t}")
    return " ".join(parts)


def sheet_skeleton(sheet: Sheet, *, top: int = 28, shapes: int = 10, titles: int = 40, tail: int = 8) -> str:
    rows = sheet.rows()
    out = [f'SHEET "{sheet.name}": {len(rows)} non-empty rows, rows {rows[0] if rows else 0}-{sheet.max_row}, '
           f'columns A-{col_letter(sheet.max_col)}']
    out.append("\nFIRST ROWS\n" + render_rows(sheet, rows[:top], limit=48))
    sigs: dict[str, list[int]] = {}
    for r in rows:
        sigs.setdefault(_sig(sheet, r), []).append(r)
    out.append(f"\nROW SHAPES (n=number, t=text, d=date; {len(sigs)} distinct shapes; most frequent first)")
    for sig, rs in sorted(sigs.items(), key=lambda kv: -len(kv[1]))[:shapes]:
        ex = render_rows(sheet, rs[:1], limit=34)
        out.append(f"x{len(rs):<5d} rows {rs[:4]}{'...' if len(rs) > 4 else ''}\n   {ex}")
    title_rows = []
    for r in rows:
        v = sheet.row_values(r)
        if 0 < len(v) <= 2 and all(isinstance(x, str) for x in v.values()):
            title_rows.append(f"{r}: " + " | ".join(str(x).strip()[:50] for x in v.values()))
    if title_rows:
        shown = title_rows if len(title_rows) <= titles else title_rows[: titles // 2] + ["…"] + title_rows[-titles // 2:]
        out.append("\nTITLE / SECTION-LIKE ROWS (<=2 text cells)\n" + "\n".join(shown))
    out.append("\nLAST ROWS\n" + render_rows(sheet, rows[-tail:], limit=48))
    return "\n".join(out)


def build_skeleton(doc: ParsedDoc, budget_chars: int = 16000) -> str:
    sheets = [s for s in doc.sheets if s.cells]
    visible = [s for s in sheets if not s.hidden] or sheets
    main = sorted(visible, key=lambda s: -len(s.cells))
    parts = [sheet_skeleton(main[0])]
    for s in sheets:
        if s is main[0]:
            continue
        parts.append(f'\nOTHER SHEET "{s.name}"{" (hidden)" if s.hidden else ""}: {len(s.cells)} cells; first rows:\n'
                     + render_rows(s, s.rows()[:12], limit=40))
    text = "\n".join(parts)
    return text if len(text) <= budget_chars else text[:budget_chars] + "\n…(truncated)"


# ------------------------------------------------------------------ prompts

GRAMMAR = """\
You write EXTRACTION SPECS for spreadsheets. A spec is JSON; a deterministic engine applies it to every row, so
you never output data rows yourself.

{
  "sheet": "<sheet name>",
  "layout": "flat" | "blocks",     // flat: one row = one record. blocks: an ANCHOR row (e.g. a unit line) followed by
                                   //  detail rows (e.g. its charge lines) that belong to it
  "first_row": <int>, "last_row": null,
  "record": [TEST, ...],           // ALL must hold for a row to be a record (flat) or block anchor (blocks)
  "skip":   [TEST, ...],           // optional: ANY holds -> row ignored (e.g. repeated headers, 'Total' rows)
  "stop":   [TEST, ...],           // optional: ANY holds -> stop reading here (trailing summaries, repeated sections)
  "duplicates": "first" | "last" | "keep",   // when the primary key repeats; default "first"
  "block": {"label_col": "H"},     // blocks only: column holding each detail line's label
  "fields": {"<key>": SOURCE, ...} // EVERY requested key must be present
}
TEST   := {"col":"A","op":"nonempty|empty|numeric|text|regex|not_regex|eq|in","value":...}
        | {"cols":["C","D"],"op":"any_numeric|none_numeric"} | {"any":[TEST]} | {"all":[TEST]} | {"not":TEST}
SOURCE := {"col":"B"}                                      value from this column of the record/anchor row
        | {"cell":"A3"}                                    a fixed cell (document-level fields: names, periods, dates)
        | {"const":null}                                   when the document has no such value
        optionally with "regex":"...","group":1 to pick part of the text (use a negative look-ahead to null out junk)
        | {"above":{"tests":[TEST],"col":"B"}}             nearest row ABOVE satisfying tests (section headers)
        | {"count_numeric_nonzero":["C","D",...]}          how many of these columns are non-zero numbers
        | {"sum_cols":["C","D"]}
        | {"detail_sum":{"label_in":["Rent"],"value_col":"I"}}      blocks: sum detail values whose label is in the list
        | {"detail_sum":{"label_not_in":["Rent"],"value_col":"I"}}  ... or whose (non-empty) label is NOT in the list
        | {"detail_value":{"label_col":"G","label_in":["Charge Total:"],"value_col":"I"}}  one labelled detail value
        | {"map":{"from":SOURCE,"table":{"x":"y"},"default":null}}
        | {"rules":[{"when":[TEST],"value":"Revenue"},...],"default":"Expense"}   first matching rule wins
        any SOURCE may add "as": string|number|integer|date|month_start|month_end|year_month|boolean
                        and "default": value used when the result is blank (e.g. 0 when the description says '0 if none')
Columns are letters exactly as in the skeleton. Return ONLY the JSON spec.
"""

SPEC_SYSTEM = """\
You are an expert at reading messy real-estate spreadsheets (rent rolls, operating statements/T12s, box scores,
budgets) and mapping them to a requested field list.

""" + GRAMMAR + """
HOW TO DECIDE
1. Read the schema description: it says what one row is and what is NOT a row (headers, totals, subtotals).
   Make `record`/`skip`/`stop` match exactly that, using shape evidence from the skeleton. A subtotal row that carries
   its own identifier (e.g. a GL code) is a row unless the description excludes it.
2. Fields that apply to the whole document ("same for every row") come from fixed cells with `cell` (+ regex).
   Fields that vary by row come from columns, `above`, `detail_*`, `rules` etc.
3. Honour each description literally: copy text exactly (trim only whitespace); numbers as printed; dates via "as";
   "0 if none"/"null if blank" -> `default`. A column that is similar-looking but not the one named is a trap: match the
   header text.
4. Blocks: when each logical record spans several rows, use layout "blocks"; anchors are the rows that start a record
   (test several columns so section headers and summary tables do NOT match). Use `stop` for trailing summary tables
   or repeated/future sections that are not part of the main listing.
5. If a value is not in the document at all, use {"const": null}.

EXAMPLES (abbreviated, different documents)
Flat table with header cells and month columns:
{"sheet":"Report1","layout":"flat","first_row":8,
 "record":[{"col":"A","op":"regex","value":"^\\\\d{3}-\\\\d{3}$"},{"cols":["C","D","E","F","G","H","I","J","K","L","M","N","O"],"op":"any_numeric"}],
 "fields":{"property_name":{"cell":"A1","regex":"Property\\\\s*=\\\\s*(.+)$"},
  "period_start":{"cell":"A3","regex":"=\\\\s*([A-Za-z]{3,9} \\\\d{4})","as":"month_start"},
  "gl_code":{"col":"A"},"line_item":{"col":"B"},
  "section":{"above":{"tests":[{"col":"B","op":"nonempty"},{"cols":["C","D","E","F","G","H","I","J","K","L","M","N","O"],"op":"none_numeric"}],"col":"B"}},
  "category":{"rules":[{"when":[{"col":"A","op":"regex","value":"^5"}],"value":"Revenue"}],"default":"Expense"},
  "amount_total":{"col":"O","as":"number"},"months_with_activity":{"count_numeric_nonzero":["C","D","E","F","G","H","I","J","K","L","M","N"]}}}
Blocks (unit line + charge lines):
{"sheet":"Rent Roll","layout":"blocks","first_row":8,
 "record":[{"col":"A","op":"nonempty"},{"col":"C","op":"numeric"},{"col":"D","op":"nonempty"}],
 "stop":[{"col":"A","op":"eq","value":"Status Summary"}],"duplicates":"first","block":{"label_col":"H"},
 "fields":{"unit_number":{"col":"A"},"resident_name":{"col":"E","regex":"^(?!\\\\s*--\\\\s*Vacant)(.+)$"},
  "monthly_rent":{"detail_sum":{"label_in":["Rent"],"value_col":"I"}},
  "monthly_other_charges":{"detail_sum":{"label_not_in":["Rent"],"value_col":"I"}},
  "monthly_total_charges":{"detail_value":{"label_col":"G","label_in":["Charge Total:"],"value_col":"I"},"default":0},
  "deposit_held":{"col":"K","as":"number","default":0},"move_in_date":{"col":"L","as":"date"}}}
"""

SPEC_USER = """\
SCHEMA: {name}
{desc}
Primary (row-identifying) field: "{primary}"

FIELDS
{fields}

WORKBOOK SKELETON
{skeleton}

Write the spec JSON now."""

AUDIT_SYSTEM = """\
You audit a spreadsheet extraction. You get the field descriptions, the spec's choices, and a few extracted records
next to the raw source rows they came from. Report ONLY definite mistakes: wrong column (including similar-looking
columns), wrong scope (document-level vs row-level), wrong type/format, wrong default, a value that should be blank
but is not, or a non-data row that was treated as a record. If everything matches, say so.
Return ONLY JSON: {"problems":[{"field":"<key>","issue":"<what is wrong and what it should be>"}]}  (empty list if fine)
"""


def _render_audit(doc: ParsedDoc, res: SheetResult, cards: list[FieldCard], n: int = 4) -> str:
    sheet = next(s for s in doc.sheets if s.name == res.sheet)
    idx = sorted({0, len(res.rows) // 2, len(res.rows) - 1} | set(random.Random(7).sample(range(len(res.rows)), min(n, len(res.rows)))))
    parts = []
    for i in idx[: n + 1]:
        raw = render_rows(sheet, res.source_rows[i][:14], limit=40)
        vals = {c.key: res.rows[i][c.key].value for c in cards}
        parts.append(f"--- record {i}\nRAW SOURCE ROWS\n{raw}\nEXTRACTED\n{json.dumps(vals, default=str, ensure_ascii=False)}")
    hdr = render_rows(sheet, sheet.rows()[:12], limit=40)
    return f"SHEET HEADER ROWS\n{hdr}\n\n" + "\n".join(parts)


# ------------------------------------------------------------------ driver

def generate_spec(doc: ParsedDoc, schema: dict, cards: list[FieldCard], primary: str, ctx: JobContext
                  ) -> tuple[dict, SheetResult]:
    s = ctx.settings
    keys = [c.key for c in cards]
    skeleton = build_skeleton(doc)
    user = SPEC_USER.format(name=schema.get("name", ""), desc=schema.get("description", ""), primary=primary,
                            fields=card_lines(cards), skeleton=skeleton)
    messages = [{"role": "system", "content": SPEC_SYSTEM}, {"role": "user", "content": user}]
    last_issue = "no attempt made"
    for attempt in range(3):
        model = s.model_spec if attempt < 2 else s.model_strong
        try:
            spec, _ = chat_json(model, messages, tracker=ctx.tracker, max_tokens=6000, deadline=ctx.deadline,
                                tag=f"spec-{attempt}")
        except LLMError as e:
            last_issue = str(e)
            if attempt == 2:
                break
            continue
        issues: list[str] = []
        res: SheetResult | None = None
        try:
            res = run_spec(doc, spec, keys, primary)
            issues = check_result(res, keys, primary, spec)
        except SpecError as e:
            issues = [f"spec could not be applied: {e}"]
        except Exception as e:  # noqa: BLE001  (e.g. bad regex / malformed values)
            issues = [f"spec raised {type(e).__name__}: {e}"]
        if not issues and res is not None:
            issues = _audit(doc, res, cards, ctx)
        if not issues and res is not None:
            if res.warnings:
                for w in res.warnings:
                    ctx.warn("spec: " + w)
            return spec, res
        last_issue = "; ".join(issues)
        ctx.warn(f"spec attempt {attempt + 1} rejected: {last_issue[:300]}")
        if res is not None and attempt == 2 and res.rows:
            # last attempt with a runnable spec: better than nothing, flagged
            ctx.warn("accepting last spec despite audit findings")
            return spec, res
        messages += [{"role": "assistant", "content": json.dumps(spec)},
                     {"role": "user", "content": "Your spec has these problems:\n- " + "\n- ".join(issues) +
                      "\nFix them and return the complete corrected spec JSON only."}]
    raise SpecFailed(last_issue)


def _audit(doc: ParsedDoc, res: SheetResult, cards: list[FieldCard], ctx: JobContext) -> list[str]:
    if not res.rows or not res.source_rows:
        return []
    try:
        user = "FIELD DESCRIPTIONS\n" + card_lines(cards) + "\n\n" + _render_audit(doc, res, cards)
        obj, _ = chat_json(ctx.settings.model_cheap, [{"role": "system", "content": AUDIT_SYSTEM},
                                                       {"role": "user", "content": user}],
                           tracker=ctx.tracker, max_tokens=1500, deadline=ctx.deadline, tag="spec-audit")
    except LLMError as e:
        ctx.warn(f"spec audit skipped: {e}")
        return []
    probs = obj.get("problems", []) if isinstance(obj, dict) else []
    return [f"{p.get('field', '?')}: {p.get('issue', '')}" for p in probs if isinstance(p, dict)][:8]

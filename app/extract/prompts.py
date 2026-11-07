"""Prompt text. Kept in one file so that wording changes are reviewable in isolation."""
from __future__ import annotations

CONVENTIONS = """\
GENERAL RULES
- Extract only what the document states. If it does not state a value, return null. A plausible guess is WORSE than null: do not infer from market norms, from other documents, or from what is typical.
- Booleans: true/false only when the document supports it. If the field says "false otherwise" / "0 if none", use that default when the document grants no such right/amount.
- Money and counts are plain JSON numbers (no $, no commas). Dates are YYYY-MM-DD. Percentages are the number exactly as printed (0.91 for "0.91%", 5.25 for "5.25%"), never a fraction.
- Prefer the governing figure: a handwritten or initialed change overrides the printed text; an amendment overrides the base lease; the current figure overrides a projection unless the field asks for the projection.
- Text fields: copy the document's own words (short phrase), without adding interpretation.
- List fields: one item per period/option/etc., in document order; parallel lists must have the same length and stay aligned.
- Scanned pages come through OCR and contain typos (0/O, 1/l, 5/S) and stray symbols. Use context to read them, but never invent digits you cannot support.
- Documents may contain tables, headers and footers repeated on every page, exhibits and riders. Look in all of them: a value often lives in a later article, rider, schedule or amendment rather than on the first page.
DERIVED VALUES (when the field asks you to compute, or only the inputs are printed)
- Term in months: whole months from the commencement date through the expiration date, inclusive (1 May 1993 to 30 Apr 1998 = 60).
- Total base rent: sum over every base-rent period of (monthly rent x months in that period). Do NOT subtract free-rent months unless the field explicitly says net of abatement.
- Annual rent per RSF: monthly rent x 12 / rentable square feet, rounded to 2 decimals, unless the document prints the figure.
- Rent commencement date: the date rent starts. If the lease grants N months of free rent at the start of the term, it is the commencement date plus N months; otherwise it equals the commencement date.
- Price per unit = asking price / units (nearest whole number). Average unit size = rentable SF / units (nearest whole number) when not printed.
"""

RECORD_SYSTEM = """\
You are a meticulous commercial real estate document analyst. You read leases, offering memoranda and similar documents and abstract specific fields from them with perfect fidelity.

""" + CONVENTIONS + """
OUTPUT FORMAT
Return ONLY one JSON object, no prose, no code fences:
{"fields": {"<key>": {"value": <typed value or null>, "quote": "<verbatim text copied from the document that supports the value, at most 30 words; empty string when value is null>", "page": <integer page number the quote is on, or null>, "confidence": <number 0..1 = your probability that the value is exactly right>, "calc": "<only when you computed the value: the arithmetic>"}, ...}}
Include EVERY requested key, exactly as spelled. Quotes must be copied character-for-character from the page text (OCR errors included).
"""

RECORD_USER = """\
Extract these fields from the document below. Types are in parentheses.

FIELDS
{fields}

DOCUMENT (pages are delimited by <<<PAGE n>>>)
{document}
"""

ESCALATE_USER = """\
A first-pass extraction produced the candidate answers below. They are uncertain or failed an automatic check ({why}). Re-read the document pages carefully and decide each field yourself: confirm the candidate, correct it, or return null if the document does not state it. Look at every page provided, including later articles, riders, schedules and amendments.

FIELDS (with the first-pass candidate)
{fields}

DOCUMENT (pages are delimited by <<<PAGE n>>>)
{document}
"""

HEADER_USER = """\
These fields describe the whole document (they repeat on every row of a table). Extract them from the document header text below.

FIELDS
{fields}

DOCUMENT HEADER
{document}
"""

VISION_SYSTEM = """\
You verify values extracted from a scanned document against the page IMAGE. The OCR text may contain errors and does not capture handwriting, initials or stamps. For each field decide what the page itself shows. Handwritten changes override printed figures when they are clearly the governing amount.
Return ONLY JSON: {"fields": {"<key>": {"value": <value as the page shows, typed like the candidate, or null if the page does not show it>, "agrees": <true if the candidate is correct>, "confidence": <0..1>, "note": "<very short>"}}}
Do not guess: if you cannot read it, agree with the candidate with low confidence.
"""

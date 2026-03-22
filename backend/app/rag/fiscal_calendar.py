"""Fiscal calendar utilities for temporal resolution.

Maps company tickers to their fiscal year end months and provides functions
to compute calendar period boundaries from fiscal quarter labels.
"""

import calendar
import re
from datetime import date

# Static mapping: ticker → month (1-12) in which the company's fiscal year ends.
# Default is 12 (December / calendar year) for any ticker not in this map.
# Verified against earnings call date patterns in backend/transcripts/.
FY_END_MONTH: dict[str, int] = {
    # January (1)
    "ASAN": 1,
    "CRM": 1,
    "CRWD": 1,
    "CXM": 1,
    "DOCU": 1,
    "GTLB": 1,
    "MDB": 1,
    "NVDA": 1,
    "OKTA": 1,
    "PATH": 1,
    "RBRK": 1,
    "SNOW": 1,
    "TTAN": 1,
    "VEEV": 1,
    "WDAY": 1,
    "IOT": 1,
    "AI": 1,
    # March (3)
    "DT": 3,
    # June (6)
    "MSFT": 6,
    "PCTY": 6,
    "TEAM": 6,
    # July (7)
    "NTNX": 7,
    "PANW": 7,
    "ZS": 7,
    # November (11)
    "ADBE": 11,
    # December (12) — all others default to 12; listed explicitly for completeness
    "AMZN": 12,
    "APPF": 12,
    "CFLT": 12,
    "DDOG": 12,
    "DOCN": 12,
    "FRSH": 12,
    "GOOGL": 12,
    "HUBS": 12,
    "KVYO": 12,
    "META": 12,
    "MNDY": 12,
    "NET": 12,
    "NOW": 12,
    "PAYC": 12,
    "PCOR": 12,
    "PLTR": 12,
    "SAIL": 1,
    "SHOP": 12,
    "TOST": 12,
    "TWLO": 12,
}

# Regex for parsing fiscal quarter strings like "Q2 FY2025", "Q4 FY25", "Q1 CY2024"
_FQ_RE = re.compile(
    r"Q([1-4])\s*(?:FY|CY)?\s*(\d{2,4})",
    re.IGNORECASE,
)


def parse_fiscal_quarter(fq: str) -> tuple[int, int] | None:
    """Parse a fiscal quarter string into (quarter_number, year).

    Handles: "Q2 FY2025", "Q4 FY25", "Q1 CY2024", "Q3 2025".
    Normalizes 2-digit years (25 → 2025). Returns None if unparseable.
    """
    if not fq:
        return None
    m = _FQ_RE.search(fq)
    if not m:
        return None
    quarter = int(m.group(1))
    year = int(m.group(2))
    if year < 100:
        year += 2000
    return quarter, year


def compute_period_end(fiscal_quarter: str, ticker: str) -> date | None:
    """Compute the calendar date a fiscal quarter ends.

    Given a fiscal quarter label (e.g. "Q2 FY2025") and a ticker, uses the
    company's fiscal year end month to determine the actual calendar period.

    Example: MSFT (FY ends June) Q2 FY2025 → Oct–Dec 2024 → period_end = 2024-12-31.
    """
    parsed = parse_fiscal_quarter(fiscal_quarter)
    if not parsed:
        return None
    quarter_num, fy_year = parsed
    fy_end_month = FY_END_MONTH.get(ticker.upper(), 12)
    return _compute_period_end_from_parts(quarter_num, fy_year, fy_end_month)


def _compute_period_end_from_parts(quarter_num: int, fy_year: int, fy_end_month: int) -> date:
    """Core period_end calculation from structured inputs.

    FY{fy_year} ends in month {fy_end_month} of calendar year {fy_year}.
    Q1 starts in the month after the previous FY ended.
    """
    # FY starts the month after it ended the previous year
    fy_start_month = (fy_end_month % 12) + 1
    fy_start_year = fy_year if fy_end_month == 12 else fy_year - 1

    # Quarter N starts (N-1)*3 months after FY start
    q_start_month = fy_start_month + (quarter_num - 1) * 3
    q_start_year = fy_start_year
    while q_start_month > 12:
        q_start_month -= 12
        q_start_year += 1

    # Quarter ends 2 months after start
    q_end_month = q_start_month + 2
    q_end_year = q_start_year
    while q_end_month > 12:
        q_end_month -= 12
        q_end_year += 1

    last_day = calendar.monthrange(q_end_year, q_end_month)[1]
    return date(q_end_year, q_end_month, last_day)


def compute_cy_quarter_end(quarter: int, year: int) -> date:
    """Compute the last day of a calendar year quarter.

    CY Q1 2025 → 2025-03-31, CY Q4 2024 → 2024-12-31.
    """
    end_month = quarter * 3
    last_day = calendar.monthrange(year, end_month)[1]
    return date(year, end_month, last_day)


def period_end_to_label(period_end: date) -> str:
    """Convert a period_end date to a human-readable calendar period label.

    Example: 2024-12-31 → "Oct–Dec 2024", 2025-03-31 → "Jan–Mar 2025".
    """
    month_names = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    end_month = period_end.month
    # Quarter start is 2 months before end
    start_month = end_month - 2
    start_year = period_end.year
    if start_month <= 0:
        start_month += 12
        start_year -= 1

    if start_year == period_end.year:
        return f"{month_names[start_month]}–{month_names[end_month]} {period_end.year}"
    return f"{month_names[start_month]} {start_year}–{month_names[end_month]} {period_end.year}"

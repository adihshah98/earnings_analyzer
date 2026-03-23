"""Tests for `app.rag.fiscal_calendar` (FY calendars, period labels, CY quarters)."""

from datetime import date

from app.rag.fiscal_calendar import (
    FY_END_MONTH,
    _compute_period_end_from_parts,
    compute_cy_quarter_end,
    compute_period_end,
    cy_quarter_label_from_period_end,
    parse_fiscal_quarter,
    period_end_to_label,
)


class TestFyEndMonthMap:
    def test_iot_and_ai_use_january_fye(self):
        """IOT/AI are mapped to FY ending January per `FY_END_MONTH`."""
        assert FY_END_MONTH.get("IOT") == 1
        assert FY_END_MONTH.get("AI") == 1


class TestCyQuarterLabelFromPeriodEnd:
    def test_q1(self):
        assert cy_quarter_label_from_period_end(date(2024, 3, 31)) == "CY Q1 2024"

    def test_q4(self):
        assert cy_quarter_label_from_period_end(date(2024, 12, 31)) == "CY Q4 2024"


class TestParseFiscalQuarter:
    def test_standard_fy(self):
        assert parse_fiscal_quarter("Q2 FY2025") == (2, 2025)

    def test_two_digit_year(self):
        assert parse_fiscal_quarter("Q4 FY25") == (4, 2025)

    def test_cy_prefix(self):
        assert parse_fiscal_quarter("Q1 CY2024") == (1, 2024)

    def test_no_prefix(self):
        assert parse_fiscal_quarter("Q3 2025") == (3, 2025)

    def test_no_space(self):
        assert parse_fiscal_quarter("Q1FY2026") == (1, 2026)

    def test_empty(self):
        assert parse_fiscal_quarter("") is None

    def test_invalid(self):
        assert parse_fiscal_quarter("some text") is None

    def test_none(self):
        assert parse_fiscal_quarter(None) is None


class TestComputePeriodEnd:
    # --- MSFT: FY ends June (6) ---
    def test_msft_q1_fy2025(self):
        assert compute_period_end("Q1 FY2025", "MSFT") == date(2024, 9, 30)

    def test_msft_q2_fy2025(self):
        assert compute_period_end("Q2 FY2025", "MSFT") == date(2024, 12, 31)

    def test_msft_q3_fy2025(self):
        assert compute_period_end("Q3 FY2025", "MSFT") == date(2025, 3, 31)

    def test_msft_q4_fy2025(self):
        assert compute_period_end("Q4 FY2025", "MSFT") == date(2025, 6, 30)

    # --- FY ends September (9), e.g. Apple ---
    def test_fy_end_sep_q1(self):
        assert _compute_period_end_from_parts(1, 2025, 9) == date(2024, 12, 31)

    # --- CRM: FY ends January (1) ---
    def test_crm_q1_fy2026(self):
        assert compute_period_end("Q1 FY2026", "CRM") == date(2025, 4, 30)

    def test_crm_q4_fy2026(self):
        assert compute_period_end("Q4 FY2026", "CRM") == date(2026, 1, 31)

    # --- GOOGL: FY ends December (12) = calendar year ---
    def test_googl_q1_fy2025(self):
        assert compute_period_end("Q1 FY2025", "GOOGL") == date(2025, 3, 31)

    def test_googl_q3_fy2025(self):
        assert compute_period_end("Q3 FY2025", "GOOGL") == date(2025, 9, 30)

    def test_googl_q4_fy2024(self):
        assert compute_period_end("Q4 FY2024", "GOOGL") == date(2024, 12, 31)

    # --- IOT: FY ends January (same mapping as CRM family in FY_END_MONTH) ---
    def test_iot_q1_fy2026(self):
        # Q1 FY2026 → period ending Apr 2025
        assert compute_period_end("Q1 FY2026", "IOT") == date(2025, 4, 30)

    def test_iot_q4_fy2025(self):
        assert compute_period_end("Q4 FY2025", "IOT") == date(2025, 1, 31)

    # --- PANW: FY ends July (7) ---
    def test_panw_q1_fy2025(self):
        assert compute_period_end("Q1 FY2025", "PANW") == date(2024, 10, 31)

    def test_panw_q4_fy2025(self):
        assert compute_period_end("Q4 FY2025", "PANW") == date(2025, 7, 31)

    # --- ADBE: FY ends November (11) ---
    def test_adbe_q1_fy2025(self):
        assert compute_period_end("Q1 FY2025", "ADBE") == date(2025, 2, 28)

    def test_adbe_q4_fy2025(self):
        assert compute_period_end("Q4 FY2025", "ADBE") == date(2025, 11, 30)

    # --- DT: FY ends March (3) ---
    def test_dt_q1_fy2025(self):
        assert compute_period_end("Q1 FY2025", "DT") == date(2024, 6, 30)

    # --- AI: FY ends January (see FY_END_MONTH) ---
    def test_ai_q1_fy2025(self):
        assert compute_period_end("Q1 FY2025", "AI") == date(2024, 4, 30)

    def test_bad_string(self):
        assert compute_period_end("not a quarter", "MSFT") is None

    def test_two_digit_year(self):
        assert compute_period_end("Q2 FY25", "MSFT") == date(2024, 12, 31)

    def test_unknown_ticker(self):
        assert compute_period_end("Q1 FY2025", "ZZZZ") == date(2025, 3, 31)

    def test_iot_q4_fy2024(self):
        assert compute_period_end("Q4 FY2024", "IOT") == date(2024, 1, 31)


class TestComputeCYQuarterEnd:
    def test_q1(self):
        assert compute_cy_quarter_end(1, 2025) == date(2025, 3, 31)

    def test_q2(self):
        assert compute_cy_quarter_end(2, 2025) == date(2025, 6, 30)

    def test_q3(self):
        assert compute_cy_quarter_end(3, 2025) == date(2025, 9, 30)

    def test_q4(self):
        assert compute_cy_quarter_end(4, 2024) == date(2024, 12, 31)


class TestPeriodEndToLabel:
    def test_q4_same_year(self):
        assert period_end_to_label(date(2024, 12, 31)) == "Oct–Dec 2024"

    def test_q1_same_year(self):
        assert period_end_to_label(date(2025, 3, 31)) == "Jan–Mar 2025"

    def test_cross_year(self):
        assert period_end_to_label(date(2026, 1, 31)) == "Nov 2025–Jan 2026"

    def test_feb(self):
        assert period_end_to_label(date(2025, 2, 28)) == "Dec 2024–Feb 2025"

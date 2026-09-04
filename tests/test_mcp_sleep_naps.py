"""Tests for nap exposure through the MCP server: descriptions, guide, summary SQL."""

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("fastmcp")

from garmy.localdb.db import HealthDB
from garmy.mcp import server


def _seed(db_path: Path) -> None:
    """Create a DB with a mix of nap days, a nap-free day, a pre-nap-support day."""
    db = HealthDB(db_path)
    today = date.today()
    db.store_health_metric(
        1,
        today - timedelta(days=1),
        sleep_duration_hours=8.0,
        nap_count=2,
        nap_duration_hours=0.75,
    )
    db.store_health_metric(
        1,
        today - timedelta(days=2),
        sleep_duration_hours=7.0,
        nap_count=0,
        nap_duration_hours=0.0,
    )
    db.store_health_metric(
        1,
        today - timedelta(days=3),
        sleep_duration_hours=6.0,
        nap_count=1,
        nap_duration_hours=0.25,
    )
    # Synced before nap support: nap columns stay NULL
    db.store_health_metric(1, today - timedelta(days=4), sleep_duration_hours=7.5)
    # Another user must not leak into user 1's summary
    db.store_health_metric(
        2,
        today - timedelta(days=1),
        sleep_duration_hours=5.0,
        nap_count=5,
        nap_duration_hours=3.0,
    )


def _run_summary(db_path: Path, user_id: int = 1, days: int = 30) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(server._HEALTH_SUMMARY_QUERY, [user_id, days]).fetchone()
        return dict(row)
    finally:
        conn.close()


class TestTableDescriptions:
    """sleep_naps is described for explore_database_structure / get_table_details."""

    def test_sleep_naps_has_dedicated_description(self) -> None:
        description = server._get_table_description("sleep_naps")

        assert description != "Health data table"
        assert "calendar_date = metric_date" in description
        assert "EXCLUDES naps" in description

    def test_unknown_table_still_falls_back(self) -> None:
        assert server._get_table_description("no_such_table") == "Health data table"


class TestHealthDataGuide:
    """The analysis guide teaches the nap semantics."""

    def test_guide_has_sleep_naps_section(self) -> None:
        guide = server._get_health_data_guide()

        assert "### sleep_naps" in guide
        assert "sleep_naps.calendar_date = daily_health_metrics.metric_date" in guide

    def test_guide_explains_total_sleep_rule(self) -> None:
        guide = server._get_health_data_guide()

        assert "COALESCE(nap_duration_hours, 0)" in guide
        assert "nap_count IS NULL" in guide
        assert "nap_duration_hours, nap_count" in guide


class TestHealthSummaryQuery:
    """The SQL behind get_health_summary reports nap totals correctly."""

    def test_query_passes_read_only_validation(self) -> None:
        server.QueryValidator.validate_query(server._HEALTH_SUMMARY_QUERY)

    def test_nap_fields(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mcp.db"
        _seed(db_path)

        summary = _run_summary(db_path)

        assert summary["total_days_with_data"] == 4
        assert summary["total_naps"] == 3
        # Average over nap days only: (0.75 + 0.25) / 2
        assert summary["avg_nap_hours_on_nap_days"] == 0.5
        # Main sleep average excludes naps: (8 + 7 + 6 + 7.5) / 4
        assert summary["avg_sleep_hours"] == 7.1

    def test_other_user_not_included(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mcp.db"
        _seed(db_path)

        summary = _run_summary(db_path, user_id=2)

        assert summary["total_naps"] == 5
        assert summary["avg_nap_hours_on_nap_days"] == 3.0

    def test_no_nap_data_yields_nulls(self, tmp_path: Path) -> None:
        db_path = tmp_path / "mcp.db"
        db = HealthDB(db_path)
        db.store_health_metric(
            1, date.today() - timedelta(days=1), sleep_duration_hours=8.0
        )

        summary = _run_summary(db_path)

        assert summary["total_days_with_data"] == 1
        assert summary["total_naps"] is None
        assert summary["avg_nap_hours_on_nap_days"] is None

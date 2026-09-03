"""Tests for nap support in localdb: extractor, DB store/migration, sync path."""

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from garmy.localdb.db import HealthDB, _parse_iso_datetime
from garmy.localdb.extractors import DataExtractor
from garmy.localdb.models import MetricType, SleepNapRecord
from garmy.localdb.sync import SyncManager
from garmy.metrics.sleep import Sleep, SleepNap, SleepSummary

DAY = date(2026, 5, 1)
NEXT_DAY = date(2026, 5, 2)


def make_nap(
    start_gmt: str = "2026-05-01T20:00:00",
    end_gmt: str = "2026-05-01T20:30:00",
    seconds: int = 1800,
    offset: int = -25200,
    feedback: Optional[str] = "IDEAL_TIMING_IDEAL_DURATION_LOW_NEED",
) -> SleepNap:
    """Build a SleepNap with invented but realistic values."""
    return SleepNap(
        nap_time_sec=seconds,
        nap_start_timestamp_gmt=start_gmt,
        nap_end_timestamp_gmt=end_gmt,
        nap_start_time_offset=offset,
        nap_end_time_offset=offset,
        nap_feedback=feedback,
        nap_source=0,
        device_id=42,
        calendar_date=DAY.isoformat(),
        user_profile_pk=12345,
    )


def make_second_nap() -> SleepNap:
    return make_nap(
        start_gmt="2026-05-01T23:00:00",
        end_gmt="2026-05-01T23:15:00",
        seconds=900,
        feedback="MULTIPLE_NAPS_DURING_DAY",
    )


def make_sleep(
    naps: Optional[List[SleepNap]] = None, nap_seconds: Optional[int] = None
) -> Sleep:
    """Build a Sleep object with an 8h main window and the given naps."""
    naps = naps if naps is not None else []
    if nap_seconds is None:
        nap_seconds = sum(n.nap_time_sec for n in naps)
    summary = SleepSummary(
        calendar_date=DAY.isoformat(),
        sleep_time_seconds=28800,
        nap_time_seconds=nap_seconds,
        deep_sleep_seconds=7200,
        light_sleep_seconds=14400,
        rem_sleep_seconds=7200,
    )
    return Sleep(sleep_summary=summary, naps=naps)


def stored_naps(db: HealthDB, user_id: int = 1) -> List[Dict[str, Any]]:
    """Read sleep_naps rows as plain dicts, ordered by GMT start."""
    with db.get_session() as session:
        rows = (
            session.query(SleepNapRecord)
            .filter(SleepNapRecord.user_id == user_id)
            .order_by(SleepNapRecord.nap_start_timestamp_gmt)
            .all()
        )
        return [
            {
                "calendar_date": r.calendar_date,
                "nap_start_timestamp_gmt": r.nap_start_timestamp_gmt,
                "nap_end_timestamp_gmt": r.nap_end_timestamp_gmt,
                "nap_start_timestamp_local": r.nap_start_timestamp_local,
                "nap_end_timestamp_local": r.nap_end_timestamp_local,
                "nap_time_seconds": r.nap_time_seconds,
                "nap_feedback": r.nap_feedback,
                "nap_source": r.nap_source,
                "device_id": r.device_id,
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class TestSleepExtractor:
    """DataExtractor nap handling for daily columns and sleep_naps rows."""

    def test_daily_columns_without_naps_are_zero_not_null(self) -> None:
        result = DataExtractor()._extract_sleep_data(make_sleep())

        assert result["nap_duration_hours"] == 0.0
        assert result["nap_count"] == 0

    def test_daily_columns_with_naps(self) -> None:
        sleep = make_sleep([make_nap(), make_second_nap()])

        result = DataExtractor()._extract_sleep_data(sleep)

        assert result["nap_duration_hours"] == pytest.approx(0.75)
        assert result["nap_count"] == 2
        # Main sleep is untouched by naps
        assert result["sleep_duration_hours"] == 8.0

    def test_daily_hours_come_from_summary_even_without_list(self) -> None:
        sleep = make_sleep(naps=[], nap_seconds=600)

        result = DataExtractor()._extract_sleep_data(sleep)

        assert result["nap_duration_hours"] == pytest.approx(600 / 3600)
        assert result["nap_count"] == 0

    def test_dispatch_via_extract_metric_data(self) -> None:
        result = DataExtractor().extract_metric_data(
            make_sleep([make_nap()]), MetricType.SLEEP
        )

        assert isinstance(result, dict)
        assert result["nap_count"] == 1

    def test_extract_sleep_naps_rows(self) -> None:
        sleep = make_sleep([make_nap(), make_second_nap()])

        rows = DataExtractor().extract_sleep_naps(sleep)

        assert len(rows) == 2
        first = rows[0]
        assert first["nap_start_timestamp_gmt"] == datetime(2026, 5, 1, 20, 0)
        assert first["nap_end_timestamp_gmt"] == datetime(2026, 5, 1, 20, 30)
        # Local = GMT + offset seconds (-7h)
        assert first["nap_start_timestamp_local"] == datetime(2026, 5, 1, 13, 0)
        assert first["nap_end_timestamp_local"] == datetime(2026, 5, 1, 13, 30)
        assert first["nap_time_seconds"] == 1800
        assert first["nap_feedback"] == "IDEAL_TIMING_IDEAL_DURATION_LOW_NEED"
        assert first["nap_source"] == 0
        assert first["device_id"] == 42
        assert rows[1]["nap_time_seconds"] == 900

    def test_extract_sleep_naps_skips_unparseable_start(self) -> None:
        sleep = make_sleep([make_nap(start_gmt="not-a-date"), make_second_nap()])

        rows = DataExtractor().extract_sleep_naps(sleep)

        assert len(rows) == 1
        assert rows[0]["nap_time_seconds"] == 900

    def test_extract_sleep_naps_without_naps_attribute(self) -> None:
        assert DataExtractor().extract_sleep_naps(object()) == []
        assert DataExtractor().extract_sleep_naps(None) == []


# ---------------------------------------------------------------------------
# HealthDB
# ---------------------------------------------------------------------------


class TestParseIsoDatetime:
    """Module-level ISO helper shared by health snapshots and naps."""

    def test_none_and_datetime_pass_through(self) -> None:
        now = datetime(2026, 5, 1, 12, 0)
        assert _parse_iso_datetime(None) is None
        assert _parse_iso_datetime(now) is now

    def test_iso_strings(self) -> None:
        assert _parse_iso_datetime("2026-05-01T20:00:00") == datetime(2026, 5, 1, 20, 0)
        aware = _parse_iso_datetime("2026-05-01T20:00:00Z")
        assert aware == datetime(2026, 5, 1, 20, 0, tzinfo=timezone.utc)

    def test_garbage_returns_none(self) -> None:
        assert _parse_iso_datetime("garbage") is None
        assert _parse_iso_datetime(12345) is None


class TestHealthDBSleepNaps:
    """HealthDB.store_sleep_naps replace-per-day semantics and migration."""

    def _rows_for(self, *naps: SleepNap) -> List[Dict[str, Any]]:
        return DataExtractor().extract_sleep_naps(make_sleep(list(naps)))

    def test_store_writes_rows(self, tmp_path: Path) -> None:
        db = HealthDB(tmp_path / "t.db")

        db.store_sleep_naps(1, DAY, self._rows_for(make_nap(), make_second_nap()))

        rows = stored_naps(db)
        assert len(rows) == 2
        assert rows[0]["calendar_date"] == DAY
        assert rows[0]["nap_start_timestamp_gmt"] == datetime(2026, 5, 1, 20, 0)
        assert rows[0]["nap_start_timestamp_local"] == datetime(2026, 5, 1, 13, 0)
        assert rows[0]["nap_time_seconds"] == 1800
        assert rows[0]["nap_feedback"] == "IDEAL_TIMING_IDEAL_DURATION_LOW_NEED"
        assert rows[0]["device_id"] == 42
        assert rows[1]["nap_feedback"] == "MULTIPLE_NAPS_DURING_DAY"

    def test_resync_replaces_day(self, tmp_path: Path) -> None:
        db = HealthDB(tmp_path / "t.db")
        db.store_sleep_naps(1, DAY, self._rows_for(make_nap(), make_second_nap()))

        # Garmin now reports only the second nap (first one deleted by the user)
        db.store_sleep_naps(1, DAY, self._rows_for(make_second_nap()))

        rows = stored_naps(db)
        assert len(rows) == 1
        assert rows[0]["nap_time_seconds"] == 900

    def test_empty_list_clears_day(self, tmp_path: Path) -> None:
        db = HealthDB(tmp_path / "t.db")
        db.store_sleep_naps(1, DAY, self._rows_for(make_nap()))

        db.store_sleep_naps(1, DAY, [])

        assert stored_naps(db) == []

    def test_other_days_and_users_untouched(self, tmp_path: Path) -> None:
        db = HealthDB(tmp_path / "t.db")
        db.store_sleep_naps(1, DAY, self._rows_for(make_nap()))
        next_day_nap = make_nap(
            start_gmt="2026-05-02T20:00:00", end_gmt="2026-05-02T20:20:00", seconds=1200
        )
        db.store_sleep_naps(1, NEXT_DAY, self._rows_for(next_day_nap))
        db.store_sleep_naps(2, DAY, self._rows_for(make_nap()))

        db.store_sleep_naps(1, DAY, [])

        assert [r["calendar_date"] for r in stored_naps(db, 1)] == [NEXT_DAY]
        assert len(stored_naps(db, 2)) == 1

    def test_same_start_under_new_calendar_date_merges(self, tmp_path: Path) -> None:
        """Same GMT start re-reported under another day must not raise."""
        db = HealthDB(tmp_path / "t.db")
        db.store_sleep_naps(1, DAY, self._rows_for(make_nap()))

        db.store_sleep_naps(1, NEXT_DAY, self._rows_for(make_nap()))

        rows = stored_naps(db)
        assert len(rows) == 1
        assert rows[0]["calendar_date"] == NEXT_DAY

    def test_accepts_iso_strings_and_skips_missing_start(self, tmp_path: Path) -> None:
        db = HealthDB(tmp_path / "t.db")
        rows: List[Dict[str, Any]] = [
            {
                "nap_start_timestamp_gmt": "2026-05-01T20:00:00",
                "nap_end_timestamp_gmt": "2026-05-01T20:30:00",
                "nap_start_timestamp_local": "2026-05-01T13:00:00",
                "nap_end_timestamp_local": "2026-05-01T13:30:00",
                "nap_time_seconds": 1800,
            },
            {"nap_start_timestamp_gmt": None, "nap_time_seconds": 100},
            {"nap_start_timestamp_gmt": "bogus", "nap_time_seconds": 100},
        ]

        db.store_sleep_naps(1, DAY, rows)

        stored = stored_naps(db)
        assert len(stored) == 1
        assert stored[0]["nap_start_timestamp_gmt"] == datetime(2026, 5, 1, 20, 0)
        assert stored[0]["nap_end_timestamp_local"] == datetime(2026, 5, 1, 13, 30)
        assert stored[0]["nap_feedback"] is None

    def test_daily_metric_round_trip_includes_nap_keys(self, tmp_path: Path) -> None:
        db = HealthDB(tmp_path / "t.db")

        db.store_health_metric(1, DAY, nap_duration_hours=0.75, nap_count=2)

        (row,) = db.get_health_metrics(1, DAY, DAY)
        assert row["nap_duration_hours"] == 0.75
        assert row["nap_count"] == 2

    def test_migration_adds_nap_columns_and_table(self, tmp_path: Path) -> None:
        """An existing DB created before nap support gains the columns/table."""
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE daily_health_metrics ("
            "user_id INTEGER NOT NULL, metric_date DATE NOT NULL, "
            "sleep_duration_hours FLOAT, PRIMARY KEY (user_id, metric_date))"
        )
        conn.execute("INSERT INTO daily_health_metrics VALUES (1, '2026-05-01', 8.0)")
        conn.commit()
        conn.close()

        HealthDB(db_path)

        conn = sqlite3.connect(str(db_path))
        try:
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(daily_health_metrics)")
            }
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            existing = conn.execute(
                "SELECT sleep_duration_hours, nap_count FROM daily_health_metrics"
            ).fetchone()
        finally:
            conn.close()

        assert {"nap_duration_hours", "nap_count"} <= columns
        assert "sleep_naps" in tables
        # Pre-existing data survives and new columns are NULL until re-synced
        assert existing == (8.0, None)


# ---------------------------------------------------------------------------
# SyncManager
# ---------------------------------------------------------------------------


class TestSyncManagerSleepNaps:
    """_sync_metric_for_date(SLEEP) writes daily columns and sleep_naps rows."""

    def _build_manager(
        self,
        tmp_path: Path,
        sleep: Any,
        metric_type: MetricType = MetricType.SLEEP,
    ) -> SyncManager:
        manager = SyncManager(db_path=tmp_path / "sync.db")
        manager.api_client = MagicMock()
        manager.api_client.metrics.get.return_value.get.return_value = sleep
        # sync_range() pre-creates a pending row per (date, metric) before the
        # per-date loop runs; update_sync_status only updates existing rows.
        manager.db.create_sync_status(1, DAY, metric_type)
        return manager

    @staticmethod
    def _stats() -> Dict[str, int]:
        return {"completed": 0, "skipped": 0, "failed": 0, "total_tasks": 1}

    def test_sleep_sync_stores_daily_columns_and_naps(self, tmp_path: Path) -> None:
        manager = self._build_manager(
            tmp_path, make_sleep([make_nap(), make_second_nap()])
        )
        stats = self._stats()

        manager._sync_metric_for_date(1, DAY, MetricType.SLEEP, stats)

        assert stats == {"completed": 1, "skipped": 0, "failed": 0, "total_tasks": 1}
        assert manager.db.get_sync_status(1, DAY, MetricType.SLEEP) == "completed"
        (daily,) = manager.db.get_health_metrics(1, DAY, DAY)
        assert daily["sleep_duration_hours"] == 8.0
        assert daily["nap_duration_hours"] == pytest.approx(0.75)
        assert daily["nap_count"] == 2
        rows = stored_naps(manager.db)
        assert [r["nap_time_seconds"] for r in rows] == [1800, 900]
        assert rows[0]["calendar_date"] == DAY

    def test_sleep_sync_without_naps_writes_zero_and_no_rows(
        self, tmp_path: Path
    ) -> None:
        manager = self._build_manager(tmp_path, make_sleep())

        manager._sync_metric_for_date(1, DAY, MetricType.SLEEP, self._stats())

        (daily,) = manager.db.get_health_metrics(1, DAY, DAY)
        assert daily["nap_duration_hours"] == 0.0
        assert daily["nap_count"] == 0
        assert stored_naps(manager.db) == []

    def test_resync_replaces_naps(self, tmp_path: Path) -> None:
        manager = self._build_manager(
            tmp_path, make_sleep([make_nap(), make_second_nap()])
        )
        manager._sync_metric_for_date(1, DAY, MetricType.SLEEP, self._stats())

        # Force re-sync (as --resync-days does) with the first nap gone
        manager.db.reset_completed_statuses(1, DAY, DAY)
        manager.api_client.metrics.get.return_value.get.return_value = make_sleep(
            [make_second_nap()]
        )
        manager._sync_metric_for_date(1, DAY, MetricType.SLEEP, self._stats())

        (daily,) = manager.db.get_health_metrics(1, DAY, DAY)
        assert daily["nap_count"] == 1
        assert daily["nap_duration_hours"] == pytest.approx(0.25)
        assert [r["nap_time_seconds"] for r in stored_naps(manager.db)] == [900]

    def test_completed_day_is_skipped_without_fetch(self, tmp_path: Path) -> None:
        manager = self._build_manager(tmp_path, make_sleep([make_nap()]))
        manager._sync_metric_for_date(1, DAY, MetricType.SLEEP, self._stats())
        manager.db.store_sleep_naps = MagicMock()  # type: ignore[method-assign]
        stats = self._stats()

        manager._sync_metric_for_date(1, DAY, MetricType.SLEEP, stats)

        assert stats["skipped"] == 1
        manager.db.store_sleep_naps.assert_not_called()

    def test_nap_store_failure_marks_day_failed(self, tmp_path: Path) -> None:
        manager = self._build_manager(tmp_path, make_sleep([make_nap()]))
        manager.db.store_sleep_naps = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("boom")
        )
        stats = self._stats()

        manager._sync_metric_for_date(1, DAY, MetricType.SLEEP, stats)

        assert stats["failed"] == 1
        assert stats["completed"] == 0
        assert manager.db.get_sync_status(1, DAY, MetricType.SLEEP) == "failed"

    def test_no_data_does_not_touch_naps(self, tmp_path: Path) -> None:
        manager = self._build_manager(tmp_path, None)
        manager.db.store_sleep_naps = MagicMock()  # type: ignore[method-assign]
        stats = self._stats()

        manager._sync_metric_for_date(1, DAY, MetricType.SLEEP, stats)

        assert stats["skipped"] == 1
        manager.db.store_sleep_naps.assert_not_called()

    def test_non_sleep_metric_never_writes_naps(self, tmp_path: Path) -> None:
        manager = self._build_manager(tmp_path, None, metric_type=MetricType.STEPS)
        manager.db.store_sleep_naps = MagicMock()  # type: ignore[method-assign]

        manager._sync_metric_for_date(1, DAY, MetricType.STEPS, self._stats())

        manager.db.store_sleep_naps.assert_not_called()

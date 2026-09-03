"""Datetime conversions must not depend on the host timezone.

Garmin ``*Local`` epochs are already shifted by the device's UTC offset, so
converting them with the host zone double-shifts them (a 22:00 bedtime became
15:00 on a UTC-7 machine), and ``*GMT`` epochs rendered as host-local time
under a GMT name. Every conversion, in memory and when persisting, now goes
through ``epoch_ms_to_utc_datetime``: GMT epochs give naive UTC and Local
epochs give the device wall clock, on any host.
"""

from datetime import datetime
from pathlib import Path
from typing import Callable

import pytest

from garmy.core.utils import TimestampMixin, epoch_ms_to_utc_datetime
from garmy.localdb.db import HealthDB
from garmy.localdb.extractors import DataExtractor
from garmy.localdb.models import BodyComposition
from garmy.metrics.body_battery import BodyBatteryReading
from garmy.metrics.sleep import Sleep, SleepSummary
from garmy.metrics.stress import StressReading

# 2026-05-01 22:00:00 read as UTC (a device-local "bedtime" epoch) and the
# matching 06:30 wake-up; both are invented values.
BEDTIME_LOCAL_MS = 1777672800000
WAKE_LOCAL_MS = 1777703400000
EPOCH_2022_MS = 1640995200000

ZONES = ["UTC", "America/Los_Angeles", "Asia/Tokyo"]


class TestEpochHelper:
    """epoch_ms_to_utc_datetime is host-timezone independent."""

    @pytest.mark.parametrize("zone", ZONES)
    def test_same_result_in_every_zone(
        self, set_timezone: Callable[[str], None], zone: str
    ) -> None:
        set_timezone(zone)

        assert epoch_ms_to_utc_datetime(EPOCH_2022_MS) == datetime(2022, 1, 1)
        assert epoch_ms_to_utc_datetime(0) == datetime(1970, 1, 1)

    def test_result_is_naive(self) -> None:
        assert epoch_ms_to_utc_datetime(EPOCH_2022_MS).tzinfo is None

    @pytest.mark.parametrize("zone", ZONES)
    def test_mixin_is_host_independent(
        self, set_timezone: Callable[[str], None], zone: str
    ) -> None:
        """The mixin is the same conversion, so in-memory values match the DB."""
        set_timezone(zone)

        assert TimestampMixin.timestamp_to_datetime(EPOCH_2022_MS) == datetime(
            2022, 1, 1
        )


class TestMetricDatetimeProperties:
    """Dataclass datetime properties are UTC (GMT epochs) or device wall clock."""

    @pytest.mark.parametrize("zone", ZONES)
    def test_sleep_summary_gmt_and_local(
        self, set_timezone: Callable[[str], None], zone: str
    ) -> None:
        set_timezone(zone)
        # Device at UTC-7: 22:00 local on 2026-05-01 == 05:00 UTC on 2026-05-02
        summary = SleepSummary(
            sleep_start_timestamp_gmt=BEDTIME_LOCAL_MS + 7 * 3600 * 1000,
            sleep_start_timestamp_local=BEDTIME_LOCAL_MS,
        )

        assert summary.sleep_start_datetime_gmt == datetime(2026, 5, 2, 5, 0)
        assert summary.sleep_start_datetime_local == datetime(2026, 5, 1, 22, 0)

    @pytest.mark.parametrize("zone", ZONES)
    def test_reading_datetimes_are_utc(
        self, set_timezone: Callable[[str], None], zone: str
    ) -> None:
        set_timezone(zone)

        battery = BodyBatteryReading(
            timestamp=EPOCH_2022_MS, level=50, status="CHARGING", version=1.0
        )
        stress = StressReading(timestamp=EPOCH_2022_MS, stress_level=25)

        assert battery.datetime == datetime(2022, 1, 1)
        assert stress.datetime == datetime(2022, 1, 1)


class TestSleepBedtimeExtraction:
    """sleep_bedtime / sleep_wake_time are the device wall clock everywhere."""

    @pytest.mark.parametrize("zone", ZONES)
    def test_bedtime_and_wake_time_are_device_wall_clock(
        self, set_timezone: Callable[[str], None], zone: str
    ) -> None:
        set_timezone(zone)
        sleep = Sleep(
            sleep_summary=SleepSummary(
                sleep_time_seconds=28800,
                sleep_start_timestamp_local=BEDTIME_LOCAL_MS,
                sleep_end_timestamp_local=WAKE_LOCAL_MS,
            )
        )

        result = DataExtractor()._extract_sleep_data(sleep)

        assert result["sleep_bedtime"] == "2026-05-01T22:00:00"
        assert result["sleep_wake_time"] == "2026-05-02T06:30:00"

    def test_missing_timestamps_stay_none(self) -> None:
        sleep = Sleep(sleep_summary=SleepSummary(sleep_time_seconds=28800))

        result = DataExtractor()._extract_sleep_data(sleep)

        assert result["sleep_bedtime"] is None
        assert result["sleep_wake_time"] is None


class TestBodyCompositionTimestamp:
    """body_composition.timestamp_gmt stores true UTC in every host zone."""

    @pytest.mark.parametrize("zone", ZONES)
    def test_timestamp_gmt_is_utc(
        self, tmp_path: Path, set_timezone: Callable[[str], None], zone: str
    ) -> None:
        set_timezone(zone)
        db = HealthDB(tmp_path / "bc.db")

        db.store_body_composition(
            1,
            {
                "sample_pk": 1001,
                "measurement_date": "2022-01-01",
                "timestamp_gmt": EPOCH_2022_MS,
                "weight_grams": 70000,
            },
        )

        with db.get_session() as session:
            stored = session.query(BodyComposition).one().timestamp_gmt
        assert stored == datetime(2022, 1, 1, 0, 0)

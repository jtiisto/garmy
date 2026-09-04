"""Sleep Data Module.

==================

This module provides direct access to Garmin sleep data from the Connect API.
Data includes comprehensive sleep metrics, stages, SpO2, respiration, and detailed
temporal readings throughout the night.

Example:
    >>> from garmy import AuthClient, APIClient, MetricAccessorFactory
    >>> auth_client = AuthClient()
    >>> api_client = APIClient(auth_client=auth_client)
    >>> auth_client.login("email@example.com", "password")
    >>>
    >>> # Get today's sleep data
    >>> factory = MetricAccessorFactory(api_client)
    >>> metrics = factory.discover_and_create_all()
    >>> sleep = metrics.get("sleep").get()
    >>> print(f"Sleep duration: {sleep.sleep_duration_hours:.1f} hours")
    >>> print(f"Deep sleep: {sleep.deep_sleep_percentage:.1f}%")
    >>> print(f"SpO2 average: {sleep.daily_sleep_dto.average_sp_o2_value}%")
    >>>
    >>> # Naps are reported separately from the main sleep window
    >>> print(f"Naps: {sleep.nap_count}, total {sleep.nap_duration_hours:.2f} h")
    >>> for nap in sleep.naps:
    ...     print(nap.nap_start_datetime_local, nap.nap_duration_minutes)

Data Source:
    Garmin Connect API endpoint: /sleep-service/sleep/dailySleepData
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from datetime import date

from ..core.base import MetricConfig
from ..core.endpoint_builders import build_sleep_endpoint as _build_sleep_endpoint
from ..core.utils import (
    TimestampMixin,
    create_nested_summary_parser,
)


@dataclass
class SleepSummary(TimestampMixin):
    """Main sleep data structure from Garmin API."""

    # Core sleep timing
    id: int = 0
    user_profile_pk: int = 0
    calendar_date: str = ""
    sleep_time_seconds: int = 0
    nap_time_seconds: int = 0
    sleep_start_timestamp_gmt: int = 0
    sleep_end_timestamp_gmt: int = 0
    sleep_start_timestamp_local: int = 0
    sleep_end_timestamp_local: int = 0

    # Sleep stages
    deep_sleep_seconds: int = 0
    light_sleep_seconds: int = 0
    rem_sleep_seconds: int = 0
    awake_sleep_seconds: int = 0
    unmeasurable_sleep_seconds: int = 0
    awake_count: int = 0

    # Sleep quality
    sleep_window_confirmed: bool = False
    sleep_window_confirmation_type: str = ""
    device_rem_capable: bool = False
    retro: bool = False
    sleep_from_device: bool = False

    # Physiological measurements
    average_sp_o2_value: Optional[int] = None
    lowest_sp_o2_value: Optional[int] = None
    highest_sp_o2_value: Optional[int] = None
    average_sp_o2_hr_sleep: Optional[int] = None
    average_respiration_value: Optional[float] = None
    lowest_respiration_value: Optional[float] = None
    highest_respiration_value: Optional[float] = None
    avg_sleep_stress: Optional[float] = None

    # Optional metadata
    auto_sleep_start_timestamp_gmt: Optional[int] = None
    auto_sleep_end_timestamp_gmt: Optional[int] = None
    sleep_quality_type_pk: Optional[int] = None
    sleep_result_type_pk: Optional[int] = None
    age_group: Optional[str] = None
    sleep_score_feedback: Optional[str] = None
    sleep_score_insight: Optional[str] = None
    sleep_score_personalized_insight: Optional[str] = None
    sleep_version: Optional[int] = None

    # Nested objects as raw dicts (following garmy philosophy)
    sleep_scores: Optional[Dict[str, Any]] = None
    sleep_need: Optional[Dict[str, Any]] = None
    next_sleep_need: Optional[Dict[str, Any]] = None

    # Per-nap entries (dailyNapDTOS, snake_cased); only present on nap days.
    # Typed access via Sleep.naps.
    daily_nap_dtos: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def sleep_start_datetime_gmt(self) -> "datetime":
        """Sleep start as a naive UTC datetime (host-timezone independent)."""
        return self.timestamp_to_datetime(self.sleep_start_timestamp_gmt)

    @property
    def sleep_end_datetime_gmt(self) -> "datetime":
        """Sleep end as a naive UTC datetime (host-timezone independent)."""
        return self.timestamp_to_datetime(self.sleep_end_timestamp_gmt)

    @property
    def sleep_start_datetime_local(self) -> "datetime":
        """Sleep start as the device's wall-clock time (naive, offset applied by Garmin)."""
        return self.timestamp_to_datetime(self.sleep_start_timestamp_local)

    @property
    def sleep_end_datetime_local(self) -> "datetime":
        """Sleep end as the device's wall-clock time (naive, offset applied by Garmin)."""
        return self.timestamp_to_datetime(self.sleep_end_timestamp_local)

    @property
    def total_sleep_duration_hours(self) -> Optional[float]:
        """Get total sleep duration in hours."""
        if self.sleep_time_seconds is None:
            return None
        return self.sleep_time_seconds / 3600

    @property
    def sleep_efficiency_percentage(self) -> Optional[float]:
        """Calculate sleep efficiency (sleep time / time in bed)."""
        if (
            self.sleep_end_timestamp_local is None
            or self.sleep_start_timestamp_local is None
            or self.sleep_time_seconds is None
        ):
            return None
        time_in_bed = (
            self.sleep_end_timestamp_local - self.sleep_start_timestamp_local
        ) / 1000
        if time_in_bed > 0:
            return (self.sleep_time_seconds / time_in_bed) * 100
        return 0


@dataclass
class SleepNap(TimestampMixin):
    """A single nap from dailySleepDTO.dailyNapDTOS.

    Garmin reports naps separately from the main sleep window:
    ``SleepSummary.sleep_time_seconds`` excludes nap time and
    ``SleepSummary.nap_time_seconds`` is the daily nap total.

    Attributes:
        nap_time_sec: Nap duration in seconds (equals end - start)
        nap_start_timestamp_gmt: ISO string in GMT, e.g. "2026-01-15T13:05:00"
        nap_end_timestamp_gmt: ISO string in GMT
        nap_start_time_offset: Local UTC offset in seconds at nap start
        nap_end_time_offset: Local UTC offset in seconds at nap end
        nap_feedback: Garmin feedback enum, e.g. IDEAL_TIMING_IDEAL_DURATION_LOW_NEED
        nap_source: Source code (0 observed for device-detected naps)
        device_id: Recording device id
        calendar_date: Sleep-service day the nap was reported under
        user_profile_pk: Garmin user profile key
    """

    nap_time_sec: int = 0
    nap_start_timestamp_gmt: str = ""
    nap_end_timestamp_gmt: str = ""
    nap_start_time_offset: int = 0
    nap_end_time_offset: int = 0
    nap_feedback: Optional[str] = None
    nap_source: Optional[int] = None
    device_id: Optional[int] = None
    calendar_date: str = ""
    user_profile_pk: int = 0

    @property
    def nap_start_datetime_gmt(self) -> Optional[datetime]:
        """Nap start as a naive GMT datetime (None if unparseable)."""
        return self.iso_to_datetime(self.nap_start_timestamp_gmt)

    @property
    def nap_end_datetime_gmt(self) -> Optional[datetime]:
        """Nap end as a naive GMT datetime (None if unparseable)."""
        return self.iso_to_datetime(self.nap_end_timestamp_gmt)

    @property
    def nap_start_datetime_local(self) -> Optional[datetime]:
        """Nap start in the device's local time (GMT + offset)."""
        start = self.nap_start_datetime_gmt
        if start is None:
            return None
        return start + timedelta(seconds=self.nap_start_time_offset or 0)

    @property
    def nap_end_datetime_local(self) -> Optional[datetime]:
        """Nap end in the device's local time (GMT + offset)."""
        end = self.nap_end_datetime_gmt
        if end is None:
            return None
        return end + timedelta(seconds=self.nap_end_time_offset or 0)

    @property
    def nap_duration_minutes(self) -> float:
        """Nap duration in minutes."""
        return (self.nap_time_sec or 0) / 60


@dataclass
class Sleep:
    """Comprehensive sleep data from Garmin Connect API.

    Raw sleep data including detailed sleep stages, SpO2, respiration, and
    temporal readings throughout the night. All data comes directly from
    Garmin's sleep service.

    Attributes:
        sleep_summary: Main sleep summary with stages, timing, and scores
        sleep_movement: Raw movement data throughout the night (list of dicts)
        wellness_epoch_spo2_data_dto_list: SpO2 readings throughout the night (list of dicts)
        wellness_epoch_respiration_data_dto_list: Respiration readings throughout the night
            (list of dicts)
        skin_temp_data_exists: Whether skin temperature data is available
        skin_temp_deviation_c: Skin temperature deviation in Celsius
        skin_temp_deviation_f: Skin temperature deviation in Fahrenheit
        naps: Individual naps for the day (list of SleepNap); empty on nap-free
            days. Nap time is excluded from sleep_duration_hours — see
            nap_duration_hours and total_sleep_with_naps_hours.

    Example:
        >>> sleep = garmy.sleep.get()
        >>> print(f"Sleep duration: {sleep.sleep_duration_hours:.1f} hours")
        >>> print(f"Deep sleep: {sleep.deep_sleep_percentage:.1f}%")
        >>> print(f"Average SpO2: {sleep.sleep_summary.average_sp_o2_value}%")
        >>>
        >>> # Access raw SpO2 readings
        >>> for reading in sleep.wellness_epoch_spo2_data_dto_list[:5]:
        >>>     print(f"SpO2: {reading['value']}% at {reading['startGMT']}")
    """

    sleep_summary: SleepSummary
    sleep_movement: List[Dict[str, Any]] = field(default_factory=list)
    wellness_epoch_spo2_data_dto_list: List[Dict[str, Any]] = field(
        default_factory=list
    )
    wellness_epoch_respiration_data_dto_list: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # Top-level skin temperature fields
    skin_temp_data_exists: bool = False
    skin_temp_deviation_c: Optional[float] = None
    skin_temp_deviation_f: Optional[float] = None

    # Typed naps built from sleep_summary.daily_nap_dtos (empty on nap-free days)
    naps: List[SleepNap] = field(default_factory=list)

    def __str__(self) -> str:
        """Format sleep data for human-readable display."""
        lines = []
        if self.sleep_duration_hours:
            lines.append(f"• Duration: {self.sleep_duration_hours:.1f} hours")
        if self.deep_sleep_percentage:
            lines.append(f"• Deep sleep: {self.deep_sleep_percentage:.1f}%")
        if self.light_sleep_percentage:
            lines.append(f"• Light sleep: {self.light_sleep_percentage:.1f}%")
        if self.rem_sleep_percentage:
            lines.append(f"• REM sleep: {self.rem_sleep_percentage:.1f}%")
        if self.awake_percentage:
            lines.append(f"• Awake: {self.awake_percentage:.1f}%")
        if self.sleep_summary.average_sp_o2_value:
            lines.append(f"• Average SpO2: {self.sleep_summary.average_sp_o2_value}%")
        if self.sleep_summary.average_respiration_value:
            lines.append(
                f"• Respiration: {self.sleep_summary.average_respiration_value:.1f} breaths/min"
            )
        if self.sleep_summary.awake_count:
            lines.append(f"• Awakenings: {self.sleep_summary.awake_count}")
        if self.nap_count or self.nap_duration_hours:
            lines.append(
                f"• Naps: {self.nap_count} ({self.nap_duration_hours * 60:.0f} min)"
            )

        # Add data availability info
        data_counts = []
        if self.spo2_readings_count:
            data_counts.append(f"{self.spo2_readings_count} SpO2 readings")
        if self.respiration_readings_count:
            data_counts.append(
                f"{self.respiration_readings_count} respiration readings"
            )
        if self.movement_readings_count:
            data_counts.append(f"{self.movement_readings_count} movement readings")

        if data_counts:
            lines.append(f"• Data available: {', '.join(data_counts)}")

        return "\n".join(lines) if lines else "Sleep data available"

    @property
    def sleep_duration_hours(self) -> Optional[float]:
        """Get total sleep duration in hours."""
        return self.sleep_summary.total_sleep_duration_hours

    @property
    def deep_sleep_percentage(self) -> Optional[float]:
        """Get deep sleep as percentage of total sleep."""
        total = self.sleep_summary.sleep_time_seconds
        deep = self.sleep_summary.deep_sleep_seconds
        if total and total > 0 and deep is not None:
            return (deep / total) * 100
        return None

    @property
    def light_sleep_percentage(self) -> Optional[float]:
        """Get light sleep as percentage of total sleep."""
        total = self.sleep_summary.sleep_time_seconds
        light = self.sleep_summary.light_sleep_seconds
        if total and total > 0 and light is not None:
            return (light / total) * 100
        return None

    @property
    def rem_sleep_percentage(self) -> Optional[float]:
        """Get REM sleep as percentage of total sleep."""
        total = self.sleep_summary.sleep_time_seconds
        rem = self.sleep_summary.rem_sleep_seconds
        if total and total > 0 and rem is not None:
            return (rem / total) * 100
        return None

    @property
    def awake_percentage(self) -> Optional[float]:
        """Get awake time as percentage of total sleep period."""
        total = self.sleep_summary.sleep_time_seconds
        awake = self.sleep_summary.awake_sleep_seconds
        if total and total > 0 and awake is not None:
            return (awake / total) * 100
        return None

    @property
    def spo2_readings_count(self) -> int:
        """Get number of SpO2 readings."""
        return len(self.wellness_epoch_spo2_data_dto_list)

    @property
    def respiration_readings_count(self) -> int:
        """Get number of respiration readings."""
        return len(self.wellness_epoch_respiration_data_dto_list)

    @property
    def movement_readings_count(self) -> int:
        """Get number of movement readings."""
        return len(self.sleep_movement)

    @property
    def nap_count(self) -> int:
        """Get number of naps reported for the day."""
        return len(self.naps)

    @property
    def nap_duration_hours(self) -> float:
        """Get total nap time in hours (excluded from sleep_duration_hours)."""
        return (self.sleep_summary.nap_time_seconds or 0) / 3600

    @property
    def total_sleep_with_naps_hours(self) -> Optional[float]:
        """Get main sleep plus naps in hours."""
        main = self.sleep_duration_hours
        if main is None:
            return None
        return main + self.nap_duration_hours


def parse_sleep_data(data: Dict[str, Any]) -> Sleep:
    """Parse sleep data including top-level skin temperature fields.

    This custom parser extends the standard nested summary parser to also
    capture skin temperature data that exists at the top level of the
    API response (not nested in dailySleepDTO).
    """
    # Use existing parser for nested summary and base data
    base_parser = create_nested_summary_parser(
        Sleep,
        SleepSummary,
        "daily_sleep_dto",
        [
            "sleep_movement",
            "wellness_epoch_spo2_data_dto_list",
            "wellness_epoch_respiration_data_dto_list",
        ],
    )

    # Parse base data
    sleep = base_parser(data)

    # Add top-level skin temp fields (these are outside dailySleepDTO)
    sleep.skin_temp_data_exists = data.get("skinTempDataExists", False)
    sleep.skin_temp_deviation_c = data.get("avgSkinTempDeviationC")
    sleep.skin_temp_deviation_f = data.get("avgSkinTempDeviationF")

    # Build typed naps from the raw (snake_cased) dailyNapDTOS entries, keeping
    # only declared SleepNap fields so extra or future API keys are ignored.
    raw_naps = getattr(sleep.sleep_summary, "daily_nap_dtos", None)
    if not isinstance(raw_naps, list):
        raw_naps = []
    nap_fields = set(SleepNap.__dataclass_fields__)
    sleep.naps = [
        SleepNap(**{k: v for k, v in nap.items() if k in nap_fields})
        for nap in raw_naps
        if isinstance(nap, dict)
    ]

    return sleep


def build_sleep_endpoint(
    date_input: Union["date", str, None] = None, api_client: Any = None, **kwargs: Any
) -> str:
    """Build the Sleep API endpoint with user ID and date."""
    return _build_sleep_endpoint(date_input, api_client, **kwargs)


# MetricConfig for auto-discovery
METRIC_CONFIG = MetricConfig(
    endpoint="",
    metric_class=Sleep,
    parser=parse_sleep_data,
    endpoint_builder=build_sleep_endpoint,
    requires_user_id=True,
    description="Comprehensive sleep data including stages, SpO2, respiration, and movement",
    version="1.0",
)

__metric_config__ = METRIC_CONFIG

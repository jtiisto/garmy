"""Shared pytest fixtures."""

import os
import time
from typing import Callable, Iterator, Optional

import pytest


@pytest.fixture
def set_timezone() -> Iterator[Callable[[str], None]]:
    """Return a callable that switches the process timezone for one test.

    Used to prove that datetime conversions and persisted timestamps do not
    depend on the host zone (CI runs in UTC, developers usually do not). The
    original TZ is restored on teardown.
    """
    if not hasattr(time, "tzset"):  # pragma: no cover - Windows
        pytest.skip("time.tzset() is not available on this platform")

    original: Optional[str] = os.environ.get("TZ")

    def _set(tz_name: str) -> None:
        os.environ["TZ"] = tz_name
        time.tzset()

    yield _set

    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()

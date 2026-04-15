"""
Session Manager — Trading session windows with DST support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timezone, timedelta
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TradingSession(str, Enum):
    AMERICAN = "AMERICAN"
    NYSE = "NYSE"
    EUROPEAN = "EUROPEAN"
    LONDON = "LONDON"
    ASIAN = "ASIAN"
    CUSTOM = "CUSTOM"
    ALL_DAY = "ALL_DAY"


class DSTMode(str, Enum):
    NO_DST = "NO_DST"
    NORTH_AMERICA = "NORTH_AMERICA"
    EUROPE = "EUROPE"


# Session windows in UTC (start_hour, start_min, end_hour, end_min)
_SESSION_WINDOWS_UTC = {
    TradingSession.AMERICAN:  (13, 0, 22, 0),
    TradingSession.NYSE:      (14, 30, 21, 0),
    TradingSession.EUROPEAN:  (7, 0, 16, 0),
    TradingSession.LONDON:    (8, 0, 17, 0),
    TradingSession.ASIAN:     (0, 0, 9, 0),
    TradingSession.ALL_DAY:   (0, 0, 23, 59),
}


@dataclass
class CustomSessionConfig:
    start_hour: int = 8
    start_minute: int = 0
    end_hour: int = 17
    end_minute: int = 0


class SessionManager:
    """
    Determines whether current UTC time falls within the configured trading session.

    Parameters
    ----------
    session : TradingSession
    dst_mode : DSTMode
    gmt_offset : float  Broker GMT offset in hours (e.g. +2 for EET)
    custom_config : CustomSessionConfig  Used only when session=CUSTOM
    """

    def __init__(
        self,
        session: TradingSession = TradingSession.LONDON,
        dst_mode: DSTMode = DSTMode.NO_DST,
        gmt_offset: float = 0.0,
        custom_config: Optional[CustomSessionConfig] = None,
    ) -> None:
        self.session = session
        self.dst_mode = dst_mode
        self.gmt_offset = gmt_offset
        self.custom_config = custom_config or CustomSessionConfig()

    def is_trading_time(self, dt: Optional[datetime] = None) -> bool:
        """Returns True if dt (UTC) is within the active session window."""
        if dt is None:
            dt = datetime.now(tz=timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Apply broker GMT offset
        local_dt = dt + timedelta(hours=self.gmt_offset)

        # Apply DST adjustment
        dst_offset = self._get_dst_offset(local_dt)
        adjusted_dt = local_dt + timedelta(hours=dst_offset)

        start, end = self.get_range_start_end()
        current_time = adjusted_dt.time().replace(second=0, microsecond=0)

        if start <= end:
            return start <= current_time <= end
        else:
            # Overnight session (e.g. 22:00 – 05:00)
            return current_time >= start or current_time <= end

    def get_range_start_end(
        self, monitoring_minutes: int = 0
    ) -> Tuple[time, time]:
        """
        Returns (session_start, session_end) as time objects.
        If monitoring_minutes > 0, returns the range-monitoring sub-window.
        """
        if self.session == TradingSession.CUSTOM:
            cfg = self.custom_config
            start = time(cfg.start_hour, cfg.start_minute)
            end = time(cfg.end_hour, cfg.end_minute)
        else:
            w = _SESSION_WINDOWS_UTC.get(self.session, (0, 0, 23, 59))
            start = time(w[0], w[1])
            end = time(w[2], w[3])

        if monitoring_minutes > 0:
            # Monitoring window starts at session open
            start_dt = datetime(2000, 1, 1, start.hour, start.minute)
            mon_end_dt = start_dt + timedelta(minutes=monitoring_minutes)
            return start, mon_end_dt.time()

        return start, end

    def is_range_forming(
        self, dt: Optional[datetime] = None, monitoring_minutes: int = 60
    ) -> bool:
        """True during the range-monitoring sub-window at session open."""
        if dt is None:
            dt = datetime.now(tz=timezone.utc)
        session_start, range_end = self.get_range_start_end(monitoring_minutes)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local_dt = dt + timedelta(hours=self.gmt_offset)
        t = local_dt.time().replace(second=0, microsecond=0)
        return session_start <= t <= range_end

    # ------------------------------------------------------------------ #
    #  DST helpers                                                         #
    # ------------------------------------------------------------------ #

    def _get_dst_offset(self, local_dt: datetime) -> float:
        if self.dst_mode == DSTMode.NO_DST:
            return 0.0
        elif self.dst_mode == DSTMode.NORTH_AMERICA:
            return self._na_dst_offset(local_dt)
        elif self.dst_mode == DSTMode.EUROPE:
            return self._eu_dst_offset(local_dt)
        return 0.0

    @staticmethod
    def _na_dst_offset(dt: datetime) -> float:
        """North America DST: 2nd Sun March → 1st Sun November."""
        year = dt.year
        dst_start = SessionManager._nth_weekday(year, 3, 6, 2)  # 2nd Sunday March
        dst_end = SessionManager._nth_weekday(year, 11, 6, 1)   # 1st Sunday November
        if dst_start <= dt.replace(tzinfo=None) < dst_end:
            return 1.0
        return 0.0

    @staticmethod
    def _eu_dst_offset(dt: datetime) -> float:
        """European DST: Last Sun March → Last Sun October."""
        year = dt.year
        dst_start = SessionManager._last_weekday(year, 3, 6)    # Last Sunday March
        dst_end = SessionManager._last_weekday(year, 10, 6)     # Last Sunday October
        if dst_start <= dt.replace(tzinfo=None) < dst_end:
            return 1.0
        return 0.0

    @staticmethod
    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
        """Return the nth weekday (0=Mon, 6=Sun) in given month."""
        d = datetime(year, month, 1)
        days_ahead = weekday - d.weekday()
        if days_ahead < 0:
            days_ahead += 7
        d = d + timedelta(days=days_ahead + (n - 1) * 7)
        return d

    @staticmethod
    def _last_weekday(year: int, month: int, weekday: int) -> datetime:
        """Return the last weekday in given month."""
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        d = next_month - timedelta(days=1)
        days_back = (d.weekday() - weekday) % 7
        return d - timedelta(days=days_back)

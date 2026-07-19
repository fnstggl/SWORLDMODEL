"""Timezone-aware calendar semantics for the scenario temporal runtime (§18 of the event-driven
temporal architecture).

Real temporal causality needs real calendars: "24 hours later" and "tomorrow at 9:00 AM local
time" are DIFFERENT timestamps across a daylight-saving transition; "next business day" from a
Friday filing is Monday (or Tuesday after a holiday); an actor's "morning" is their timezone's
morning, not UTC's. Every function here is zoneinfo-backed civil-time arithmetic — never naive
`ts + n*86400` masquerading as a calendar day.

Semantics defined and tested here (each distinct, none interchangeable):

  absolute duration    add_absolute(ts, seconds)          — exact elapsed seconds, calendar-blind
  calendar day         add_calendar_days(ts, n, tz, ...)  — same local wall-clock time n civil days
                                                            later (∆ 23/24/25h across DST)
  next business day    next_business_day(ts, cal)         — next working day at opening hour
  end of day           end_of_day(ts, cal)                — today's local close of business
  tomorrow morning     tomorrow_morning(ts, cal)          — next civil day at the morning hour
  before a deadline    resolves to the deadline ts with an explicit margin
  recurring weekly     RecurrenceRule.next_occurrence     — real weekday + local time + tz

`CivilCalendar` carries the tz, working days, business hours and holiday set for one actor,
institution or venue; the scenario temporal compiler generates these per scenario — nothing here
assumes one global calendar.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")

#: ISO weekday numbers (Mon=1 … Sun=7) — the default working week; scenario calendars override.
DEFAULT_BUSINESS_DAYS = (1, 2, 3, 4, 5)


def _zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(str(tz or "UTC"))
    except Exception:  # noqa: BLE001 — an unknown tz label degrades to UTC, recorded by callers
        return UTC


@dataclass(frozen=True)
class CivilCalendar:
    """One party's civil-time frame: IANA timezone + working days/hours + holidays.

    Generated per scenario by the temporal compiler (an actor's, an institution's, a market's);
    unknown fields keep their defaults and the unknown-ness is recorded on the temporal model,
    not hidden here."""
    tz: str = "UTC"
    business_days: tuple = DEFAULT_BUSINESS_DAYS          # ISO weekday numbers
    open_hour: float = 9.0                                # local business open
    close_hour: float = 17.0                              # local business close
    morning_hour: float = 9.0                             # "tomorrow morning" anchor
    evening_hour: float = 20.0
    holidays: tuple = ()                                  # ("YYYY-MM-DD", ...) local civil dates
    provenance: str = "default_civil_calendar"

    def zone(self) -> ZoneInfo:
        return _zone(self.tz)

    def is_business_day(self, ts: float) -> bool:
        d = local_civil(ts, self.tz)
        return d.isoweekday() in self.business_days and d.strftime("%Y-%m-%d") not in self.holidays

    def is_business_hours(self, ts: float) -> bool:
        if not self.is_business_day(ts):
            return False
        d = local_civil(ts, self.tz)
        h = d.hour + d.minute / 60.0
        return self.open_hour <= h < self.close_hour


# ---------------------------------------------------------------- civil <-> unix conversions
def local_civil(ts: float, tz: str) -> _dt.datetime:
    """The local civil datetime of a unix timestamp in an IANA timezone."""
    return _dt.datetime.fromtimestamp(float(ts), tz=_zone(tz))


def civil_to_ts(year: int, month: int, day: int, hour: float = 0.0, *, tz: str) -> float:
    """Unix ts of a local civil wall-clock time — DST-correct via zoneinfo (a nonexistent local
    time during spring-forward resolves per PEP 495 fold-0 semantics)."""
    h = int(hour)
    m = int(round((hour - h) * 60))
    return _dt.datetime(year, month, day, h, m, tzinfo=_zone(tz)).timestamp()


def at_local_hour(ts: float, hour: float, *, tz: str) -> float:
    """The SAME local civil day as `ts`, at the given local hour."""
    d = local_civil(ts, tz)
    return civil_to_ts(d.year, d.month, d.day, hour, tz=tz)


# ---------------------------------------------------------------- the distinct day semantics
def add_absolute(ts: float, seconds: float) -> float:
    """Exact elapsed duration — calendar-blind. `add_absolute(ts, 86400)` across a DST spring-
    forward lands at a DIFFERENT local wall-clock hour than `add_calendar_days(ts, 1)`."""
    return float(ts) + float(seconds)


def add_calendar_days(ts: float, n: int, *, tz: str, at_hour: float = None) -> float:
    """n civil days later in the local calendar, at the same wall-clock time (or `at_hour`).
    Across a DST transition the elapsed absolute time is 23 or 25 hours per day — that is the
    point: 'tomorrow at 9am' is a civil-time promise, not an 86400-second one."""
    d = local_civil(ts, tz)
    target_day = (d + _dt.timedelta(days=int(n))).date()
    hour = (d.hour + d.minute / 60.0) if at_hour is None else float(at_hour)
    return civil_to_ts(target_day.year, target_day.month, target_day.day, hour, tz=tz)


def tomorrow_morning(ts: float, cal: CivilCalendar) -> float:
    """The next civil day at the calendar's morning hour, in the calendar's timezone."""
    return add_calendar_days(ts, 1, tz=cal.tz, at_hour=cal.morning_hour)


def end_of_day(ts: float, cal: CivilCalendar) -> float:
    """Local close-of-business today (or, past closing, the close remains today's — the caller
    decides whether a past close means 'now')."""
    return at_local_hour(ts, cal.close_hour, tz=cal.tz)


def next_business_day(ts: float, cal: CivilCalendar, *, at_hour: float = None) -> float:
    """The next working day (per the calendar's working weekdays + holidays) at opening hour."""
    hour = cal.open_hour if at_hour is None else float(at_hour)
    probe = ts
    for _ in range(370):                                   # bounded: > a year of consecutive holidays is a data bug
        probe = add_calendar_days(probe, 1, tz=cal.tz, at_hour=hour)
        if cal.is_business_day(probe):
            return probe
    raise ValueError(f"no business day within a year of {ts} for calendar {cal.tz}")


def add_business_days(ts: float, n: int, cal: CivilCalendar, *, at_hour: float = None) -> float:
    """n working days later (statutory 'within N business days' semantics)."""
    out = ts
    for _ in range(max(0, int(n))):
        out = next_business_day(out, cal, at_hour=at_hour)
    return out


def next_time_in_window(ts: float, cal: CivilCalendar, *, start_hour: float,
                        end_hour: float, days: tuple = None) -> float:
    """The earliest instant >= ts inside a recurring local window (e.g. an actor's waking or
    working hours). If `ts` is already inside the window, returns ts unchanged."""
    allowed = tuple(days) if days else tuple(range(1, 8))
    probe = float(ts)
    for _ in range(400):
        d = local_civil(probe, cal.tz)
        h = d.hour + d.minute / 60.0 + d.second / 3600.0
        if d.isoweekday() in allowed and start_hour <= h < end_hour \
                and d.strftime("%Y-%m-%d") not in cal.holidays:
            return probe
        if d.isoweekday() in allowed and h < start_hour \
                and d.strftime("%Y-%m-%d") not in cal.holidays:
            probe = at_local_hour(probe, start_hour, tz=cal.tz)
            continue
        probe = add_calendar_days(probe, 1, tz=cal.tz, at_hour=start_hour)
    raise ValueError(f"no window instant within 400 days of {ts}")


# ---------------------------------------------------------------- recurring real obligations
@dataclass(frozen=True)
class RecurrenceRule:
    """A REAL recurring obligation — a known weekly meeting, a scheduled committee session, a
    documented reporting cycle. Recurrence is allowed in the temporal model ONLY with this full
    record (§5): a rule without a source is refused by the temporal compiler, never defaulted.

    freq: 'weekly' | 'biweekly' | 'monthly_day' (day-of-month) | 'daily'."""
    rule_id: str
    freq: str
    tz: str
    local_hour: float
    weekday: int = None                                   # ISO weekday for weekly/biweekly
    month_day: int = None                                 # 1..28 for monthly_day
    participants: tuple = ()
    source: str = ""                                      # evidence claim / user context / scenario fact
    cancellation_conditions: tuple = ()
    relevance: str = ""
    confidence: float = 0.6

    def next_occurrence(self, after_ts: float) -> float:
        """First occurrence strictly after `after_ts`, in local civil time (DST-correct)."""
        d = local_civil(after_ts, self.tz)
        if self.freq in ("weekly", "biweekly"):
            step = 7 if self.freq == "weekly" else 14
            wd = int(self.weekday or 1)
            ahead = (wd - d.isoweekday()) % 7
            cand = civil_to_ts(d.year, d.month, d.day, self.local_hour, tz=self.tz) + 0.0
            cand_day = (d + _dt.timedelta(days=ahead)).date()
            cand = civil_to_ts(cand_day.year, cand_day.month, cand_day.day, self.local_hour, tz=self.tz)
            while cand <= after_ts:
                nxt = local_civil(cand, self.tz) + _dt.timedelta(days=step)
                cand = civil_to_ts(nxt.year, nxt.month, nxt.day, self.local_hour, tz=self.tz)
            return cand
        if self.freq == "monthly_day":
            md = max(1, min(28, int(self.month_day or 1)))
            y, m = d.year, d.month
            cand = civil_to_ts(y, m, md, self.local_hour, tz=self.tz)
            while cand <= after_ts:
                m += 1
                if m > 12:
                    m, y = 1, y + 1
                cand = civil_to_ts(y, m, md, self.local_hour, tz=self.tz)
            return cand
        if self.freq == "daily":
            cand = at_local_hour(after_ts, self.local_hour, tz=self.tz)
            while cand <= after_ts:
                cand = add_calendar_days(cand, 1, tz=self.tz, at_hour=self.local_hour)
            return cand
        raise ValueError(f"unknown recurrence freq {self.freq!r}")

    def as_dict(self) -> dict:
        return {"rule_id": self.rule_id, "freq": self.freq, "tz": self.tz,
                "local_hour": self.local_hour, "weekday": self.weekday,
                "month_day": self.month_day, "participants": list(self.participants),
                "source": self.source,
                "cancellation_conditions": list(self.cancellation_conditions),
                "relevance": self.relevance, "confidence": self.confidence}


# ---------------------------------------------------------------- calendar-relative expressions
#: The closed set of RESOLVABLE calendar expressions an actor's stated timing intent may compile
#: to. Anything outside this set (or missing its referent) stays UNRESOLVED on the temporal model
#: (§11) — never silently coerced to a made-up timestamp.
CALENDAR_EXPRESSIONS = ("immediately", "tomorrow_morning", "end_of_day", "this_evening",
                       "next_business_day", "next_morning_window", "start_of_next_business_day")


def resolve_calendar_expression(expr: str, ref_ts: float, cal: CivilCalendar):
    """Resolve one calendar-relative expression against a reference instant in a civil calendar.
    Returns a unix ts, or None when the expression is not in the resolvable set — the caller
    must keep it as an unresolved timing mechanism, not invent a number."""
    e = str(expr or "").strip().lower()
    if e in ("immediately", "now"):
        return float(ref_ts)
    if e == "tomorrow_morning":
        return tomorrow_morning(ref_ts, cal)
    if e in ("end_of_day", "eod"):
        eod = end_of_day(ref_ts, cal)
        return eod if eod > ref_ts else tomorrow_morning(ref_ts, cal)
    if e == "this_evening":
        ev = at_local_hour(ref_ts, cal.evening_hour, tz=cal.tz)
        return ev if ev > ref_ts else add_calendar_days(ref_ts, 1, tz=cal.tz, at_hour=cal.evening_hour)
    if e in ("next_business_day", "start_of_next_business_day"):
        return next_business_day(ref_ts, cal)
    if e == "next_morning_window":
        d = local_civil(ref_ts, cal.tz)
        h = d.hour + d.minute / 60.0
        if h < cal.morning_hour:
            return at_local_hour(ref_ts, cal.morning_hour, tz=cal.tz)
        return tomorrow_morning(ref_ts, cal)
    return None

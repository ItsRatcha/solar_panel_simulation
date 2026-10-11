"""Solar position from the NOAA Solar Calculator equations.

Accuracy is about 0.01 deg in zenith/azimuth for dates 1800-2100, which is far
better than needed for irradiance and tracker modelling.  Azimuth is measured
clockwise from north (0 = N, 90 = E, 180 = S, 270 = W).

Reference: NOAA Global Monitoring Laboratory, "Solar Calculation Details"
(after Meeus, *Astronomical Algorithms*, 1991).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

# Default site: Pathum Thani, Thailand
LATITUDE = 14.07
LONGITUDE = 100.60
TZ_HOURS = 7.0


@dataclass(frozen=True)
class Site:
    latitude: float = LATITUDE
    longitude: float = LONGITUDE
    tz_hours: float = TZ_HOURS
    altitude_m: float = 5.0


@dataclass(frozen=True)
class SunPosition:
    zenith: float  # apparent (refraction-corrected) [deg]
    azimuth: float  # [deg] clockwise from north
    elevation: float  # apparent [deg]
    declination: float  # [deg]
    eq_time_min: float  # equation of time [min]

    @property
    def is_up(self) -> bool:
        return self.elevation > 0.0


def _julian_day(dt_utc: datetime) -> float:
    y, m = dt_utc.year, dt_utc.month
    d = (dt_utc.day + (dt_utc.hour + (dt_utc.minute + (dt_utc.second
         + dt_utc.microsecond * 1e-6) / 60.0) / 60.0) / 24.0)
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def _sun_terms(jc: float) -> tuple[float, float]:
    """Declination [deg] and equation of time [min] at Julian century jc."""
    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * jc) + math.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.degrees(math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(app_long))))
    y = math.tan(math.radians(eps / 2)) ** 2
    l0r = math.radians(l0)
    eot = 4 * math.degrees(
        y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
        - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
    return decl, eot


def _refraction(elev: float) -> float:
    """Atmospheric refraction correction [deg] (NOAA approximation)."""
    if elev > 85.0:
        return 0.0
    te = math.tan(math.radians(elev))
    if elev > 5.0:
        r = 58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5
    elif elev > -0.575:
        r = 1735.0 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))
    else:
        r = -20.772 / te
    return r / 3600.0


def sun_position(when: datetime, site: Site = Site()) -> SunPosition:
    """Sun position at `when`; naive datetimes are local time of `site`."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone(timedelta(hours=site.tz_hours)))
    utc = when.astimezone(timezone.utc)
    jc = (_julian_day(utc) - 2451545.0) / 36525.0
    decl, eot = _sun_terms(jc)
    minutes_utc = utc.hour * 60 + utc.minute + (utc.second + utc.microsecond * 1e-6) / 60
    tst = (minutes_utc + eot + 4 * site.longitude) % 1440.0  # true solar time [min]
    ha = tst / 4 - 180 if tst / 4 >= 0 else tst / 4 + 180
    lat, dr, har = math.radians(site.latitude), math.radians(decl), math.radians(ha)
    cos_z = math.sin(lat) * math.sin(dr) + math.cos(lat) * math.cos(dr) * math.cos(har)
    zen = math.degrees(math.acos(max(-1.0, min(1.0, cos_z))))
    sz = math.sin(math.radians(zen))
    if abs(sz) < 1e-9:
        az = 180.0
    else:
        ca = (math.sin(lat) * cos_z - math.sin(dr)) / (math.cos(lat) * sz)
        az = math.degrees(math.acos(max(-1.0, min(1.0, ca))))
        az = (az + 180.0) % 360.0 if ha > 0 else (540.0 - az) % 360.0
    elev = 90.0 - zen
    elev_app = elev + _refraction(elev)
    return SunPosition(90.0 - elev_app, az, elev_app, decl, eot)


def solar_noon(day: date, site: Site = Site()) -> datetime:
    """Local clock time of solar noon."""
    noon = datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc)
    jc = (_julian_day(noon) - 2451545.0) / 36525.0
    _, eot = _sun_terms(jc)
    minutes = 720 - 4 * site.longitude - eot + site.tz_hours * 60
    return datetime(day.year, day.month, day.day) + timedelta(minutes=minutes)


def sunrise_sunset(day: date, site: Site = Site()) -> tuple[datetime, datetime]:
    """Local clock times of sunrise and sunset (apparent elevation -0.833 deg)."""
    noon = solar_noon(day, site)
    jc = (_julian_day(noon.replace(tzinfo=timezone(timedelta(hours=site.tz_hours)))
                      .astimezone(timezone.utc)) - 2451545.0) / 36525.0
    decl, _ = _sun_terms(jc)
    lat, dr = math.radians(site.latitude), math.radians(decl)
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(lat) * math.cos(dr))
              - math.tan(lat) * math.tan(dr))
    ha = math.degrees(math.acos(max(-1.0, min(1.0, cos_ha))))
    half = timedelta(minutes=4 * ha)
    return noon - half, noon + half

"""Clear-sky irradiance, GHI decomposition and plane-of-array transposition.

* Clear sky: Ineichen & Perez (2002) with Linke turbidity TL (default 4.5,
  typical of humid tropical air) and Kasten & Young (1989) air mass.
* Decomposition of measured/cloudy GHI into DNI + DHI: Erbs et al. (1982).
* Transposition to a tilted plane: Hay & Davies (1980) sky diffuse model
  plus isotropic ground reflection with albedo rho.

All angles in degrees, irradiance in W/m^2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SOLAR_CONSTANT = 1361.0


@dataclass(frozen=True)
class Irradiance:
    ghi: float
    dni: float
    dhi: float


@dataclass(frozen=True)
class POA:
    total: float
    beam: float
    sky_diffuse: float
    ground: float
    aoi: float  # angle of incidence [deg]


def extraterrestrial(day_of_year: int) -> float:
    """Extraterrestrial normal irradiance (Spencer 1971)."""
    b = 2 * math.pi * (day_of_year - 1) / 365.0
    return SOLAR_CONSTANT * (1.00011 + 0.034221 * math.cos(b) + 0.00128 * math.sin(b)
                             + 0.000719 * math.cos(2 * b) + 0.000077 * math.sin(2 * b))


def airmass(zenith: float) -> float:
    """Relative air mass, Kasten & Young (1989); inf when the sun is down."""
    if zenith >= 90.0:
        return math.inf
    return 1.0 / (math.cos(math.radians(zenith)) + 0.50572 * (96.07995 - zenith) ** -1.6364)


def clearsky_ineichen(zenith: float, day_of_year: int, linke: float = 4.5,
                      altitude_m: float = 0.0) -> Irradiance:
    if zenith >= 90.0:
        return Irradiance(0.0, 0.0, 0.0)
    am = airmass(zenith)
    am_abs = am * math.exp(-altitude_m / 8434.5)  # pressure-corrected air mass
    e0 = extraterrestrial(day_of_year)
    cos_z = math.cos(math.radians(zenith))
    fh1 = math.exp(-altitude_m / 8000.0)
    fh2 = math.exp(-altitude_m / 1250.0)
    cg1 = 5.09e-5 * altitude_m + 0.868
    cg2 = 3.92e-5 * altitude_m + 0.0387
    ghi = cg1 * e0 * cos_z * math.exp(-cg2 * am_abs * (fh1 + fh2 * (linke - 1)))
    ghi = max(ghi, 0.0)
    b = 0.664 + 0.163 / fh1
    dni = b * e0 * math.exp(-0.09 * am_abs * (linke - 1))
    # DNI cannot exceed what GHI implies
    dni_cap = (ghi * (1 - (0.1 - 0.2 * math.exp(-linke)) / (0.1 + 0.882 / fh1))) / cos_z
    dni = max(min(dni, dni_cap), 0.0)
    dhi = max(ghi - dni * cos_z, 0.0)
    return Irradiance(ghi, dni, dhi)


def erbs(ghi: float, zenith: float, day_of_year: int) -> Irradiance:
    """Split GHI into DNI and DHI from the clearness index kt (Erbs 1982)."""
    if zenith >= 87.0 or ghi <= 0.0:
        return Irradiance(max(ghi, 0.0), 0.0, max(ghi, 0.0))
    cos_z = math.cos(math.radians(zenith))
    kt = min(ghi / (extraterrestrial(day_of_year) * cos_z), 1.0)
    if kt <= 0.22:
        df = 1.0 - 0.09 * kt
    elif kt <= 0.80:
        df = 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4
    else:
        df = 0.165
    dhi = df * ghi
    dni = (ghi - dhi) / cos_z
    return Irradiance(ghi, max(dni, 0.0), dhi)


def angle_of_incidence(tilt: float, surf_az: float, zenith: float, sun_az: float) -> float:
    t, z = math.radians(tilt), math.radians(zenith)
    c = math.cos(z) * math.cos(t) + math.sin(z) * math.sin(t) * math.cos(math.radians(sun_az - surf_az))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def poa_hay_davies(irr: Irradiance, tilt: float, surf_az: float, zenith: float, sun_az: float,
                   day_of_year: int, albedo: float = 0.2) -> POA:
    """Plane-of-array irradiance on a surface (tilt, azimuth clockwise from N)."""
    aoi = angle_of_incidence(tilt, surf_az, zenith, sun_az)
    cos_aoi = max(math.cos(math.radians(aoi)), 0.0)
    if zenith >= 90.0:
        cos_aoi = 0.0
    beam = irr.dni * cos_aoi
    e0 = extraterrestrial(day_of_year)
    ai = min(irr.dni / e0, 1.0) if e0 > 0 else 0.0  # anisotropy index
    cos_z = max(math.cos(math.radians(zenith)), 0.01745)  # cap at 89 deg as in pvlib
    rb = cos_aoi / cos_z
    t = math.radians(tilt)
    sky = irr.dhi * (ai * rb + (1 - ai) * (1 + math.cos(t)) / 2)
    ground = irr.ghi * albedo * (1 - math.cos(t)) / 2
    sky, ground = max(sky, 0.0), max(ground, 0.0)
    return POA(beam + sky + ground, beam, sky, ground, aoi)

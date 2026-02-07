"""Synthetic satellite orbits for the GNSS twin."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt

import numpy as np

MU_EARTH = 3.986004418e14
OMEGA_EARTH = 7.2921159e-5


@dataclass(frozen=True)
class Satellite:
    """Simple satellite state."""

    prn: int
    position_ecef_m: np.ndarray
    clock_bias_s: float


class SyntheticOrbitModel:
    """Generate a deterministic, circular constellation."""

    def __init__(
        self,
        num_sats: int = 8,
        radius_m: float = 26_560_000.0,
        inclination_deg: float = 55.0,
        clock_bias_s: float = 1e-6,
    ) -> None:
        self.num_sats = num_sats
        self.radius_m = radius_m
        self.inclination_rad = np.deg2rad(inclination_deg)
        self.clock_bias_s = clock_bias_s
        self._raan = np.linspace(0.0, 2.0 * np.pi, num_sats, endpoint=False)
        self._phase = np.linspace(0.0, 2.0 * np.pi, num_sats, endpoint=False)

    def get_state(self, time_s: float) -> list[Satellite]:
        """Return satellite states at the requested epoch."""

        mean_motion = sqrt(MU_EARTH / self.radius_m**3)
        earth_rot = OMEGA_EARTH * time_s
        sats: list[Satellite] = []
        for prn, raan, phase in zip(range(1, self.num_sats + 1), self._raan, self._phase):
            theta = mean_motion * time_s + phase
            x_orb = self.radius_m * cos(theta)
            y_orb = self.radius_m * sin(theta)
            z_orb = 0.0
            x_inc = x_orb
            y_inc = y_orb * cos(self.inclination_rad) - z_orb * sin(self.inclination_rad)
            z_inc = y_orb * sin(self.inclination_rad) + z_orb * cos(self.inclination_rad)
            x_ecef = x_inc * cos(raan + earth_rot) - y_inc * sin(raan + earth_rot)
            y_ecef = x_inc * sin(raan + earth_rot) + y_inc * cos(raan + earth_rot)
            position = np.array([x_ecef, y_ecef, z_inc], dtype=float)
            sats.append(Satellite(prn=prn, position_ecef_m=position, clock_bias_s=self.clock_bias_s))
        return sats

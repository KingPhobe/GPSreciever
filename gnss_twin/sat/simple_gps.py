"""Simplified GPS-like constellation model."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from gnss_twin.models import Constellation, SvState

MU_EARTH = 3.986004418e14
OMEGA_EARTH = 7.2921159e-5


@dataclass(frozen=True)
class SimpleGpsConfig:
    """Configuration for the simplified GPS constellation."""

    num_sats: int = 24
    num_planes: int = 6
    radius_m: float = 26_560_000.0
    inclination_deg: float = 55.0
    seed: int | None = 0
    clock_bias_sigma_s: float = 50e-9
    clock_drift_sigma_sps: float = 1e-10
    enable_clock: bool = True


def _rot_z(angle_rad: float) -> np.ndarray:
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _rot_x(angle_rad: float) -> np.ndarray:
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ],
        dtype=float,
    )


class SimpleGpsConstellation(Constellation):
    """Deterministic GPS-like constellation with circular orbits."""

    def __init__(self, config: SimpleGpsConfig | None = None) -> None:
        self.config = config or SimpleGpsConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._num_sats = self.config.num_sats
        self._num_planes = max(1, min(self.config.num_planes, self._num_sats))
        self._radius_m = self.config.radius_m
        self._inclination_rad = np.deg2rad(self.config.inclination_deg)
        self._mean_motion = float(np.sqrt(MU_EARTH / self._radius_m**3))

        self._plane_raan = np.linspace(0.0, 2.0 * np.pi, self._num_planes, endpoint=False)
        self._plane_offsets = self._rng.uniform(0.0, 2.0 * np.pi, size=self._num_planes)
        self._plane_index = np.array([i % self._num_planes for i in range(self._num_sats)], dtype=int)
        sats_per_plane = ceil(self._num_sats / self._num_planes)
        self._mean_anom = np.array(
            [
                (2.0 * np.pi * (i // self._num_planes) / sats_per_plane)
                + self._plane_offsets[i % self._num_planes]
                for i in range(self._num_sats)
            ],
            dtype=float,
        )

        if self.config.enable_clock:
            self._clk_bias = self._rng.normal(0.0, self.config.clock_bias_sigma_s, size=self._num_sats)
            self._clk_drift = self._rng.normal(
                0.0, self.config.clock_drift_sigma_sps, size=self._num_sats
            )
        else:
            self._clk_bias = np.zeros(self._num_sats)
            self._clk_drift = np.zeros(self._num_sats)

        self._inclination_matrix = _rot_x(self._inclination_rad)

    def get_sv_states(self, t: float) -> list[SvState]:
        """Return satellite states at the requested epoch."""

        earth_rot = OMEGA_EARTH * t
        rot_earth = _rot_z(earth_rot)
        omega = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)
        sv_states: list[SvState] = []

        for idx in range(self._num_sats):
            raan = self._plane_raan[self._plane_index[idx]]
            theta = self._mean_motion * t + self._mean_anom[idx]

            r_orb = np.array(
                [
                    self._radius_m * np.cos(theta),
                    self._radius_m * np.sin(theta),
                    0.0,
                ],
                dtype=float,
            )
            v_orb = np.array(
                [
                    -self._radius_m * self._mean_motion * np.sin(theta),
                    self._radius_m * self._mean_motion * np.cos(theta),
                    0.0,
                ],
                dtype=float,
            )

            rot_plane = _rot_z(raan) @ self._inclination_matrix
            r_eci = rot_plane @ r_orb
            v_eci = rot_plane @ v_orb

            r_ecef = rot_earth @ r_eci
            v_ecef = rot_earth @ v_eci - np.cross(omega, r_ecef)

            clk_bias = float(self._clk_bias[idx] + self._clk_drift[idx] * t)
            clk_drift = float(self._clk_drift[idx])
            sv_states.append(
                SvState(
                    sv_id=f"G{idx + 1:02d}",
                    t=t,
                    pos_ecef_m=r_ecef,
                    vel_ecef_mps=v_ecef,
                    clk_bias_s=clk_bias,
                    clk_drift_sps=clk_drift,
                )
            )
        return sv_states

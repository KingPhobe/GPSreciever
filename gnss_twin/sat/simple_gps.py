"""Simplified GPS constellation model with circular orbits."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt

import numpy as np

from gnss_twin.models import Constellation, SvState

MU_EARTH = 3.986004418e14
OMEGA_EARTH = 7.2921159e-5


@dataclass(frozen=True)
class SimpleGpsConfig:
    """Configuration for the simplified GPS constellation."""

    num_planes: int = 6
    sats_per_plane: int = 4
    radius_m: float = 26_560_000.0
    inclination_deg: float = 55.0
    seed: int = 0
    raan_jitter_deg: float = 2.0
    mean_anomaly_jitter_deg: float = 5.0
    clock_bias_s: float = 0.0
    clock_amplitude_s: float = 5e-7
    clock_rate_rad_s: float = 2.0 * np.pi / (12.0 * 3600.0)


class SimpleGpsConstellation(Constellation):
    """Generate deterministic GPS-like satellite states."""

    def __init__(self, config: SimpleGpsConfig | None = None) -> None:
        self.config = config or SimpleGpsConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._init_constellation()

    def _init_constellation(self) -> None:
        num_planes = self.config.num_planes
        sats_per_plane = self.config.sats_per_plane
        base_raan = np.linspace(0.0, 2.0 * np.pi, num_planes, endpoint=False)
        raan_jitter = np.deg2rad(self.config.raan_jitter_deg)
        self._raans = base_raan + self._rng.uniform(-raan_jitter, raan_jitter, size=num_planes)

        base_anomaly = np.linspace(0.0, 2.0 * np.pi, sats_per_plane, endpoint=False)
        anomaly_jitter = np.deg2rad(self.config.mean_anomaly_jitter_deg)
        mean_anomalies = []
        clock_phases = []
        for _ in range(num_planes):
            offsets = self._rng.uniform(-anomaly_jitter, anomaly_jitter, size=sats_per_plane)
            mean_anomalies.extend((base_anomaly + offsets).tolist())
            clock_phases.extend(self._rng.uniform(0.0, 2.0 * np.pi, size=sats_per_plane))
        self._mean_anomalies = np.array(mean_anomalies, dtype=float)
        self._clock_phases = np.array(clock_phases, dtype=float)

    def get_sv_states(self, t: float) -> list[SvState]:
        """Return satellite states for the requested epoch time."""

        num_sats = self.config.num_planes * self.config.sats_per_plane
        mean_motion = sqrt(MU_EARTH / self.config.radius_m**3)
        inclination_rad = np.deg2rad(self.config.inclination_deg)
        earth_rot = OMEGA_EARTH * t
        cos_e = cos(earth_rot)
        sin_e = sin(earth_rot)
        omega_vec = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)

        sv_states: list[SvState] = []
        for idx in range(num_sats):
            plane_idx = idx // self.config.sats_per_plane
            raan = self._raans[plane_idx]
            mean_anomaly = mean_motion * t + self._mean_anomalies[idx]

            x_orb = self.config.radius_m * cos(mean_anomaly)
            y_orb = self.config.radius_m * sin(mean_anomaly)
            vx_orb = -self.config.radius_m * mean_motion * sin(mean_anomaly)
            vy_orb = self.config.radius_m * mean_motion * cos(mean_anomaly)

            x_inc = x_orb
            y_inc = y_orb * cos(inclination_rad)
            z_inc = y_orb * sin(inclination_rad)
            vx_inc = vx_orb
            vy_inc = vy_orb * cos(inclination_rad)
            vz_inc = vy_orb * sin(inclination_rad)

            cos_r = cos(raan)
            sin_r = sin(raan)
            x_eci = x_inc * cos_r - y_inc * sin_r
            y_eci = x_inc * sin_r + y_inc * cos_r
            z_eci = z_inc
            vx_eci = vx_inc * cos_r - vy_inc * sin_r
            vy_eci = vx_inc * sin_r + vy_inc * cos_r
            vz_eci = vz_inc

            r_eci = np.array([x_eci, y_eci, z_eci], dtype=float)
            v_eci = np.array([vx_eci, vy_eci, vz_eci], dtype=float)
            v_eci_corr = v_eci - np.cross(omega_vec, r_eci)

            x_ecef = x_eci * cos_e - y_eci * sin_e
            y_ecef = x_eci * sin_e + y_eci * cos_e
            vx_ecef = v_eci_corr[0] * cos_e - v_eci_corr[1] * sin_e
            vy_ecef = v_eci_corr[0] * sin_e + v_eci_corr[1] * cos_e

            position = np.array([x_ecef, y_ecef, z_eci], dtype=float)
            velocity = np.array([vx_ecef, vy_ecef, v_eci_corr[2]], dtype=float)

            clock_phase = self._clock_phases[idx]
            clock_bias = self.config.clock_bias_s + self.config.clock_amplitude_s * sin(
                self.config.clock_rate_rad_s * t + clock_phase
            )
            clock_drift = (
                self.config.clock_amplitude_s
                * self.config.clock_rate_rad_s
                * cos(self.config.clock_rate_rad_s * t + clock_phase)
            )
            sv_states.append(
                SvState(
                    sv_id=f"G{idx + 1:02d}",
                    t=t,
                    pos_ecef_m=position,
                    vel_ecef_mps=velocity,
                    clk_bias_s=clock_bias,
                    clk_drift_sps=clock_drift,
                )
            )
        return sv_states

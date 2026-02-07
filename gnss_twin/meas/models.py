"""Measurement and propagation models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_twin.sat.orbit import Satellite

LIGHT_SPEED = 299_792_458.0


@dataclass(frozen=True)
class Measurement:
    """Synthetic pseudorange measurement."""

    prn: int
    pseudorange_m: float
    truth_range_m: float
    iono_delay_m: float
    tropo_delay_m: float
    noise_m: float
    elevation_rad: float
    sat_clock_bias_s: float
    sat_position_ecef_m: np.ndarray


@dataclass(frozen=True)
class NoiseModel:
    """Simple Gaussian noise model."""

    sigma_m: float = 1.0

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(0.0, self.sigma_m))


def _compute_elevation(receiver_ecef_m: np.ndarray, sat_ecef_m: np.ndarray) -> float:
    los = sat_ecef_m - receiver_ecef_m
    los_unit = los / np.linalg.norm(los)
    up_unit = receiver_ecef_m / np.linalg.norm(receiver_ecef_m)
    return float(np.arcsin(np.clip(np.dot(los_unit, up_unit), -1.0, 1.0)))


def iono_delay_m(elevation_rad: float, zenith_delay_m: float = 5.0) -> float:
    return float(zenith_delay_m / max(np.sin(elevation_rad), 0.2))


def tropo_delay_m(elevation_rad: float, zenith_delay_m: float = 2.3) -> float:
    return float(zenith_delay_m / max(np.sin(elevation_rad), 0.2))


def compute_measurements(
    receiver_ecef_m: np.ndarray,
    receiver_clock_bias_s: float,
    satellites: list[Satellite],
    noise_model: NoiseModel,
    rng: np.random.Generator,
) -> list[Measurement]:
    """Compute pseudorange measurements for the given satellites."""

    measurements: list[Measurement] = []
    for sat in satellites:
        sat_pos = sat.position_ecef_m
        sat_clock = sat.clock_bias_s
        truth_range = float(np.linalg.norm(sat_pos - receiver_ecef_m))
        elevation = _compute_elevation(receiver_ecef_m, sat_pos)
        iono = iono_delay_m(elevation)
        tropo = tropo_delay_m(elevation)
        noise = noise_model.sample(rng)
        pseudorange = truth_range + iono + tropo + noise + LIGHT_SPEED * (receiver_clock_bias_s - sat_clock)
        measurements.append(
            Measurement(
                prn=sat.prn,
                pseudorange_m=pseudorange,
                truth_range_m=truth_range,
                iono_delay_m=iono,
                tropo_delay_m=tropo,
                noise_m=noise,
                elevation_rad=elevation,
                sat_clock_bias_s=sat_clock,
                sat_position_ecef_m=sat_pos,
            )
        )
    return measurements

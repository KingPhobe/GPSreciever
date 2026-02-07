"""Pseudorange measurement models and synthetic measurement source."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_twin.models import Constellation, GnssMeasurement, MeasurementSource, ReceiverTruth
from gnss_twin.utils.angles import elev_az_from_rx_sv

LIGHT_SPEED_MPS = 299_792_458.0


def geometric_range_m(receiver_ecef_m: np.ndarray, sv_ecef_m: np.ndarray) -> float:
    """Compute geometric range between receiver and satellite."""

    return float(np.linalg.norm(sv_ecef_m - receiver_ecef_m))


def pseudorange_m(
    receiver_ecef_m: np.ndarray,
    receiver_clock_bias_s: float,
    sv_ecef_m: np.ndarray,
    sv_clock_bias_s: float,
) -> float:
    """Compute pseudorange including receiver and satellite clock biases."""

    range_m = geometric_range_m(receiver_ecef_m, sv_ecef_m)
    clock_term_m = LIGHT_SPEED_MPS * (receiver_clock_bias_s - sv_clock_bias_s)
    return range_m + clock_term_m


@dataclass
class SyntheticMeasurementSource(MeasurementSource):
    """Generate truth-only pseudorange measurements from a constellation."""

    constellation: Constellation
    receiver_truth: ReceiverTruth
    elevation_mask_deg: float = 10.0
    cn0_dbhz: float = 45.0

    def get_measurements(self, t: float) -> list[GnssMeasurement]:
        sv_states = self.constellation.get_sv_states(t)
        measurements: list[GnssMeasurement] = []
        for state in sv_states:
            elev_deg, az_deg = elev_az_from_rx_sv(self.receiver_truth.pos_ecef_m, state.pos_ecef_m)
            if elev_deg < self.elevation_mask_deg:
                continue
            pr_m = pseudorange_m(
                self.receiver_truth.pos_ecef_m,
                self.receiver_truth.clk_bias_s,
                state.pos_ecef_m,
                state.clk_bias_s,
            )
            measurements.append(
                GnssMeasurement(
                    sv_id=state.sv_id,
                    t=t,
                    pr_m=pr_m,
                    prr_mps=None,
                    cn0_dbhz=self.cn0_dbhz,
                    elev_deg=elev_deg,
                    az_deg=az_deg,
                    flags={"truth": True},
                )
            )
        return measurements

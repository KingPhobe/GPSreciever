"""Holdover decision logic from PPS error and ground PPS age."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HoldoverConfig:
    """Thresholds for holdover health checks."""

    max_abs_pps_err_s: float = 50e-6
    max_time_since_ground_pps_s: float = 2.0


class HoldoverMonitor:
    """Track latest ground PPS timing and evaluate holdover state."""

    def __init__(self, config: HoldoverConfig | None = None) -> None:
        self._config = config or HoldoverConfig()
        self.last_ground_pps_t_true_s: float | None = None

    def update(
        self,
        t_true_s: float,
        pps_err_s: float | None,
        saw_ground_pps: bool,
    ) -> dict[str, float | bool | None]:
        """Update monitor state and return holdover health telemetry."""

        if saw_ground_pps:
            self.last_ground_pps_t_true_s = float(t_true_s)

        if self.last_ground_pps_t_true_s is None:
            time_since_ground = float("inf")
        else:
            time_since_ground = float(t_true_s) - self.last_ground_pps_t_true_s

        ok_err = (pps_err_s is not None) and (
            abs(float(pps_err_s)) <= self._config.max_abs_pps_err_s
        )
        ok_time = time_since_ground <= self._config.max_time_since_ground_pps_s
        holdover_ok = ok_time and ok_err

        return {
            "holdover_ok": holdover_ok,
            "time_since_ground_pps_s": time_since_ground,
            "abs_pps_err_s": None if pps_err_s is None else abs(float(pps_err_s)),
        }

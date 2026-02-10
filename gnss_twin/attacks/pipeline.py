"""Attack pipeline utilities for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass, field

from gnss_twin.attacks.base import AttackModel, AttackReport
from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState


@dataclass
class AttackPipeline:
    """Apply a sequence of attacks to a measurement batch."""

    attacks: list[AttackModel] = field(default_factory=list)

    def reset(self, seed: int | None = None) -> None:
        for attack in self.attacks:
            attack.reset(seed)

    def apply(
        self,
        measurements: list[GnssMeasurement],
        sv_states: list[SvState],
        *,
        rx_truth: ReceiverTruth,
    ) -> tuple[list[GnssMeasurement], AttackReport]:
        report = AttackReport()
        if not self.attacks:
            return list(measurements), report
        sv_by_id = {state.sv_id: state for state in sv_states}
        attacked: list[GnssMeasurement] = []
        for meas in measurements:
            updated = meas
            sv_state = sv_by_id.get(meas.sv_id)
            if sv_state is None:
                attacked.append(updated)
                continue
            for attack in self.attacks:
                updated, delta = attack.apply(updated, sv_state, rx_truth=rx_truth)
                if delta.applied:
                    report.applied_count += 1
                    report.pr_bias_sum_m += delta.pr_bias_m
                    report.prr_bias_sum_mps += delta.prr_bias_mps
            attacked.append(updated)
        return attacked, report

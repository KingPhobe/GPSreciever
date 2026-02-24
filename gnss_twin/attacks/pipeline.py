"""Attack pipeline utilities for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass, field

from gnss_twin.attacks.base import AttackModel, AttackReport
from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState


@dataclass
class AttackPipeline:
    """Apply a sequence of attacks to a measurement batch."""

    attacks: list[AttackModel] = field(default_factory=list)
    _visible_sv_setters: list = field(default_factory=list, init=False, repr=False)
    _visible_sv_setters_cache_key: tuple[int, ...] | None = field(default=None, init=False, repr=False)

    def reset(self, seed: int | None = None) -> None:
        for attack in self.attacks:
            attack.reset(seed)

    def _get_visible_sv_setters(self) -> list:
        cache_key = tuple(id(attack) for attack in self.attacks)
        if self._visible_sv_setters_cache_key != cache_key:
            setters = []
            for attack in self.attacks:
                set_visible_sv_ids = getattr(attack, "set_visible_sv_ids", None)
                if callable(set_visible_sv_ids):
                    setters.append(set_visible_sv_ids)
            self._visible_sv_setters = setters
            self._visible_sv_setters_cache_key = cache_key
        return self._visible_sv_setters

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
        visible_sv_ids = [meas.sv_id for meas in measurements]
        for set_visible_sv_ids in self._get_visible_sv_setters():
            set_visible_sv_ids(visible_sv_ids)
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

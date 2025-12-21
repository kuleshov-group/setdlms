import logging
from typing import Any, List, Literal, Optional, Union

from composer import Event, Logger, State, Time, TimeUnit
from composer.algorithms.ema import EMA
from composer.core.algorithm import Algorithm
from composer.utils import reproducibility

log = logging.getLogger(__name__)


def _is_eval_event(event: Event, state: State) -> bool:
    evaluators_executing = []
    for evaluator in state.evaluators:
        evaluators_executing.append(evaluator.eval_interval(state, event))
    return any(evaluators_executing)

class NoiseLevelAnnealing(Algorithm):
    def __init__(self, anneal_duration: str = "0ba", final_scale: float = 1.0):
        super().__init__()
        self.anneal_duration = Time.from_timestring(anneal_duration)
        assert (
            self.anneal_duration.unit == TimeUnit.BATCH
            or self.anneal_duration.unit == TimeUnit.EPOCH
        ), "Only batch or epoch durations are supported."
        self.annealing_progress = 0.0
        self.final_scale = final_scale
        self._increase_deferred_until_eval_end = False

    def match(self, event: Event, state: State) -> bool:
        if self.anneal_duration.value == 0:
            return False

        if event == Event.AFTER_LOAD:
            return True

        # Execute the "deferred" `apply`
        if event == Event.EVAL_AFTER_ALL and self._increase_deferred_until_eval_end:
            self._increase_deferred_until_eval_end = False  # reset
            log.info("Executing deferred noise level annealing.")
            return True

        if _is_eval_event(event, state):
            log.info("Deferring noise level annealing until end of eval.")
            self._increase_deferred_until_eval_end = True
            return False

        # Currently, only batch or epoch are supported for scheduling
        if event not in [Event.BATCH_END, Event.EPOCH_END]:
            return False

        current_time = state.timestamp.get(self.anneal_duration.unit).value
        if current_time <= self.anneal_duration.value:
            return True
        return False

    @staticmethod
    def _select_model_from_state(state: State):
        if hasattr(state.model, "module"):
            return state.model.module.model
        else:
            return state.model.model

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        current_time = state.timestamp.get(self.anneal_duration.unit).value
        annealing_progress = min(1.0, current_time / self.anneal_duration.value)
        model = self._select_model_from_state(state)
        new_scale = model.noise_schedule.block_size + (annealing_progress * (self.final_scale - model.noise_schedule.block_size))
        model.noise_schedule.init_schedule(scale=new_scale)

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["annealing_progress"] = self.annealing_progress
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.annealing_progress = state_dict["annealing_progress"]
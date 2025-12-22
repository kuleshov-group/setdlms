import logging
from typing import Any, List, Literal, Optional, Union
import math
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
        block_size = model.noise_schedule.block_size

        def scale_fx(x):
            try:
                result = (2 - x - block_size) / (1 - x) - 1
            except ZeroDivisionError:
                result = block_size
            return min(result, block_size)
        
        new_scale = scale_fx(annealing_progress * (block_size - 1) + 1)
        # new_scale = block_size + (annealing_progress * (self.final_scale - block_size))
        model.noise_schedule.init_schedule(scale=new_scale)

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["annealing_progress"] = self.annealing_progress
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.annealing_progress = state_dict["annealing_progress"]


class BlockSizeAnnealing(Algorithm):
    def __init__(
        self,
        max_block_size: int,
        schedule: Union[List[str], str],
        factor: int = 2,
        increase_via_add_or_multiply: Literal["add", "multiply"] = "multiply",
    ):
        super().__init__()
        self.max_block_size = max_block_size
        self.schedule = Time.from_timestring(schedule)
        assert (
            self.schedule.unit == TimeUnit.BATCH
            or self.schedule.unit == TimeUnit.EPOCH
        ), "Only batch or epoch intervals are supported."
        self.factor = factor
        self.increase_via_add_or_multiply = increase_via_add_or_multiply
        if self.increase_via_add_or_multiply == "add":
            self.num_increases = self.max_block_size // self.factor
        elif self.increase_via_add_or_multiply == "multiply":
            self.num_increases = int(math.log(self.max_block_size, self.factor))
        self.schedule = self.schedule / self.num_increases
        print("Setting schedule to ", self.schedule)
        # -1: sentinel value to indicate schedule has not started
        self.block_size = -1
        self._increase_deferred_until_eval_end = False

    def match(self, event: Event, state: State) -> bool:
        if event == Event.AFTER_LOAD:
            return True

        # Execute the "deferred" `apply`
        if event == Event.EVAL_AFTER_ALL and self._increase_deferred_until_eval_end:
            self._increase_deferred_until_eval_end = False  # reset
            log.info("Executing deferred block size increase.")
            return True

        # Currently, only batch or epoch are supported for scheduling increase
        if event not in [Event.BATCH_END, Event.EPOCH_END]:
            return False

        if isinstance(self.schedule, list):
            for s in self.schedule:
                current_time = state.timestamp.get(s.unit).value
                if current_time == s.value:
                    # If trainer will execute eval, wait to apply block increase until
                    # after eval loop end
                    if _is_eval_event(event, state):
                        log.info("Deferring block size increase until end of eval.")
                        self._increase_deferred_until_eval_end = True
                        return False
                    return True
            return False
        current_time = state.timestamp.get(self.schedule.unit).value
        if current_time > 0 and current_time % self.schedule.value == 0:
            # If trainer will execute eval, wait to apply block increase until after
            # eval loop end
            if _is_eval_event(event, state):
                log.info("Deferring block size increase until end of eval.")
                self._increase_deferred_until_eval_end = True
                return False
            return True

        return False

    @staticmethod
    def _select_model_from_state(state: State):
        if hasattr(state.model, "module"):
            return state.model.module.model
        else:
            return state.model.model

    def _maybe_increase_block_size(self, current_block_size):
        if current_block_size >= self.max_block_size:
            return current_block_size
        if current_block_size == 1:
            return 2
        if self.increase_via_add_or_multiply == "add":
            return current_block_size + self.factor
        return current_block_size * self.factor

    def _maybe_update_config_model_collators(
        self,
        state: State,
        new_block_size: Optional[int] = None,
    ) -> int:
        model = self._select_model_from_state(state)
        if new_block_size is None:
            new_block_size = self._maybe_increase_block_size(model.config.block_size)
        if model.config.block_size >= new_block_size:
            return model.config.block_size
        # Update model config
        model.config.block_size = new_block_size
        # Update model
        model.update_static_mask(model.generate_static_mask())
        # Update EMA model (if applicable)
        for alg in state.algorithms:
            if isinstance(alg, EMA):
                if getattr(alg, "ema_model", None) is not None:
                    alg.ema_model.swap_params(state.model)
                    model.update_static_mask(model.generate_static_mask())
                    alg.ema_model.swap_params(state.model)
        # Update collators' block size
        if new_block_size != state.train_dataloader.collate_fn.block_size:
            state.train_dataloader.collate_fn.update_block_size(new_block_size)
        for e in state.evaluators:
            if new_block_size != e.dataloader.dataloader.collate_fn.block_size:
                e.dataloader.dataloader.collate_fn.update_block_size(new_block_size)
        return new_block_size

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.AFTER_LOAD:
            if self.block_size > 0:
                restored_block_size = self._maybe_update_config_model_collators(
                    state,
                    self.block_size,
                )
                self.block_size = restored_block_size
                log.info(f"Restored block size value to {restored_block_size}.")
            else:
                model = self._select_model_from_state(state)
                self.block_size = model.config.block_size
            return
        new_block_size = self._maybe_update_config_model_collators(state)
        if new_block_size != self.block_size:
            log.info(f"Block size updated: {self.block_size} --> {new_block_size}")
            self.block_size = new_block_size

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["block_size"] = self.block_size
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.block_size = state_dict["block_size"]
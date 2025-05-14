from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class SamplerConfig(OrderedDict):
    num_samples: int = 1
    batch_size: int = 1
    max_length: int = 512
    num_steps: int = 1000
    min_t: float = 1e-5
    block_size: int = 512
    top_p: float = 0.9
    pad_context: bool = False
    greedy: bool = False
    use_x0_pred: bool = False
    first_hitting: bool = False
    low_confidence_remasking: bool = False
    disable_cache: bool = False
    kv_caching: bool = False
    shift_logits: bool = False
    repetition_penalty: float = 1.0

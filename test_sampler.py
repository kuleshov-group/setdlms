import os

import hydra
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from composer.models import HuggingFaceModel

from scripts.utils import register_useful_resolvers, load_model_from_ckpt_dir_path

register_useful_resolvers()

### LOAD CHECKPOINT
CKPT_DIR = "/home/ubuntu/runs/dllm-dev/gsm8k-block4-bs96-keep1-causalencfalse-max20000ba-lr1e-5-warmup1000ba-gc1.0-wd1e-5-bd3_qwen600m_v1"

def main():
  base_config = OmegaConf.load(os.path.join(CKPT_DIR, "config.yaml"))
  sampler_overrides = [
      "composer.loggers=null",
      "sampler_config.block_size=4",
      "sampler_config.first_hitting=true",
      "sampler_config.use_x0_pred=true",
      "sampler_config.greedy=true",
      "sampler_config.low_confidence_remasking=true",
      "sampler_config.disable_cache=false",
      "sampler_config.kv_caching=true",
      "sampler_config.min_t=1e-5",
      "sampler_config.shift_logits=true",
      "sampler_config.top_p=0.85",
      "sampler_config.num_steps=1000",
      "sampler_config.pad_context=false",
  ]
  config = OmegaConf.merge(base_config, OmegaConf.from_dotlist(sampler_overrides))

  ckpt_file = (
      f"{CKPT_DIR}/checkpoints/best-rank0.pt"
  )

  tokenizer = AutoTokenizer.from_pretrained(
      config.tokenizer.pretrained_model_name_or_path,
      trust_remote_code=True,
      use_fast=False,
  )

  model = load_model_from_ckpt_dir_path(
      path_to_ckpt_dir=CKPT_DIR,
      load_ema_weights=True,
      ckpt_file="best-rank0.pt",
      verbose=False,
  ).to("cuda")
  model.sampler_config = config.sampler_config

  ### PREPARE SAMPLING
  #PROMPT="<|im_end|>Please reason step by step, and put your final answer within \\boxed{}. Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|im_end|>Answer:"
  PROMPT="<|im_end|>Please reason step by step, and put your final answer within \\boxed{}. Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>Answer: "

  prompt = tokenizer(
      PROMPT,
      return_tensors="pt",
      padding=True,
      truncation=True,
  ).to(model.device)

  """
  Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
  She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
  \\boxed{18}
  """

  ### SAMPLE
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()

  samples, NFEs = model.generate(
      batch_size=1, max_length=512, context=prompt["input_ids"], tokenizer=tokenizer
  )
  end.record()
  torch.cuda.synchronize()
  print(f"Time taken: {start.elapsed_time(end)} ms")
  print(tokenizer.decode(samples[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()

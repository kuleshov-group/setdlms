import datetime
import json
import os

import evaluate
import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM
from transformers.generation import StopStringCriteria

from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
    print_and_save_config,
    register_useful_resolvers,
    set_seed,
)


def gather_results(results, world_size):
    # Each GPU has local 'results' (any pickle-able object)
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, results)

    # gathered_results is now a list of lists (one per rank)
    all_results = []
    for partial in gathered_results:
        all_results.extend(partial)  # type: ignore

    return all_results


def setup_ddp() -> int:
    """Sets up torch.distributed and selects GPU.

    Returns:
        (int) local_rank
    """
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    print_and_save_config(cfg, resolve=True, save_cfg=False)
    set_seed(cfg.seed)
    local_rank = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # Load the dataset
    dataset = hydra.utils.instantiate(
        cfg.task.dataset,
        tokenizer=tokenizer,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
    )
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=True
    )

    # Load model
    try:
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=cfg.pretrained_model_name_or_path,
            load_ema_weights=cfg.load_ema_weights,
            ckpt_file=cfg.ckpt_file,
        )
    except FileNotFoundError:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
    model = model.to(device)
    model.eval()
    gen_kwargs = hydra.utils.instantiate(cfg.gen_kwargs)
    stop_tokens = None
    if "stopping_criteria" in gen_kwargs:
        for sc in gen_kwargs["stopping_criteria"]:
            if isinstance(sc, StopStringCriteria):
                stop_tokens = list(sc.stop_strings)
                break

    # Iterate through the dataset and sample
    generated_samples = []
    for elem_id, elem in tqdm(
        enumerate(dataloader),
        desc="Generating",
        total=len(dataloader),
        disable=(local_rank != 0),
    ):
        input_ids = elem["input_ids"].to(device)  # type: ignore
        input_ids = input_ids[:, 1:]  # remove bos
        prompt_ids = (
            torch.tensor(tokenizer.encode(dataset.target_prompt_text.strip()))
            .to(input_ids.dtype)
            .to(input_ids.device)
            .unsqueeze(0)
        )
        input_ids = torch.cat((input_ids, prompt_ids), dim=-1)
        # Generate samples
        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                disable_pbar=(local_rank != 0),
                tokenizer=tokenizer,
                **gen_kwargs,
            )
        outputs = outputs[:, input_ids.shape[-1] :]
        # Decode the generated samples
        outputs = tokenizer.decode(outputs[0])
        # Post-process:
        outputs = outputs.replace(" .", ".")
        if stop_tokens is not None:
            for st in stop_tokens:
                outputs = outputs.split(st)[0]
        decoded_samples = dataset.target_prompt_text + outputs.strip()
        if local_rank == 0:
            print("Output:", decoded_samples)
        generated_samples.append(decoded_samples)

    # Compute metrics
    references = dataset.target_references
    local_indices = list(sampler)[: len(generated_samples)]
    references = [references[i] for i in local_indices]
    world_size = dist.get_world_size()
    generated_samples = gather_results(generated_samples, world_size)
    references = gather_results(references, world_size)
    if local_rank == 0:
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("sacrebleu")
        meteor = evaluate.load("meteor")
        rouge_scores = rouge.compute(
            predictions=generated_samples, references=references
        )
        bleu_score = bleu.compute(
            predictions=generated_samples, references=[[ref] for ref in references]
        )
        meteor_score = meteor.compute(
            predictions=generated_samples, references=references
        )

        # Display results
        print("\n=== Evaluation Metrics ===")
        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        print(f"BLEU:    {bleu_score['score']:.4f}")
        print(f"METEOR:  {meteor_score['meteor']:.4f}")

        res_for_json = [
            {"ground_truth": references[i], "result": generated_samples[i]}
            for i in range(len(generated_samples))
        ]

        samples_path = f"{cfg.output_path}/seq2seq_eval_{cfg.task}_output"
        if not os.path.exists(samples_path):
            os.makedirs(samples_path)
        with open(f"{samples_path}/all_ranks.json", "w") as f:
            json.dump(
                res_for_json,
                f,  # type: ignore
                indent=2,
            )
        with open(f"{samples_path}/metrics.json", "w") as f:
            json.dump(
                {
                    "ROUGE-1": rouge_scores["rouge1"],
                    "ROUGE-2": rouge_scores["rouge2"],
                    "ROUGE-L": rouge_scores["rougeL"],
                    "BLEU": bleu_score["score"],
                    "METEOR": meteor_score["meteor"],
                },
                f,  # type: ignore
                indent=2,
            )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    register_useful_resolvers()
    main()

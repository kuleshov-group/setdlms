import datetime
import inspect
import json
import os
import random
from argparse import ArgumentParser, ArgumentTypeError

import evaluate
import nltk
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
)

from scripts.utils import (
    EOSStoppingCriteria,
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
)
from src.datasets.tokenize_on_demand import CNNDailyMailDataset, WMTDataset
from src.sampler import SamplerConfig


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def gather_results(results, world_size):
    # Each GPU has local 'results' (any picklable object)
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, results)

    # gathered_results is now a list of lists (one per rank)
    all_results = []
    for partial in gathered_results:
        all_results.extend(partial)

    return all_results


def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    local_rank = setup_ddp()  # sets up torch.distributed and selects GPU
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    tokenizer = maybe_add_missing_special_tokens(tokenizer)

    # Load the dataset
    if args.dataset == "cnndm":
        dataset_cls = CNNDailyMailDataset
    elif args.dataset == "wmt":
        dataset_cls = WMTDataset
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    dataset = dataset_cls(
        tokenizer=tokenizer,
        split="test",
        max_seq_len=args.max_length,
        separate_input_output=True,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
    )
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=True
    )

    # Load model
    hf_model = False
    try:
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=args.model_path,
            load_ema_weights=args.load_ema_weights,
            ckpt_file=args.ckpt_file,
        ).to(device)
    except FileNotFoundError:
        hf_model = True
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        ).to(device)

    model.eval()
    # Load sampler
    sampler_config = SamplerConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        min_t=args.min_t,
        top_p=args.top_p,
        pad_context=args.pad_context,
        greedy=args.greedy,
        use_x0_pred=args.use_x0_pred,
        first_hitting=args.first_hitting,
        low_confidence_remasking=args.low_confidence_remasking,
        disable_cache=args.disable_cache,
        kv_caching=args.kv_caching,
        max_length=args.max_length
        if args.max_length is not None
        else model.config.length,
        block_size=args.block_size
        if args.block_size is not None
        else model.config.block_size,
        shift_logits=args.shift_logits
        if args.shift_logits is not None
        else model.config.shift_logits,
        repetition_penalty=args.repetition_penalty
        if args.repetition_penalty is not None
        else model.config.repetition_penalty,
    )
    model.sampler_config = sampler_config
    stop_token_ids = [
        tokenizer.encode("<|im_end|>")[0],
        tokenizer.encode("<|endoftext|>")[0],
    ]
    if args.dataset == "cnndm":
        cnndm_stop_tokens = [
            "Summary:",
            # "CLICK",
            # "Click ",
            # "READ:",
            # "READ HERE",
            # "NEW:",
            # "Sources:",
            # "Follow ",
            # "Follow: ",
            # "Related:",
            # "Source:",
            # "Author:",
            # "CNN.com",
            # "Read:",
        ]
        for cnndm_stop_token in cnndm_stop_tokens:
            stop_token_ids.append(tokenizer.encode(cnndm_stop_token)[0])
    eos_stopping_criteria = EOSStoppingCriteria(stop_token_ids)

    # Iterate through the dataset and sample
    generated_samples = []
    for elem_id, elem in tqdm(
        enumerate(dataloader),
        desc="Generating",
        total=len(dataloader),
        disable=(local_rank != 0),
    ):
        stopping_criteria = StoppingCriteriaList([eos_stopping_criteria])
        input_ids = elem["input_ids"].to(device)
        input_ids = input_ids[:, 1:]  # remove bos
        # TODO also have option for wmt
        if args.dataset == "cnndm":
            prompt_ids = (
                torch.tensor(tokenizer.encode("Summary:"))
                .to(input_ids.dtype)
                .to(input_ids.device)
                .unsqueeze(0)
            )
        elif args.dataset == "wmt":
            prompt_ids = (
                torch.tensor(tokenizer.encode("Translation:"))
                .to(input_ids.dtype)
                .to(input_ids.device)
                .unsqueeze(0)
            )
        input_ids = torch.cat((input_ids, prompt_ids), dim=-1)
        # Generate samples
        with torch.no_grad():
            if not hf_model:
                outputs, _ = model.generate(
                    max_length=input_ids.shape[-1] + args.max_new_tokens,
                    context=input_ids,
                    device=device,
                    stopping_criteria=stopping_criteria,
                    disable_pbar=(local_rank != 0),
                    repetition_penalty=args.repetition_penalty,
                    len_penalty=args.len_penalty,
                    regulation_start=args.regulation_start,
                    # tokenizer=tokenizer,
                )
            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    top_k=None,
                    repetition_penalty=args.repetition_penalty,
                    exponential_decay_length_penalty=(
                        args.regulation_start,
                        args.len_penalty,
                    )
                    if args.len_penalty != 1
                    else None,
                )
        outputs = outputs[:, input_ids.shape[-1] :]
        # Decode the generated samples
        outputs = tokenizer.decode(outputs[0])
        outputs = outputs.replace(" .", ".")

        for stop_token_id in stop_token_ids:
            stop_token = tokenizer.decode(stop_token_id)
            outputs = outputs.split(stop_token)[0]

        # For WMT, only use the first sentence (test set only contains single sentences)
        if args.dataset == "wmt":
            outputs = outputs.split(". ")[0] + "."

        if local_rank == 0:
            print("Output:", outputs)
        if args.dataset == "cnndm":
            decoded_samples = "Summary:" + outputs
        elif args.dataset == "wmt":
            decoded_samples = "Translation:" + outputs
        generated_samples.append(decoded_samples)

    # Compute metrics
    references = (
        dataset.dataset["highlights"]
        if args.dataset == "cnndm"
        else [d["translation"][dataset.target] for d in dataset.dataset]
    )
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

        print("Settings:", sampler_config)

        res_for_json = [
            {"ground_truth": references[i], "result": generated_samples[i]}
            for i in range(len(generated_samples))
        ]

        samples_path = f"{args.output_path}/seq2seq_eval_{args.dataset}_output"
        if not os.path.exists(samples_path):
            os.makedirs(samples_path)
        with open(f"{samples_path}/all_ranks.json", "w") as f:
            json.dump(
                res_for_json,
                f,  # type: ignore
                indent=2,
            )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # Download NLTK data required for METEOR
    try:
        nltk.download("wordnet")
    except:  # noqa: E722
        pass
    try:
        nltk.download("punkt")
    except:  # noqa: E722
        pass

    parser = ArgumentParser(description="Seq2seq evaluation script")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cnndm", "wmt"],
        help="The dataset to use for evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, help="Max max_new_tokens (i.e., output) length."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to the model checkpoint.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="The path to the output directory.",
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        help="The name of the checkpoint file.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="The name or path of the tokenizer.",
    )
    parser.add_argument(
        "--load_ema_weights",
        action="store_true",
        help="Whether to load EMA weights.",
    )
    sampler_parser = parser.add_argument_group("Sampler arguments")
    sig = inspect.signature(SamplerConfig)
    for k, v in sig.parameters.items():
        if k == "self":
            continue
        if hasattr(v, "default"):
            sampler_parser.add_argument(
                f"--{k}",
                type=type(v.default) if not isinstance(v.default, bool) else str2bool,
                default=v.default,
                help=f"SamplerConfig {k} parameter.",
            )
        else:
            sampler_parser.add_argument(
                f"--{k}",
                type=v.annotation,
                help=f"SamplerConfig {k} parameter.",
            )
    opts = parser.parse_args()
    print("running with arguments:", vars(opts))
    set_seed(1234)
    main(opts)

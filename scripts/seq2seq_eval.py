import inspect
from argparse import ArgumentParser

import evaluate
import nltk
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.utils import (
    load_model_from_ckpt_dir_path,
    maybe_add_missing_special_tokens,
)
from src.datasets.tokenize_on_demand import CNNDailyMailDataset, WMTDataset
from src.sampler import SamplerConfig


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Load model
    try:
        model = load_model_from_ckpt_dir_path(
            path_to_ckpt_dir=args.model_path,
            load_ema_weights=args.load_ema_weights,
        ).to(device)
    except FileNotFoundError:
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
    )
    model.sampler_config = sampler_config

    # Iterate through the dataset and sample
    generated_samples = []
    for elem in tqdm(dataset, desc="Sampling"):
        input_ids = elem["input_ids"].unsqueeze(0).to(device)
        # Generate samples
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
            )
        # Decode the generated samples
        decoded_samples = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        # TODO: Apply post-processing to decoded samples if needed
        # TODO: Remove context from generated samples if needed
        # TODO: Truncate generated samples at eot_token / im_end_token
        generated_samples.append(decoded_samples)

    # Compute metrics
    references = (
        dataset.dataset["highlights"]
        if args.dataset == "cnndm"
        else [d["translation"][dataset.target] for d in dataset.dataset]
    )
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    rouge_scores = rouge.compute(predictions=generated_samples, references=references)
    bleu_score = bleu.compute(
        predictions=generated_samples, references=[[ref] for ref in references]
    )
    meteor_score = meteor.compute(predictions=generated_samples, references=references)

    # Display results
    print("\n=== Evaluation Metrics ===")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"BLEU:    {bleu_score['score']:.4f}")
    print(f"METEOR:  {meteor_score['meteor']:.4f}")


if __name__ == "__main__":
    # Download NLTK data required for METEOR
    nltk.download("wordnet")
    nltk.download("punkt")

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
        "--tokenizer_name_or_path",
        type=str,
        help="The name or path of the tokenizer.",
    )
    parser.add_argument(
        "--load_ema_weights",
        action="store_true",
        default=True,
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
                type=type(v.default),
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
    main(opts)

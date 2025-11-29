"""Run a single inference with the quantized LLaDA model saved in ./quant-llada-8b-aqlm

This script loads the Transformers LLaDA model using the project's loader which knows how to
load AQLM-quantized weights. It then runs one generation for a sample prompt and prints output.

Usage: python run_quant_llada_inference.py

Notes:
- Requires the AQLM repository dependencies (torch, transformers, accelerate, etc.).
- Adjust DEVICE to 'cpu' or 'cuda' depending on availability.
- By default this uses the local quantized folder `./quant-llada-8b-aqlm` in the repo root.
"""

import time

# imported from generate.py in LLaDA repo, with minor modifications

import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
import torch.nn.functional as F

PROMPT = "Translate the following English sentence to French: 'The quick brown fox jumps over the lazy dog.'"
PROMPT = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
PROMPT = "Explain the theory of relativity in simple terms."
# PROMPT = "Explain Quantum Computing in simple terms."

# Set the device to either 'cuda' or 'cpu' depending on availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    torch.manual_seed(0)

    # Load the model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(
        "kuleshov-group/e2d2-gsm8k-finetune-Qwen3-2B",
        trust_remote_code=True,
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(
        "kuleshov-group/e2d2-gsm8k-finetune-Qwen3-2B",
        trust_remote_code=True,
    )

    # Tokenize the prompt
    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

    # Run forward pass (generation)
    print("Running forward/generation ...")
    start = time.time()

    # Generate model output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            max_length=150,  # Set a maximum length for the output
            batch_size=1,
            num_beams=5,  # Using beam search
            no_repeat_ngram_size=2,  # Avoid repeating n-grams
            early_stopping=True,  # Stop when the model generates a valid sentence
            device=DEVICE
        )

    end = time.time()

    print(f"\nGeneration took {end - start:.2f} seconds")

    # Decode and print the result
    print("\n=== Prompt ===")
    print(PROMPT)
    print("\n=== Output ===")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
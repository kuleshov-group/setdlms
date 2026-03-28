import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from gsm8k_diffusion.block_diffusion import BlockDiffusionModel
    from gsm8k_diffusion.set_diffusion import SetDiffusionModel
else:
    BlockDiffusionModel = None
    SetDiffusionModel = None


class DummyTokenizer:
    def __init__(self):
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.mask_token_id = 99
        self.pad_token_id = 0

    def get_added_vocab(self):
        return {
            self.mask_token: self.mask_token_id,
            self.pad_token: self.pad_token_id,
        }


class DummyForwardOutput:
    def __init__(self, logits, denoiser_output=None):
        self.logits = logits
        self.denoiser_output = denoiser_output


class DummyGenerateOutput(dict):
    def __init__(self, sequences, parallelism_factor=0.0, inf_budget=None):
        super().__init__(
            sequences=sequences,
            parallelism_factor=parallelism_factor,
            inf_budget=inf_budget,
        )
        self.sequences = sequences


class DummyDenoiser:
    def __init__(self, length=8):
        self.config = type("Config", (), {"length": length})()
        self.forward_calls = []
        self.generate_calls = []

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        return DummyForwardOutput(
            logits=torch.ones(1, 4, 7),
            denoiser_output=torch.full((1, 4, 7), 2.0),
        )

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return DummyGenerateOutput(
            sequences=torch.tensor([[11, 12, 13, 14]], dtype=torch.long),
            parallelism_factor=2.5,
            inf_budget=1.25,
        )


@unittest.skipIf(torch is None, "torch is not installed in this validation environment")
class WrapperParityTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.backbone = torch.nn.Embedding(128, 8)

    def test_set_forward_prefers_original_denoiser_output(self):
        denoiser = DummyDenoiser()
        model = SetDiffusionModel(
            model=self.backbone,
            tokenizer=self.tokenizer,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            device="cpu",
            denoiser=denoiser,
            source="checkpoint_dir",
        )
        logits = model.forward(
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
        self.assertEqual(tuple(logits.shape), (1, 4, 7))
        self.assertTrue(torch.equal(logits, torch.full((1, 4, 7), 2.0)))
        self.assertEqual(len(denoiser.forward_calls), 1)

    def test_block_forward_matches_mdlm_postprocessing_without_denoiser(self):
        mask_token_id = self.tokenizer.mask_token_id

        class DummyBackbone:
            def __init__(self, mask_token_id):
                self.mask_token_id = mask_token_id

            def __call__(self, **kwargs):
                logits = torch.zeros(1, 2, 128)
                logits[0, 0, 7] = 5.0
                logits[0, 1, self.mask_token_id] = 10.0
                logits[0, 1, 6] = 9.0
                return DummyForwardOutput(logits=logits)

        model = BlockDiffusionModel(
            model=DummyBackbone(mask_token_id),
            tokenizer=self.tokenizer,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            block_size=4,
            align_inputs_to_blocks=True,
            device="cpu",
            denoiser=None,
            source="pretrained",
        )
        log_probs = model.forward(
            input_ids=torch.tensor([[7, self.tokenizer.mask_token_id]], dtype=torch.long),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
        )
        self.assertEqual(int(log_probs[0, 0].argmax().item()), 7)
        self.assertEqual(int(log_probs[0, 1].argmax().item()), 6)
        self.assertLess(log_probs[0, 1, self.tokenizer.mask_token_id].item(), -1e6)

    def test_block_generate_delegates_and_splits_trailing_masks(self):
        denoiser = DummyDenoiser()
        model = BlockDiffusionModel(
            model=self.backbone,
            tokenizer=self.tokenizer,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            block_size=4,
            align_inputs_to_blocks=True,
            device="cpu",
            denoiser=denoiser,
            source="checkpoint_dir",
        )
        result = model.generate(
            input_ids=torch.tensor([[11, 12, 99, 99]], dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            confidence_threshold=0.9,
            use_cache=True,
        )
        self.assertEqual(result.generation_order, [])
        self.assertTrue(result.metadata["delegated_to_original_generate"])
        self.assertAlmostEqual(result.metadata["parallelism_factor"], 2.5)
        call = denoiser.generate_calls[0]
        self.assertTrue(torch.equal(call["inputs"], torch.tensor([[11, 12]])))
        self.assertEqual(call["max_new_tokens"], 2)

    def test_set_generate_delegates_when_window_size_is_explicit(self):
        denoiser = DummyDenoiser(length=16)
        model = SetDiffusionModel(
            model=self.backbone,
            tokenizer=self.tokenizer,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            device="cpu",
            denoiser=denoiser,
            source="checkpoint_dir",
        )
        result = model.generate(
            input_ids=torch.tensor([[21, 22, 99, 99]], dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            confidence_threshold=0.75,
            max_window_size=4,
            use_cache=True,
        )
        self.assertEqual(result.generation_order, [])
        self.assertEqual(result.metadata["generation_order_source"], "unavailable_from_original_generate")
        call = denoiser.generate_calls[0]
        self.assertTrue(torch.equal(call["inputs"], torch.tensor([[21, 22]])))
        self.assertEqual(call["max_new_tokens"], 2)

    def test_set_generate_rejects_no_cache_with_original_denoiser(self):
        denoiser = DummyDenoiser(length=16)
        model = SetDiffusionModel(
            model=self.backbone,
            tokenizer=self.tokenizer,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            device="cpu",
            denoiser=denoiser,
            source="checkpoint_dir",
        )
        with self.assertRaisesRegex(ValueError, "use_cache=True"):
            model.generate(
                input_ids=torch.tensor([[21, 22, 99, 99]], dtype=torch.long),
                attention_mask=torch.ones(1, 4, dtype=torch.long),
                confidence_threshold=0.75,
                max_window_size=4,
                use_cache=False,
            )


if __name__ == "__main__":
    unittest.main()

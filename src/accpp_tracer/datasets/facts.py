"""Factual recall dataset for simple knowledge probing."""

import torch
from transformer_lens import HookedTransformer


class FactsDataset:
    """Simple factual recall dataset.

    Contains prompts testing factual knowledge (capitals, sports, arithmetic)
    with known ground-truth answers.

    Args:
        model: HookedTransformer model instance (used for tokenization).
        prepend_bos: Whether to prepend beginning-of-sequence token.
        device: Torch device.
    """

    def __init__(
        self,
        model: HookedTransformer,
        prepend_bos: bool = False,
        device: str = "cpu",
    ):
        self.sentences = [
            "Fact: the capital of the state containing Dallas is",
            "Q: What sport Michael Jordan played? A: Basketball\nQ: What sport Tom Brady played? A:",
            "Q: What sport Tom Brady played? A:",
            "Fact: the capital of the state containing San Francisco is",
            "Fact: the capital of the state containing Buffalo is",
            "1+1=",
        ]

        self.toks = model.to_tokens(self.sentences, prepend_bos=prepend_bos).to(device)

        self.word_idx: dict[str, torch.Tensor] = {"end": []}

        pad_token_id = model.tokenizer.pad_token_id
        for prompt_id in range(len(self.toks)):
            end_id = torch.where(self.toks[prompt_id] == pad_token_id)[0]
            if len(end_id) == 0:
                self.word_idx["end"].append(self.toks.shape[1] - 1)
            else:
                end_id = end_id[0] - 1
                self.word_idx["end"].append(end_id.item())
        self.word_idx["end"] = torch.tensor(self.word_idx["end"], device=device)

        self.answers = [
            " Austin",
            " Football",
            " Football",
            " Sacramento",
            " Albany",
            "2",
        ]

        self.answers_id = torch.tensor(
            model.tokenizer(self.answers, add_special_tokens=False)["input_ids"]
        ).squeeze().to(device)

    def __len__(self) -> int:
        return len(self.toks)

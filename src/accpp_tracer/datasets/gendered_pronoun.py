"""Gendered Pronoun prediction dataset.

Modified from the original implementation (Automatic_Circuits_Script.ipynb):
https://cmathw.itch.io/identifying-a-preliminary-circuit-for-predicting-gendered-pronouns-in-gpt-2-smal
"""

import torch
from transformer_lens import HookedTransformer


class GenderedPronoun:
    """Gendered pronoun prediction dataset.

    Generates prompts like "So {name} is a really great friend, isn't"
    where the model must predict " he" or " she" based on the name.

    Args:
        model: HookedTransformer model instance.
        model_family: Model family ("gpt2", "pythia", "gemma").
        device: Torch device.
        prepend_bos: Whether to prepend beginning-of-sequence token.
    """

    def __init__(
        self,
        model: HookedTransformer,
        model_family: str,
        device: str = "cpu",
        prepend_bos: bool = False,
    ):
        if model_family == "gemma" and not prepend_bos:
            raise ValueError(
                f"model_family='gemma' requires prepend_bos=True"
            )

        self.templates = [
            "So {name} is a really great friend, isn't",
            "So {name} is such a good cook, isn't",
            "So {name} is a very good athlete, isn't",
            "So {name} is a really nice person, isn't",
            "So {name} is such a funny person, isn't",
        ]

        self.male_names = [
            "John", "David", "Mark", "Paul", "Ryan",
            "Gary", "Jack", "Sean", "Carl", "Joe",
        ]

        self.female_names = [
            "Mary", "Lisa", "Anna", "Sarah", "Amy",
            "Carol", "Karen", "Susan", "Julie", "Judy",
        ]

        self.sentences: list[str] = []
        self.answers: list[str] = []
        self.wrongs: list[str] = []
        self.responses = [" he", " she"]

        for name in self.male_names + self.female_names:
            for template in self.templates:
                self.sentences.append(template.format(name=name))

        batch_size = len(self.sentences)
        count = 0
        for _ in range(batch_size):
            if count < (0.5 * len(self.sentences)):
                self.answers.append(self.responses[0])
                self.wrongs.append(self.responses[1])
                count += 1
            else:
                self.answers.append(self.responses[1])
                self.wrongs.append(self.responses[0])

        self.tokens = model.to_tokens(self.sentences, prepend_bos=prepend_bos).to(
            device
        )
        self.toks = self.tokens  # Alias for compatibility with TracingDataset
        self.answers = torch.tensor(
            model.tokenizer(self.answers, add_special_tokens=False)["input_ids"]
        ).squeeze().to(device)
        self.wrongs = torch.tensor(
            model.tokenizer(self.wrongs, add_special_tokens=False)["input_ids"]
        ).squeeze().to(device)

        if not prepend_bos and model_family in ["gpt2", "pythia"]:
            self.word_idx = {
                "end": torch.full((batch_size,), self.tokens.shape[1] - 1, dtype=int),
                "end-1": torch.full((batch_size,), self.tokens.shape[1] - 2, dtype=int),
                "punct": torch.full((batch_size,), self.tokens.shape[1] - 3, dtype=int),
                "noun": torch.full((batch_size,), self.tokens.shape[1] - 4, dtype=int),
                "adj": torch.full((batch_size,), self.tokens.shape[1] - 5, dtype=int),
                "I2": torch.full((batch_size,), self.tokens.shape[1] - 6, dtype=int),
                "I1": torch.full((batch_size,), self.tokens.shape[1] - 7, dtype=int),
                "is": torch.full((batch_size,), self.tokens.shape[1] - 8, dtype=int),
                "name": torch.full((batch_size,), self.tokens.shape[1] - 9, dtype=int),
                "starts": torch.zeros(batch_size, dtype=int),
            }
        elif prepend_bos and model_family in ["gpt2", "pythia"]:
            self.word_idx = {
                "end": torch.full((batch_size,), self.tokens.shape[1] - 1, dtype=int),
                "end-1": torch.full((batch_size,), self.tokens.shape[1] - 2, dtype=int),
                "punct": torch.full((batch_size,), self.tokens.shape[1] - 3, dtype=int),
                "noun": torch.full((batch_size,), self.tokens.shape[1] - 4, dtype=int),
                "adj": torch.full((batch_size,), self.tokens.shape[1] - 5, dtype=int),
                "I2": torch.full((batch_size,), self.tokens.shape[1] - 6, dtype=int),
                "I1": torch.full((batch_size,), self.tokens.shape[1] - 7, dtype=int),
                "is": torch.full((batch_size,), self.tokens.shape[1] - 8, dtype=int),
                "name": torch.full((batch_size,), self.tokens.shape[1] - 9, dtype=int),
                "So": torch.full((batch_size,), self.tokens.shape[1] - 10, dtype=int),
                "starts": torch.zeros(batch_size, dtype=int),
            }
        elif model_family in ["gemma"]:
            self.word_idx = {
                "end": torch.full((batch_size,), self.tokens.shape[1] - 1, dtype=int),
                "end-1": torch.full((batch_size,), self.tokens.shape[1] - 2, dtype=int),
                "end-2": torch.full((batch_size,), self.tokens.shape[1] - 3, dtype=int),
                "punct": torch.full((batch_size,), self.tokens.shape[1] - 4, dtype=int),
                "noun": torch.full((batch_size,), self.tokens.shape[1] - 5, dtype=int),
                "adj": torch.full((batch_size,), self.tokens.shape[1] - 6, dtype=int),
                "I2": torch.full((batch_size,), self.tokens.shape[1] - 7, dtype=int),
                "I1": torch.full((batch_size,), self.tokens.shape[1] - 8, dtype=int),
                "is": torch.full((batch_size,), self.tokens.shape[1] - 9, dtype=int),
                "name": torch.full((batch_size,), self.tokens.shape[1] - 10, dtype=int),
                "So": torch.full((batch_size,), self.tokens.shape[1] - 11, dtype=int),
                "starts": torch.zeros(batch_size, dtype=int),
            }

        for key in self.word_idx:
            self.word_idx[key] = self.word_idx[key].to(device)

    def __len__(self) -> int:
        return len(self.sentences)

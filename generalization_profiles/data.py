import math
from dataclasses import dataclass
from typing import cast

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


@dataclass
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class TokenDataset:
    def get_samples(self, indices: torch.Tensor) -> TokenBatch:
        raise NotImplementedError

    def get_sample_indices_for_batch(self, batch_index: int) -> torch.Tensor:
        raise NotImplementedError

    def get_num_batches(self) -> int:
        raise NotImplementedError


class MacroBatchedDataset(TokenDataset):
    def __init__(self, original_dataset: TokenDataset, batching_factor: int):
        self.original_dataset = original_dataset
        self.batching_factor = batching_factor

    def get_samples(self, indices: torch.Tensor) -> TokenBatch:
        return self.original_dataset.get_samples(indices)

    def get_sample_indices_for_batch(self, batch_index: int) -> torch.Tensor:
        original_num_batches = self.original_dataset.get_num_batches()
        first_og_batch_index = self.batching_factor * batch_index
        indices = []
        for b in torch.arange(self.batching_factor) + first_og_batch_index:
            b = int(b)
            if b >= original_num_batches:
                break
            indices.append(
                self.original_dataset.get_sample_indices_for_batch(b)
            )
        return torch.cat(indices)

    def get_num_batches(self) -> int:
        return math.ceil(
            self.original_dataset.get_num_batches() / self.batching_factor
        )


class DummyIMDBTokenDataset(TokenDataset):
    def __init__(self, tokenizer: AutoTokenizer, batch_size: int = 128):
        self.tokenizer = tokenizer
        self.dataset = cast(
            Dataset, load_dataset('stanfordnlp/imdb')['unsupervised']
        )
        self.batch_size = batch_size

    def get_num_batches(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def get_sample_indices_for_batch(self, batch_index: int) -> torch.Tensor:
        if not 0 <= batch_index < self.get_num_batches():
            raise ValueError(f'Batch index out of bounds: {batch_index}')
        return (
            torch.arange(self.batch_size, dtype=torch.long)
            + self.batch_size * batch_index
        )

    def get_samples(self, indices: torch.Tensor) -> TokenBatch:
        texts = self.dataset[indices]['text']
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True)  # type: ignore
        return TokenBatch(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
        )

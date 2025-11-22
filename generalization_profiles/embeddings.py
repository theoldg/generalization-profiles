import shelve
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from fire import Fire
import datasets
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from generalization_profiles.pythia_model import PythiaModel


CACHE_LOCATION = Path("cache/embeddings")
ALIBABA_MODEL = "Alibaba-NLP/gte-multilingual-base"


@dataclass
class TextSample:
    value: str
    seq_idx: int


def load_and_detokenize_pile_subset() -> list[TextSample]:
    data = datasets.load_dataset("pietrolesci/pile-deduped-subset")
    all_data = datasets.concatenate_datasets(
        [data["train"], data["validation"]]  # type: ignore
    )

    # We'll use seq_idx as unique IDs.
    assert len(set(all_data["seq_idx"])) == len(all_data)

    # Doesn't matter which one, we just want the tokenizer.
    model = PythiaModel.from_variant_and_revision("70m-deduped", 0)
    tokenizer = model.tokenizer

    samples = []
    for tokens, seq_idx in zip(tqdm(all_data["input_ids"]), all_data["seq_idx"]):
        samples.append(
            TextSample(
                value=tokenizer.decode(tokens),
                seq_idx=seq_idx,
            )
        )
    return samples


@dataclass
class Embedding:
    value: np.ndarray
    seq_idx: int


def compute_and_cache_embeddings(
    cache_location: str | Path,
    encoder: SentenceTransformer,
    data: list[TextSample],
) -> list[Embedding]:
    # I tried a bunch of different multiprocessing approaches
    # but on my mac a stupid loop performs the best.

    Path(cache_location).parent.mkdir(parents=True, exist_ok=True)
    embeddings = []

    try:
        with shelve.open(cache_location) as shelf:
            for text_sample in tqdm(data):
                shelf_key = str(text_sample.seq_idx)

                if shelf_key in shelf:
                    embeddings.append(shelf[shelf_key])
                    continue

                embedding = Embedding(
                    value=encoder.encode(text_sample.value),
                    seq_idx=text_sample.seq_idx,
                )
                embeddings.append(embedding)
                shelf[shelf_key] = embedding
    except KeyboardInterrupt:
        pass

    return embeddings


class Embeddings:
    # (num samples, embedding dimension)
    values: np.ndarray

    # (num samples,)
    seq_idx: np.ndarray

    def __init__(self, values: np.ndarray, seq_idx: np.ndarray):
        self.values = values
        self.seq_idx = seq_idx
        self._id_map = {seq_index: i for i, seq_index in enumerate(self.seq_idx)}
        self._map_idx = np.vectorize(self._id_map.__getitem__)

    def __getitem__(self, item):
        # Accepts `seq_idx` values or arrays.
        return self.values[self._map_idx(item)]


def load_from_cache(model: str) -> Embeddings:
    cache_location = (CACHE_LOCATION / "numpy" / model).with_suffix(".npz")
    data = np.load(cache_location)
    return Embeddings(
        values=data["values"],
        seq_idx=data["seq_idx"],
    )


def main(model=ALIBABA_MODEL):
    data = load_and_detokenize_pile_subset()
    embeddings = compute_and_cache_embeddings(
        encoder=SentenceTransformer(model, trust_remote_code=True),
        cache_location=CACHE_LOCATION / model,
        data=data,
    )

    numpy_save_path = Path(CACHE_LOCATION / "numpy" / model)
    numpy_save_path.parent.mkdir(parents=True, exist_ok=True)
    seq_idx, values = zip(*[(e.seq_idx, e.value) for e in embeddings])
    np.savez(numpy_save_path, seq_idx=seq_idx, values=values)


if __name__ == "__main__":
    Fire(main)

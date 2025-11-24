from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import datasets

from generalization_profiles.embeddings import Embeddings


@dataclass
class Neighbors:
    # (num seeds,)
    seq_idx: np.ndarray

    # (num seeds, top k)
    neighbor_idx: np.ndarray

    # (num seeds, top k)
    cosine_similarity: np.ndarray


class NeighborFinder:
    def __init__(self, embeddings: Embeddings, neighbor_set_ids: list[int]):
        self.embeddings = embeddings
        self.neighbor_set_ids = np.array(neighbor_set_ids)
        self.neighbor_set = embeddings[neighbor_set_ids]

    def find_neighbors(self, seeds: np.ndarray, top_k: int) -> Neighbors:
        seed_vecs = self.embeddings[seeds]
        # (num seeds, size of neighbor set)
        similarities = cosine_similarity(seed_vecs, self.neighbor_set)
        top_args = np.argsort(similarities, axis=1)[..., ::-1][..., :top_k]
        neighbor_idx = self.neighbor_set_ids[top_args]
        similarities = np.take_along_axis(similarities, top_args, axis=1)
        return Neighbors(
            seq_idx=seeds,
            neighbor_idx=neighbor_idx,
            cosine_similarity=similarities,
        )


def find_all_neighbors(
    embeddings: Embeddings,
    neighbor_set_ids: list[int],
    top_k: int,
    batch_size: int = 1024,
    verbose: bool = True,
) -> Neighbors:
    finder = NeighborFinder(
        embeddings=embeddings, neighbor_set_ids=neighbor_set_ids
    )
    chunks: list[Neighbors] = []
    for i in tqdm(
        range(0, len(embeddings), batch_size),
        desc='Finding neighbors',
        disable=not verbose,
    ):
        id_chunk = embeddings.seq_idx[i : i + batch_size]
        chunks.append(finder.find_neighbors(id_chunk, top_k))

    return Neighbors(
        seq_idx=np.concatenate([n.seq_idx for n in chunks]),
        neighbor_idx=np.concatenate([n.neighbor_idx for n in chunks], axis=0),
        cosine_similarity=np.concatenate(
            [n.cosine_similarity for n in chunks], axis=0
        ),
    )


@dataclass
class Surprisals:
    """For a fixed model, contains the surpisal of sample n at step m."""

    # (n steps,)
    step: np.ndarray

    # (n samples,)
    seq_idx: np.ndarray

    # (n steps, n samples)
    values: np.ndarray


def _to_dataframe(data: datasets.Dataset) -> pd.DataFrame:
    # Not sure how this works but it's way faster than `pd.DataFrame(data)`
    return data.with_format('pandas')[:]  # type: ignore


def _load_surprisals_single(data: datasets.Dataset) -> Surprisals:
    df = _to_dataframe(data)
    df = df.sort_values(by=['step', 'seq_idx'])
    steps = []
    values = []
    seq_idx: np.ndarray | None = None
    for step, values_at_step in df.groupby('step', sort=False):
        steps.append(step)
        values.append(values_at_step['sup_seq'].values)

        # Make sure the seq_idx are the same for every step
        current_seq_idx = np.array(values_at_step['seq_idx'].values)
        if seq_idx is None:
            seq_idx = current_seq_idx
        else:
            assert (seq_idx == current_seq_idx).all()
            
    assert seq_idx is not None
    
    return Surprisals(
        step=np.array(steps),
        seq_idx=seq_idx,
        values=np.stack(values)
    )
    
    
def load_surprisals() -> dict[str, Surprisals]:
    dataset = datasets.load_dataset('pietrolesci/pythia-deduped-stats')
    return {
        model_name: _load_surprisals_single(data)  # type: ignore
        for model_name, data in dataset.items()  # type: ignore
    }

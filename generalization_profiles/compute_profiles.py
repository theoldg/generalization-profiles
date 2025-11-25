import typing
from dataclasses import dataclass

import datasets
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from generalization_profiles.embeddings import Embeddings


@dataclass
class Neighbors:
    # (num seeds,)
    seq_idx: np.ndarray

    # (num seeds, top k)
    neighbor_idx: np.ndarray

    # (num seeds, top k)
    cosine_similarity: np.ndarray

    def __post_init__(self):
        id_map = {seq_index: i for i, seq_index in enumerate(self.seq_idx)}
        self._map_idx = np.vectorize(id_map.__getitem__)

    def __getitem__(self, item):
        return self.neighbor_idx[self._map_idx(item)]


class NeighborFinder:
    def __init__(self, embeddings: Embeddings, neighbor_set_ids: np.ndarray):
        self.embeddings = embeddings
        self.neighbor_set_ids = neighbor_set_ids
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
    neighbor_set_ids: np.ndarray,
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

    def __post_init__(self):
        id_map = {id: i for i, id in enumerate(self.seq_idx)}
        self.map_idx = np.vectorize(id_map.__getitem__)


@typing.no_type_check
def _to_dataframe(data: datasets.Dataset) -> pd.DataFrame:
    # Not sure how this works but it's way faster than `pd.DataFrame(data)`
    return data.with_format('pandas')[:]


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
        step=np.array(steps), seq_idx=seq_idx, values=np.stack(values)
    )


@typing.no_type_check
def load_surprisals() -> dict[str, Surprisals]:
    dataset = datasets.load_dataset('pietrolesci/pythia-deduped-stats')
    return {
        model_name: _load_surprisals_single(data)
        for model_name, data in dataset.items()
    }


def compute_aggregated_surprisals(
    surprisals: Surprisals,
    embeddings: Embeddings,
    neighbor_set_ids: np.ndarray,
    top_k: int,
) -> Surprisals:
    neighbors = find_all_neighbors(
        embeddings=embeddings,
        neighbor_set_ids=neighbor_set_ids,
        top_k=top_k,
        verbose=False,
    )
    neighbor_ids = neighbors[surprisals.seq_idx]

    return Surprisals(
        step=surprisals.step,
        seq_idx=surprisals.seq_idx,
        values=np.take(
            surprisals.values,
            surprisals.map_idx(neighbor_ids),
            axis=1,
        ).mean(-1),
    )


def compute_generalization_profile(
    surprisals: Surprisals,
    embeddings: Embeddings,
    top_k: int,
    macro_batching_factor: int = 5,
    memorization_only: bool = False,
) -> np.ndarray:
    checkpoint_interval = 1000
    pythia_batch_size = 1024

    # For each seq_idx, the first checkpoint that has seen this sample
    # in training - rounded up according to macro_batching_factor.
    # The validation samples get a value of -1.
    first_seen_step = (
        (
            np.floor(
                surprisals.seq_idx
                / (
                    pythia_batch_size
                    * macro_batching_factor
                    * checkpoint_interval
                )
            ).astype(int)
            + 1
        )
        * checkpoint_interval
        * macro_batching_factor
    )
    first_seen_step[first_seen_step <= 0] = -1
    first_seen_step[first_seen_step > surprisals.step.max()] = -1

    surprisals = Surprisals(
        step=surprisals.step[::macro_batching_factor],
        values=surprisals.values[::macro_batching_factor],
        seq_idx=surprisals.seq_idx,
    )

    if memorization_only:
        neighborhood_surprisals = surprisals
    else:
        neighbor_set_ids = surprisals.seq_idx[first_seen_step == -1]
        neighborhood_surprisals = compute_aggregated_surprisals(
            surprisals=surprisals,
            embeddings=embeddings,
            neighbor_set_ids=neighbor_set_ids,
            top_k=top_k,
        )

    # Average the surprisals within each macro-batch.
    n_steps = len(neighborhood_surprisals.step)
    agg_surprisals = np.zeros((n_steps, n_steps))
    target_index = np.abs(first_seen_step) // (
        macro_batching_factor * checkpoint_interval
    )
    _, counts = np.unique(target_index, return_counts=True)
    for step_i in range(len(neighborhood_surprisals.step)):
        np.add.at(
            agg_surprisals[step_i],
            target_index,
            neighborhood_surprisals.values[step_i],
        )
    agg_surprisals /= counts[None, :]

    # agg_surprisals[i, j] is the surprisal
    # at model checkpoint i for samples first treated at step j
    # (in terms of macro-batches i and j)

    profile = np.zeros((n_steps, n_steps - 1))
    y = agg_surprisals
    for c in range(n_steps):
        for g in range(1, n_steps):
            if g <= c:
                profile[c, g - 1] = (
                    (y[c, g] - y[g - 1, g]) - (y[c, 0] - y[g - 1, 0])
                )
            else:
                profile[c, g - 1] = np.nan

    return profile

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

    def __post_init__(self):
        id_map = {seq_index: i for i, seq_index in enumerate(self.seq_idx)}
        self._map_idx = np.vectorize(id_map.__getitem__)

    def __getitem__(self, item):
        return self.neighbor_idx[self._map_idx(item)]

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

    def __post_init__(self):
        id_map = {id: i for i, id in enumerate(self.seq_idx)}
        self.map_idx = np.vectorize(id_map.__getitem__)

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


def compute_aggregated_surprisals(
        surprisals: Surprisals,
        embeddings: Embeddings,
        neighbor_set_ids: set[int],
        top_k: int,
) -> Surprisals:
    # Only those IDs for which surprisals exist.
    filtered_neighbor_set = list(neighbor_set_ids.intersection(surprisals.seq_idx))
    neighbors = find_all_neighbors(
        embeddings=embeddings,
        neighbor_set_ids=filtered_neighbor_set,
        top_k=top_k,
        verbose=False,
    )
    neighbor_ids = neighbors[surprisals.seq_idx]
    return Surprisals(
        step=surprisals.step,
        seq_idx=surprisals.seq_idx,
        # Magic.
        values=surprisals.values[
            :, surprisals.map_idx(neighbor_ids)
        ].mean(-1)
    )


# def compute_generalization_profile(
#         surprisals: Surprisals,
#         embeddings: Embeddings,
#         neighbor_set_ids: set[int],
#         top_k: int,
#         macro_batching_factor: int = 5,
# ) -> np.ndarray:
#     checkpoint_interval = 1000
#     pythia_batch_size = 1024

#     # For each seq_idx, the first checkpoint that has seen this sample
#     # in training - rounded up according to macro_batching_factor.
#     # The validation samples get a value of -1.
#     first_seen_step = (
#         np.floor(
#             surprisals.seq_idx 
#             / (pythia_batch_size * macro_batching_factor * checkpoint_interval)
#         ).astype(int)
#         + 1
#     ) * checkpoint_interval * macro_batching_factor
#     first_seen_step[first_seen_step <= 0] = -1

#     surprisals = Surprisals(
#         step=surprisals.step[::macro_batching_factor],
#         values=surprisals.values[::macro_batching_factor],
#         seq_idx=surprisals.seq_idx,
#     )

#     # For the neighbor set, only consider the samples 
#     # that have available surprisal values.
#     neighbor_set_ids = neighbor_set_ids.intersection(surprisals.seq_idx)

#     agg_surprisals = compute_aggregated_surprisals(
#         surprisals=surprisals,
#         embeddings=embeddings,
#         neighbor_set_ids=neighbor_set_ids,
#         top_k=top_k,
#     )

#     # Average the surprisals within each macro-batch.
#     x = np.zeros((len(agg_surprisals.step), len(agg_surprisals.step)))
#     target_index = np.abs(first_seen_step) // (macro_batching_factor * checkpoint_interval)
#     print(x.shape, target_index.shape, agg_surprisals.values.shape)
#     np.add.at(x, target_index, agg_surprisals.values)
#     return locals()



def seq_idx_to_step(seq_idx: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """
    Converts sequence ID to the training step it was introduced.
    Assumes seq_idx is 1-based (1..1024 -> Step 1).
    Negative seq_idx (validation) returns -1.
    """
    # Filter validation data
    steps = np.full_like(seq_idx, -1, dtype=int)
    mask = seq_idx > 0
    
    # Calculation: ceil(idx / 1024)
    # Equivalent to (idx - 1) // 1024 + 1 for integer arithmetic
    steps[mask] = (seq_idx[mask] - 1) // batch_size + 1
    return steps


@dataclass
class GeneralizationProfile:
    profile: np.ndarray
    steps: np.ndarray
    counts: np.ndarray


def compute_generalization_profile(
    surprisals: Surprisals,
    embeddings: Embeddings,
    neighbor_set_ids: set[int],
    top_k: int,
    macro_batching_factor: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # 1. Downsample steps based on macro_batching_factor
    # If original steps are [1000, 2000, 3000, 4000...] and factor is 2,
    # we keep [1000, 3000...].
    # NOTE: The resulting matrix will have dimensions (len(subsampled_steps), len(subsampled_steps))
    subsampled_steps = surprisals.step[::macro_batching_factor]
    
    surprisals_sub = Surprisals(
        step=subsampled_steps,
        values=surprisals.values[::macro_batching_factor],
        seq_idx=surprisals.seq_idx,
    )

    # 2. Compute the neighbor surprisals (Aggregated Surprisals)
    # This gives us the average loss of neighbors for every sample at every time step.
    # neighbor_set_ids must intersect with what we have surprisals for
    valid_neighbors = neighbor_set_ids.intersection(surprisals_sub.seq_idx)
    
    agg_surprisals = compute_aggregated_surprisals(
        surprisals=surprisals_sub,
        embeddings=embeddings,
        neighbor_set_ids=valid_neighbors,
        top_k=top_k,
    )

    # 3. Determine the Cohort Index for every sample
    # Calculate the exact training step the sample was introduced
    exact_train_steps = seq_idx_to_step(agg_surprisals.seq_idx)
    
    # Map the exact training step to the index in our subsampled_steps array.
    # np.searchsorted finds the index where exact_train_steps should be inserted to maintain order.
    # Since checkpoints are usually [1000, 2000...], if a sample is at step 1050, 
    # it "appears" in the model at step 2000 (the next checkpoint).
    # side='left' means if exact match, we take that index. If 1050 and list is [1000, 2000], index is 1.
    cohort_indices = np.searchsorted(subsampled_steps, exact_train_steps, side='left')
    
    # Filter out samples that:
    # a) Are validation data (exact_train_steps == -1)
    # b) Were introduced AFTER our last recorded step (cohort_indices >= len)
    n_steps = len(subsampled_steps)
    valid_mask = (exact_train_steps > 0) & (cohort_indices < n_steps)
    
    valid_cohorts = cohort_indices[valid_mask]
    valid_values = agg_surprisals.values[:, valid_mask] # Shape: (n_steps, n_valid_samples)

    # 4. Aggregate into the Matrix
    # Matrix shape: (Cohort Index, Time Index)
    # We want to average the neighbor-surprisals for all samples belonging to the same cohort.
    
    sum_matrix = np.zeros((n_steps, n_steps))
    count_matrix = np.zeros((n_steps,))
    
    # Add counts (how many samples are in each cohort)
    np.add.at(count_matrix, valid_cohorts, 1)
    
    # Add values. Since valid_values is (Steps, Samples), and we group by Samples (columns),
    # we iterate over the time dimension to use np.add.at
    for t in range(n_steps):
        np.add.at(sum_matrix[:, t], valid_cohorts, valid_values[t, :])

    # Avoid division by zero for empty cohorts
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_matrix = sum_matrix / count_matrix[:, None]
    
    # Fill empty cohorts with NaNs or 0s depending on preference (usually NaN)
    avg_matrix[np.isnan(avg_matrix)] = np.nan
    return locals()
    return GeneralizationProfile(
        profile=avg_matrix,
        steps=subsampled_steps,
        counts=count_matrix,
    )

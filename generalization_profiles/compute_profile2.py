from dataclasses import dataclass
from functools import cache
import typing

import datasets
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from generalization_profiles import pythia_facts
from generalization_profiles.embeddings import Embeddings, load_embeddings_from_cache


@dataclass
class Profile:
    """A memorization of generalization profile."""

    n_macro_batches: int

    # Shape: (n_macro_batches,)
    # The training step for each macro batch index.
    step: np.ndarray

    # Shape: (n_macro_batches, n_macro_batches)
    # values[i, j] is the profile value for the checkpoint at macro batch i
    # and treatment macro batch j.
    values: np.ndarray

    # Shape: (n_macro_batches, n_macro_batches)
    # variance[i, j] is the variance for values[i, j].
    variance: np.ndarray


@dataclass
class Surprisals:
    """For a fixed model, contains the surpisal of sample n at step m."""

    # Shape: (n steps,)
    step: np.ndarray

    # Shape: (n samples,)
    seq_idx: np.ndarray

    # Shape: (n steps, n samples)
    values: np.ndarray


@typing.no_type_check
@cache
def _load_dataset() -> dict[str, datasets.Dataset]:
    return datasets.load_dataset("pietrolesci/pythia-deduped-stats")


@typing.no_type_check
def _to_dataframe(data: datasets.Dataset) -> pd.DataFrame:
    # Not sure how/why this works but it's way faster than `pd.DataFrame(data)`
    return data.with_format("pandas")[:]


def _load_surprisals_for_variant(model_variant: str) -> Surprisals:
    data = _load_dataset()[model_variant]
    df = _to_dataframe(data)

    df = df.loc[df.seq_idx < pythia_facts.FIRST_REPEATED_SEQ_IDX]
    df = df.sort_values(by=["step", "seq_idx"])

    steps = []
    values = []
    seq_idx: np.ndarray | None = None
    for step, values_at_step in df.groupby("step", sort=False):
        steps.append(step)
        values.append(values_at_step["sup_seq"].values)

        # Make sure the seq_idx are the same for every step
        current_seq_idx = np.array(values_at_step["seq_idx"].values)
        if seq_idx is None:
            seq_idx = current_seq_idx
        else:
            assert (seq_idx == current_seq_idx).all()

    assert seq_idx is not None
    return Surprisals(step=np.array(steps), seq_idx=seq_idx, values=np.stack(values))


def _compute_profile_from_surprisals(
    surprisals: Surprisals,
) -> Profile:
    ...

def _macro_batch(
        surprisals: Surprisals,
        macro_batching_factor: int,
) -> Surprisals:
    return Surprisals(
        step=surprisals.step[::macro_batching_factor],
        values=surprisals.values[::macro_batching_factor],
        seq_idx=surprisals.seq_idx,
    )


def _aggregate_surprisals_over_neighborhoods(
        surprisals: Surprisals,
        embeddings: Embeddings,
        top_k: int,
) -> Surprisals:
    ...


def compute_memorization_profile(
    model_variant: str,
    macro_batching_factor: int,
) -> Profile:
    surprisals = _load_surprisals_for_variant(model_variant)
    surprisals = _macro_batch(surprisals, macro_batching_factor)
    return _compute_profile_from_surprisals(surprisals)

def compute_generalization_profile(
    model_variant: str,
    macro_batching_factor: int,
    top_k: int,
) -> Profile:
    surprisals = _load_surprisals_for_variant(model_variant)
    surprisals = _macro_batch(surprisals, macro_batching_factor)
    embeddings = load_embeddings_from_cache(model_variant)
    surprisals = _aggregate_surprisals_over_neighborhoods(surprisals, embeddings, top_k)
    return _compute_profile_from_surprisals(surprisals)

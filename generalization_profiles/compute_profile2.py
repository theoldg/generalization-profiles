from dataclasses import dataclass
from functools import cache
import typing

import datasets
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from differences import ATTgt

from generalization_profiles import pythia_facts
from generalization_profiles.embeddings import (
    Embeddings,
    load_embeddings_from_cache,
    ALIBABA_MODEL,
)


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

    def __post_init__(self):
        m = {id: i for i, id in enumerate(self.seq_idx)}
        self._idx_map = np.vectorize(m.__getitem__)

    def seq_idx_to_data_idx(self, seq_idx: np.ndarray) -> np.ndarray:
        """
        If sample with seq_idx=123 is stored at index 7 in values,
        then seq_idx_to_data_idx(123) = 7
        """
        return self._idx_map(seq_idx)


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

        # Everything is lined up: the array seq_idx is the same at every step.
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
    """
    Computes the memorization/generalization profile using the Callaway-Sant'Anna 
    Difference-in-Differences estimator via the 'differences' package.
    """
    # 1. Convert Surprisals matrix to a long-form DataFrame
    # surprisals.values shape: (n_steps, n_samples)
    n_steps, n_samples = surprisals.values.shape
    
    # Flatten the values and create corresponding index arrays
    flat_values = surprisals.values.flatten()
    step_indices = np.repeat(surprisals.step, n_samples)
    seq_indices = np.tile(surprisals.seq_idx, n_steps)

    df = pd.DataFrame({
        "step": step_indices,
        "seq_idx": seq_indices,
        "surprisal": flat_values
    })

    # 2. Assign Cohorts
    # Following compute_attgt.py logic: 
    # seq_idx > 0 are treated units. Their 'cohort' is the step they enter training.
    # In this context, we treat the 'seq_idx' as the indicator of which macro-batch 
    # the sample belongs to.
    
    # Define a mapping or logic for cohorts. 
    # If seq_idx corresponds to the training step directly:
    df["cohort"] = np.where(df["seq_idx"] > 0, df["seq_idx"], np.nan)
    
    # 3. Fit the Diff-in-Diff Model
    # We use 'seq_idx' as the unit (individual sample) and 'step' as time.
    att_model = ATTgt(
        data=df.set_index(["seq_idx", "step"]), 
        cohort_name="cohort"
    )
    
    # We use 'dr' (doubly robust) and 'never_treated' (seq_idx <= 0) as control
    att_results = att_model.fit(
        target_col="surprisal",
        est_method="dr",
        control_group="never_treated",
        n_jobs=-1 # Use all available cores
    )

    # 4. Extract results into Profile
    # att_results typically contains (cohort, time) as indices.
    # We need to map these back to the grid defined by our macro-batches.
    
    # Pivot results to get the (n_macro_batches, n_macro_batches) shape
    # Values: ATT(g, t)
    # Variance: (Standard Error)^2
    res_df = att_results.reset_index()
    
    # Ensure we only include steps and cohorts present in our macro-batched surprisals
    pivot_values = res_df.pivot(index="cohort", columns="step", values=("point_estimate", "surprisal"))
    pivot_std = res_df.pivot(index="cohort", columns="step", values=("standard_error", "surprisal"))

    # Clean up column/index levels from pivot
    pivot_values.columns = pivot_values.columns.droplevel(0)
    pivot_std.columns = pivot_std.columns.droplevel(0)

    return Profile(
        n_macro_batches=len(surprisals.step),
        step=surprisals.step,
        values=pivot_values.to_numpy(),
        variance=np.square(pivot_std.to_numpy())
    )
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
    validation_idx = surprisals.seq_idx[surprisals.seq_idx < 0]
    validation_embeddings = embeddings[validation_idx]
    all_embeddings = embeddings[surprisals.seq_idx]

    # If we increase the data size, this will have to be batched.
    similarities = cosine_similarity(all_embeddings, validation_embeddings)

    top_args = np.argpartition(similarities, -top_k, axis=1)[:, -top_k:]
    neighbor_idx = validation_idx[top_args]

    aggregated_values = np.take(
        surprisals.values,
        surprisals.seq_idx_to_data_idx(neighbor_idx),
        axis=1,
    ).mean(-1)

    return Surprisals(
        step=surprisals.step,
        seq_idx=surprisals.seq_idx,
        values=aggregated_values,
    )


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
    embedding_model: str = ALIBABA_MODEL,
) -> Profile:
    surprisals = _load_surprisals_for_variant(model_variant)
    surprisals = _macro_batch(surprisals, macro_batching_factor)
    embeddings = load_embeddings_from_cache(embedding_model)
    surprisals = _aggregate_surprisals_over_neighborhoods(surprisals, embeddings, top_k)
    return _compute_profile_from_surprisals(surprisals)


if __name__ == "__main__":
    compute_generalization_profile(
        model_variant="70m",
        macro_batching_factor=10,
        top_k=8,
    )

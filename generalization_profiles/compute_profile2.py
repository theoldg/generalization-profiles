from dataclasses import dataclass
from functools import cache
import typing
import warnings

import datasets
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
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
    # std[i, j] is the standard deviation for values[i, j].
    std: np.ndarray


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
    macro_batching_factor: int,
) -> Profile:
    """
    Computes the DiD using `differences`.
    """
    # Adapt surprisals to expected format.
    df_data = {"step": [], "seq_idx": [], "surprisal": []}
    for (step_index, seq_idx_index), surprisal in np.ndenumerate(surprisals.values):
        df_data["step"].append(surprisals.step[step_index])
        df_data["seq_idx"].append(surprisals.seq_idx[seq_idx_index])
        df_data["surprisal"].append(surprisal)
    df = pd.DataFrame(df_data)
    macro_batch_size = (
        pythia_facts.BATCH_SIZE
        * pythia_facts.CHECKPOINT_INTERVAL
        * macro_batching_factor
    )
    # The cohort is the macro batch index (and NaN for validation samples).
    df["cohort"] = np.where(
        df["seq_idx"] > 0,
        1 + df["seq_idx"] // macro_batch_size,
        np.nan,
    )
    att_model = ATTgt(data=df.set_index(["seq_idx", "step"]), cohort_name="cohort")
    att_results = att_model.fit(
        "surprisal",
        est_method="dr",
        control_group="never_treated",
        # n_jobs=-1,
    )

    att_results.columns = att_results.columns.droplevel([0, 1])
    res_df = att_results.reset_index()

    return res_df

    pivot_values = res_df.pivot(index="cohort", columns="time", values="ATT")
    pivot_std = res_df.pivot(index="cohort", columns="time", values="std_error")

    # 6. Reindex to ensure the matrix is square and aligned with surprisals.step
    # Some (cohort, time) pairs might be missing or NaN (e.g., pre-treatment)
    full_index = surprisals.step[surprisals.step > 0] # Training steps
    
    # Reindex and fill with 0 or NaN as appropriate for your profile
    pivot_values = pivot_values.reindex(index=full_index, columns=surprisals.step).to_numpy()
    pivot_std = np.square(
        pivot_std.reindex(index=full_index, columns=surprisals.step).to_numpy()
    )

    return Profile(
        n_macro_batches=len(surprisals.step),
        step=surprisals.step,
        values=pivot_values,
        std=pivot_std
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
    return _compute_profile_from_surprisals(surprisals, macro_batching_factor)


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
    return _compute_profile_from_surprisals(surprisals, macro_batching_factor)


if __name__ == "__main__":
    compute_generalization_profile(
        model_variant="70m",
        macro_batching_factor=10,
        top_k=8,
    )

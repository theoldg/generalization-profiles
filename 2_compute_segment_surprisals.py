from dataclasses import dataclass
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from fire import Fire
import numpy as np
from tqdm.auto import tqdm
import pandas as pd

from generalization_profiles.pythia import MODEL_VARIANTS


"""
A cool 500 GB, downloaded via:

huggingface-cli download \
    "pietrolesci/pythia-deduped-stats-raw" \
    --repo-type dataset \
    --local-dir ./data \
    --max-workers 32
"""
DATA_DIR = Path("data")


@dataclass
class VariantStep:
    variant: str
    step: int
    parquet_path: Path


def get_steps_for_variant(variant: str) -> list[VariantStep]:
    variant_dir = DATA_DIR / variant
    assert variant_dir.exists()
    step_paths = sorted(variant_dir.glob("*/*.parquet"))
    assert step_paths

    def _get_step(p: Path) -> int:
        m = re.search(r"-step(\d+)\.", p.name)
        assert m, str(p)
        return int(m.groups()[0])

    return [
        VariantStep(
            variant=variant,
            step=_get_step(p),
            parquet_path=p,
        )
        for p in step_paths
    ]


def compute_surprisals(
    raw_stats_df: pd.DataFrame,
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    assert (segments_df.start_idx == 0).sum() == 0
    dataset_size = len(raw_stats_df)

    # Maps [seq_idx] -> [index in raw_stats_df].
    seq_idx_to_seq_i = {idx: i for i, idx in enumerate(raw_stats_df.seq_idx)}
    # Surprisals are not available for all segments.
    segments_df = (
        segments_df
        .loc[segments_df.seq_idx.isin(set(seq_idx_to_seq_i))]
        .reset_index(drop=True)
    )
    # For each segment in segments_df, the index of the corresponding
    # row in raw_stats_df.
    # Shape: (n_segments,)
    segment_seq_i = segments_df.seq_idx.map(seq_idx_to_seq_i.__getitem__)

    # Sanity check
    assert (raw_stats_df.seq_idx.values[segment_seq_i] == segments_df.seq_idx.values).all()

    # Shape: (dataset_size, seq_length)
    raw_surprisals = np.stack(raw_stats_df.sup.values)  # type: ignore
    # Shape: (dataset_size, 1 + seq_length)
    sup_cumsum = np.hstack((np.zeros((dataset_size, 1)), raw_surprisals.cumsum(-1)))

    # Shape: (n_segments,)
    start = segments_df.start_idx - 1
    assert (start >= 0).all()
    # Shape: (n_segments,)
    end = start + segments_df.num_tokens

    segment_total_sup = (
        sup_cumsum[segment_seq_i, end] - sup_cumsum[segment_seq_i, start]
    )
    avg_sup = segment_total_sup / segments_df.num_tokens.values
    return pd.DataFrame(
        {
            "seq_idx": segments_df.seq_idx,
            "start_idx": segments_df.start_idx,
            "num_tokens": segments_df.num_tokens,
            "avg_sup": avg_sup,
        }
    )


def process_step(variant_step: VariantStep):
    target_path = f"results/segment_surprisals/{variant_step.variant}/{variant_step.step}.parquet"
    target_path = Path(target_path)
    if target_path.exists():
        raise FileExistsError(str(target_path))
    target_path.parent.mkdir(exist_ok=True, parents=True)
    raw_stats_df = pd.read_parquet(variant_step.parquet_path)
    assert SEGMENTS_DF is not None
    sup_df = compute_surprisals(raw_stats_df, SEGMENTS_DF)
    sup_df.to_parquet(target_path)


def load_segments(segments_path: str):
    global SEGMENTS_DF
    SEGMENTS_DF = pd.read_parquet(segments_path)


if __name__ == "__main__":
    SEGMENTS_DF = None
    Fire(load_segments)
    assert isinstance(SEGMENTS_DF, pd.DataFrame)

    SEGMENTS_DF = SEGMENTS_DF.loc[SEGMENTS_DF.start_idx != 0].reset_index(drop=True)

    steps = []
    for variant in MODEL_VARIANTS:
        steps.extend(get_steps_for_variant(variant))
    
    with ThreadPoolExecutor(32) as executor:
        futures = set()
        for step in steps:
            futures.add(executor.submit(process_step, step))
        
        for f in tqdm(as_completed(futures), total=len(futures)):
            f.result()

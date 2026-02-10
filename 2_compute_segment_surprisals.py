from dataclasses import dataclass
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fire import Fire
import numpy as np
import datasets
from huggingface_hub import HfFileSystem
import pandas as pd

from generalization_profiles.pythia import MODEL_VARIANTS, FIRST_STEP_OF_SECOND_EPOCH

datasets.disable_caching()


@dataclass(order=True)
class VariantStep:
    variant: str
    step: int
    data_dir: str


def get_steps_for_variant(variant: str) -> list[VariantStep]:
    fs = HfFileSystem()
    prefix = "datasets/pietrolesci/pythia-deduped-stats-raw/"
    ls = fs.ls(f"{prefix}{variant}/")
    ret = []
    for item in ls:
        data_dir = item["name"].removeprefix(prefix)
        step = int(re.search(r"step(\d+)", data_dir).groups()[0])
        ret.append(
            VariantStep(
                variant=variant,
                step=step,
                data_dir=data_dir,
            )
        )
    ret = [r for r in ret if r.step < FIRST_STEP_OF_SECOND_EPOCH]
    return sorted(ret)


def compute_surprisals(
    raw_stats_df: pd.DataFrame,
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    assert (segments_df.start_idx == 0).sum() == 0
    dataset_size = len(raw_stats_df)

    seq_idx_to_seq_i = {idx: i for i, idx in enumerate(raw_stats_df.seq_idx)}
    # Surprisals are not available for all segments    
    segments_df = segments_df.loc[segments_df.seq_idx.isin(set(seq_idx_to_seq_i))]
    segment_seq_i = segments_df.seq_idx.map(seq_idx_to_seq_i.__getitem__)
    
    # shape: (dataset_size, seq_length)
    sup = np.stack(raw_stats_df.sup.values)  # type: ignore
    sup_cumsum = np.hstack((np.zeros((dataset_size, 1)), sup.cumsum(-1)))
    
    start = segments_df.start_idx - 1
    assert (start >= 0).all()
    end = segments_df.start_idx + segments_df.num_tokens - 1
    
    segment_total_sup = (
        sup_cumsum[segment_seq_i, end] - sup_cumsum[segment_seq_i, start]
    )
    avg_sup = segment_total_sup / segments_df.num_tokens  
    return pd.DataFrame(
        {
            "seq_idx": segments_df.seq_idx,
            "start_idx": segments_df.start_idx,
            "num_tokens": segments_df.num_tokens,
            "avg_sup": avg_sup,
        }
    )


def process_step(step: VariantStep):
    print("Processing", step.data_dir)
    target_path = f"results/segment_surprisals/{step.variant}/{step.step}.parquet"
    target_path = Path(target_path)
    if target_path.exists():
        print("SKIPPING BECAUSE EXISTS:", target_path)
        return
    target_path.parent.mkdir(exist_ok=True, parents=True)

    ds = datasets.load_dataset(
        "pietrolesci/pythia-deduped-stats-raw",
        data_dir=step.data_dir,
    )

    raw_stats_df: pd.DataFrame = ds["train"].with_format("pandas")[:]  # type: ignore
    assert SEGMENTS_DF is not None
    sup_df = compute_surprisals(raw_stats_df, SEGMENTS_DF)
    sup_df.to_parquet(target_path)
    ds.cleanup_cache_files()


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

    with ThreadPoolExecutor(16) as executor:
        executor.map(process_step, steps)

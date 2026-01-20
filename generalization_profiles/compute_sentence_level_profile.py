from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from differences import ATTgt

from generalization_profiles.aggregate_surprisals import K_VALUES
from generalization_profiles.pythia import MODEL_VARIANTS
from generalization_profiles import pythia


def build_df(
    k: int,
    model: str,
    source_segments: pd.DataFrame,
):
    assert k in K_VALUES
    assert model in MODEL_VARIANTS

    ki = list(K_VALUES).index(k)
    model_agg_dir = Path("results/aggregated_surprisals") / model
    assert model_agg_dir.exists()
    paths = list(model_agg_dir.glob("*.npy"))

    def _load_agg(p: Path):
        step = int(p.with_suffix("").name)
        return step, np.copy(np.load(p, mmap_mode="r")[:, ki])

    with ThreadPoolExecutor(8) as executor:
        loaded = list(tqdm(executor.map(_load_agg, paths)))

    dfs = []
    for step, sup in loaded:
        s = source_segments[["seq_idx"]].copy()
        s["step"] = step
        s["sup_seq"] = sup
        s["seg_i"] = np.arange(len(s))
        dfs.append(s)

    return pd.concat(dfs)


if __name__ == "__main__":
    source_segments = pd.read_parquet("results/neighbors/source_segments.parquet")
    df = build_df(k=16, model="6.9b", source_segments=source_segments)

    macro_batching_factor = 1
    # The cohort is the macro batch index (and NaN for validation samples).
    macro_batch_size = (
        pythia.BATCH_SIZE * pythia.CHECKPOINT_INTERVAL * macro_batching_factor
    )
    df["cohort"] = (
        (1 + df["seq_idx"] // macro_batch_size).astype(int)
        * pythia.CHECKPOINT_INTERVAL
        * macro_batching_factor
    ).astype(int)
    df.loc[df["cohort"] <= 0, "cohort"] = np.nan
    att_model = ATTgt(data=df.set_index(["seg_i", "step"]), cohort_name="cohort")
    att_results = att_model.fit(
        "sup_seq",
        est_method="dr",
        # est_method="reg",
        control_group="never_treated",
        n_jobs=-1,
    )

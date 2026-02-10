# Only way I found to silence warnings when importing `differences`.
import os
import warnings

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
os.environ["PYTHONWARNINGS"] = (
    "ignore:pkg_resources is deprecated:UserWarning:property_cached,"
)

from typing import Literal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from differences import ATTgt
from fire import Fire
import pickle

from generalization_profiles.compute_profile import posprocess_attgt
from aggregate_surprisals import K_VALUES
from generalization_profiles.pythia import MODEL_VARIANTS
from generalization_profiles import pythia


def build_df(
    k: int,
    model: str,
    source_segments: pd.DataFrame,
    sampling_ratio: float,
    verbose: bool = False,
):
    assert k in K_VALUES
    assert model in MODEL_VARIANTS

    ki = list(K_VALUES).index(k)
    model_agg_dir = Path("results/aggregated_surprisals") / model
    assert model_agg_dir.exists()

    def _get_step(p: Path) -> int:
        return int(p.with_suffix("").name)

    paths = [
        p for p in model_agg_dir.glob("*.npy")
        if _get_step(p) < pythia.FIRST_STEP_OF_SECOND_EPOCH
    ]

    sampling_mask = np.random.random(size=len(source_segments)) < sampling_ratio
    source_segments = source_segments.loc[sampling_mask]

    def _load_agg(p: Path):
        data = np.copy(
            np.load(p, mmap_mode="r")[sampling_mask, ki]
        )
        return _get_step(p), data

    with ThreadPoolExecutor(8) as executor:
        loaded = list(tqdm(
            executor.map(_load_agg, paths),
            desc='Loading neighborhood surprisals',
            disable=not verbose,
        ))

    dfs = []
    for step, sup in loaded:
        s = source_segments[["seq_idx"]].copy()
        s["step"] = step
        s["sup_seq"] = sup
        s["seg_i"] = np.arange(len(s))
        dfs.append(s)

    return pd.concat(dfs)


def create_profile(
    k: int,
    model: str,
    sampling_ratio: float,
    output_dir: str | Path = 'results/sentence_level_profiles',
    if_exists: Literal['skip', 'replace', 'error'] = 'replace',
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f'{model}_k={k}.pkl'

    if target_path.exists():
        match if_exists:
            case 'skip':
                return
            case 'replace':
                pass
            case 'error':
                raise FileExistsError(f'File already exists: {target_path}')
            case _:
                raise ValueError(f'Incorrect argument {if_exists = }')

    source_segments = pd.read_parquet("results/neighbors/source_segments.parquet")
    df = build_df(
        k=k,
        model=model,
        source_segments=source_segments,
        sampling_ratio=sampling_ratio,
    )

    # The cohort is the batch index (and NaN for validation samples).
    macro_batch_size = pythia.BATCH_SIZE * pythia.CHECKPOINT_INTERVAL
    df["cohort"] = (
        (1 + df["seq_idx"] // macro_batch_size).astype(int)
        * pythia.CHECKPOINT_INTERVAL
    ).astype(int)
    df.loc[df["cohort"] <= 0, "cohort"] = np.nan
    att_model = ATTgt(data=df.set_index(["seg_i", "step"]), cohort_name="cohort")
    att_results = att_model.fit(
        "sup_seq",
        est_method="dr",
        control_group="never_treated",
        n_jobs=-1,
    )

    profile = posprocess_attgt(att_results)
    target_path.write_bytes(pickle.dumps(profile))


if __name__ == "__main__":
    Fire(create_profile)

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from tqdm.auto import tqdm


K_VALUES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])


def set_seg_id(df):
    # A bit of a hack but hey it's fast
    df.index = (df.seq_idx.values << 32) | (df.start_idx.values << 16)| (df.num_tokens.values)


def _aggregate_surprisals_for_checkpoint(path: Path):
    *_, model_size, checkpoint = path.with_suffix('').parts
    target_path = Path('results') / 'aggregated_surprisals' / model_size / f'{checkpoint}.npy'
    if target_path.exists():
        return

    # Very short, very magic.
    some_surprisals = pd.read_parquet(path)
    set_seg_id(some_surprisals)
    valid_surprisals_df = some_surprisals.loc[VALID_SEGMENTS.index]
    valid_surprisals = valid_surprisals_df.avg_sup.values
    x = valid_surprisals[NEIGHBOR_IDX]
    x = x.cumsum(1)
    x = x[:, K_VALUES - 1]
    x /= K_VALUES[None, :]
    x = x.astype(np.float32)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_path, x)


if __name__ == '__main__':
    SOURCE_SEGMENTS = pd.read_parquet('results/neighbors/source_segments.parquet')
    NEIGHBOR_IDX = np.vstack(SOURCE_SEGMENTS.neighbor_idx.values)  # type: ignore
    VALID_SEGMENTS = pd.read_parquet('results/neighbors/valid_segments.parquet')
    set_seg_id(VALID_SEGMENTS)

    paths = list(Path('results/segment_surprisals').glob('**/*.parquet'))
    with ThreadPoolExecutor(8) as executor:
        results = list(tqdm(
            executor.map(_aggregate_surprisals_for_checkpoint, paths),
            total=len(paths),
        ))

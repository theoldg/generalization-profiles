from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, ThreadPoolExecutor
from tqdm.auto import tqdm

from generalization_profiles import pythia


def load_test_seq_idx():
    s = Path('results/test_seq_idx.txt').read_text()
    return [int(line) for line in s.splitlines()]


def find_neighbors(seed_set, neighbor_set, k=256):
    # Returns indices into the neighbor set.
    sims = cosine_similarity(seed_set, neighbor_set)
    top_args = np.argpartition(sims, -k, axis=1)[:, -k:]
    top_similarities = np.take_along_axis(sims, top_args, axis=1)
    assert (top_similarities.max(1) == sims.max(1)).all()
    sorted_sub_idx = (-top_similarities).argsort(1)
    top_args = np.take_along_axis(
        top_args,
        sorted_sub_idx,
        axis=1,
    )
    top_similarities = np.take_along_axis(
        top_similarities,
        sorted_sub_idx,
        axis=1,
    )
    return top_args, top_similarities


def _mp_find_neighbors(i, batch_size, embds, valid_embds):
    top_args, top_sims = find_neighbors(
        embds,
        valid_embds,
    )
    return i, batch_size, top_args, top_sims


if __name__ == '__main__':
    idx = load_test_seq_idx()
    embds = np.load('results/segments_dedup_embeddings.npy')
    segments = pd.read_parquet('results/segments_dedup.parquet')

    valid_mask = segments.seq_idx.isin(idx)
    valid_segments = segments.loc[valid_mask].reset_index(drop=True)
    valid_embds = embds[valid_mask]

    source_mask = (
        (segments.seq_idx > 0) 
        & (segments.seq_idx < pythia.FIRST_REPEATED_SEQ_IDX)
    ) | valid_mask
    source_embds = embds[source_mask]
    
    ###
    # source_embds = source_embds[:3000]
    ###

    print(f'Num validation samples: {valid_mask.sum()}')

    batch_size = 1024
    all_infos = []
    n_processes = 16

    with (
        ProcessPoolExecutor(n_processes) as executor,
        # ThreadPoolExecutor(n_processes) as executor,
        tqdm(total=1 + len(source_embds) // batch_size) as pbar,
    ):
        futures = set()
        i = 0
        while True:
            for _ in range(2 * n_processes - len(futures)):
                batch_embds = source_embds[i:i+batch_size]
                if len(batch_embds) == 0:
                    break
                f = executor.submit(
                        _mp_find_neighbors,
                        i,
                        batch_size,
                        batch_embds,
                        valid_embds,
                    )
                futures.add(f)
                i += batch_size

            if len(futures) == 0:
                break
            
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in completed:
                pbar.update()
                result = f.result()
                all_infos.append(result)

    Path('results/neighbor_info_raw.pkl').write_bytes(
        pickle.dumps(all_infos)
    )

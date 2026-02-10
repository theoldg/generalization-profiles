from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import re
import os

from fire import Fire
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from generalization_profiles import compute_profile
from generalization_profiles import pythia


def find_neighbors(seed_set, neighbor_set, k):
    # Returns indices into the neighbor set.
    sims = cosine_similarity(seed_set, neighbor_set)

    # This is actually very dank, it finds the unordered top k
    # using a heap and then sorts them later.
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


def is_sentence_valid(t: str) -> bool:
    """Done this late in the pipeline so that it can be easily replaced."""
    if len(t) < 15:
        return False
    
    if len(t) > 500:
        return False

    # Removes most code
    if '  ' in t or '\t' in t[1:]:
        return False
    
    # Not more than 30% of all chars are digits 
    # (DOIs and stuff)
    num_digits = sum(char.isdigit() for char in t)
    if num_digits > 0.3 * len(t):
        return False
    
    # Should have at least one word starting with a lowercase letter.
    if not re.search(r' [a-z]', t):
        return False

    return True



def choose_seq_idx():
    datasets = list(compute_profile.load_dataset().values())
    
    common_idx = set(datasets[0].seq_idx.unique())
    for surprisals_df in datasets[1:]:
        common_idx &= set(surprisals_df.seq_idx.unique())

    source_idx = {i for i in common_idx if i < pythia.FIRST_REPEATED_SEQ_IDX}
    valid_idx = {i for i in source_idx if i < 0}

    return {
        'source': source_idx,
        'valid': valid_idx,
    }


def _mp_task(
    start_index: int,
    end_index: int,
    source_embds_path: Path,
    valid_embds_path: Path,
    output_path: Path,
    n_neighbors: int,
):
    args_output_path = output_path / f'args_{start_index}_{end_index}.npy'
    sims_output_path = output_path / f'sims_{start_index}_{end_index}.npy'

    if args_output_path.exists() and sims_output_path.exists():
        return

    source_embds = np.load(source_embds_path, mmap_mode='r')
    valid_embds = np.load(valid_embds_path, mmap_mode='r')

    args, sims = find_neighbors(
        source_embds[start_index:end_index],
        valid_embds,
        k=n_neighbors,
    )

    np.save(args_output_path, args)
    np.save(sims_output_path, sims)


def _find_files(dir: Path, prefix: str):
    def parse_lower_bound(p: Path):
        return int(re.search(r'_(\d+)_\d+.npy', p.name).groups()[0])
    files = list(dir.glob(f'{prefix}*'))
    return sorted(files, key=parse_lower_bound)


def _load_and_stack(files: list[Path]) -> np.ndarray:
    with ThreadPoolExecutor(32) as executor:
        arrays = list(
            tqdm(
                executor.map(np.load, files),
                desc='Collecting files',
                total=len(files),
            )
        )
    return np.vstack(arrays)



def main(
    segments_parquet: Path = Path('results/segments.parquet'),
    embeddings_npy: Path = Path('results/embeddings/embeddings.npy'),
    n_neighbors: int = 1024,
    output_dir: Path = Path('results/neighbors'),
):
    segments_df = pd.read_parquet(segments_parquet)
    embeddings = np.load(embeddings_npy, mmap_mode='r')
    assert len(segments_df) == len(embeddings)

    is_valid = segments_df.text.map(is_sentence_valid)
    is_valid &= segments_df.start_idx != 0
    is_valid &= (segments_df.start_idx + segments_df.num_tokens) != 2049

    intermediate_output_dir = output_dir / 'intermediate'
    intermediate_output_dir.mkdir(parents=True, exist_ok=True)

    chosen_seq_idx = choose_seq_idx()

    valid_mask = segments_df.seq_idx.isin(chosen_seq_idx['valid']) & is_valid
    print(f'Validation samples: {valid_mask.sum():,}')
    valid_segments = segments_df.loc[valid_mask]
    valid_embds = embeddings[valid_mask]
    valid_embds_path = intermediate_output_dir / 'valid_embds.npy'
    np.save(valid_embds_path, valid_embds)

    source_mask = segments_df.seq_idx.isin(chosen_seq_idx['source']) & is_valid
    print(f'Source samples: {source_mask.sum():,}')
    source_segments = segments_df.loc[source_mask]
    source_embds = embeddings[source_mask]
    source_embds_path = intermediate_output_dir / 'source_embds.npy'
    np.save(source_embds_path, source_embds)

    mp_output_path = intermediate_output_dir / 'mp_output'
    mp_output_path.mkdir(exist_ok=True)
    
    batch_size = 1024
    n_processes = min(os.cpu_count() or 1, 32)
    with ProcessPoolExecutor(n_processes) as executor:
        futures = set()
        for i in range(0, len(source_embds), batch_size):
            f = executor.submit(
                _mp_task,
                start_index=i,
                end_index=i + batch_size,
                source_embds_path=source_embds_path,
                valid_embds_path=valid_embds_path,
                output_path=mp_output_path,
                n_neighbors=n_neighbors,
            )
            futures.add(f)
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc='Computing neighbors'):
            pass

    # Luckily just about fits in ram.
    args = _load_and_stack(_find_files(mp_output_path, 'args'))
    sims = _load_and_stack(_find_files(mp_output_path, 'sims'))

    source_segments['neighbor_idx'] = list(args)
    source_segments['neighbor_similarity'] = list(sims)

    source_segments.to_parquet(output_dir / 'source_segments.parquet')
    valid_segments.to_parquet(output_dir / 'valid_segments.parquet')


if __name__ == '__main__':
    Fire(main)

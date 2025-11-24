from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from generalization_profiles.embeddings import Embeddings


@dataclass
class Neighbors:
    # (num seeds, top k)
    seq_idx: np.ndarray

    # (num seeds, top k)
    cosine_similarity: np.ndarray


class NeighborFinder:
    def __init__(self, embeddings: Embeddings, neighbor_set_ids: list[int]):
        self.embeddings = embeddings
        self.neighbor_set_ids = np.array(neighbor_set_ids)
        self.neighbor_set = embeddings[neighbor_set_ids]

    def find_neighbors(self, seeds: np.ndarray, top_k: int) -> Neighbors:
        seed_vecs = self.embeddings[seeds]
        # (num seeds, size of neighbor set)
        similarities = cosine_similarity(seed_vecs, self.neighbor_set)
        top_args = np.argsort(similarities, axis=1)[..., ::-1][..., :top_k]
        neighbor_idx = self.neighbor_set_ids[top_args]
        similarities = np.take_along_axis(similarities, top_args, axis=1)
        return Neighbors(
            seq_idx=neighbor_idx,
            cosine_similarity=similarities,
        )


def find_all_neighbors(
    embeddings: Embeddings,
    neighbor_set_ids: list[int],
    top_k: int,
    batch_size: int = 1024,
    verbose: bool = True,
) -> Neighbors:
    finder = NeighborFinder(
        embeddings=embeddings, neighbor_set_ids=neighbor_set_ids
    )
    chunks: list[Neighbors] = []
    for i in tqdm(
        range(0, len(embeddings), batch_size),
        desc='Finding neighbors',
        disable=not verbose,
    ):
        id_chunk = embeddings.seq_idx[i : i + batch_size]
        chunks.append(finder.find_neighbors(id_chunk, top_k))

    print(
        f"""
        {chunks[0].seq_idx.shape = }
        {chunks[0].cosine_similarity.shape = }            
    """
    )
    return Neighbors(
        seq_idx=np.concatenate([n.seq_idx for n in chunks], axis=0),
        cosine_similarity=np.concatenate(
            [n.cosine_similarity for n in chunks], axis=0
        ),
    )

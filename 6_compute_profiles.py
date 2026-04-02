import itertools
from typing import Literal

from tqdm.auto import tqdm
from fire import Fire

from generalization_profiles import compute_sentence_level_profile
from generalization_profiles.neighborhood_aggregation import K_VALUES
from generalization_profiles.pythia import MODEL_VARIANTS


def main(
        sampling_ratio: float,
        n_jobs: int = 32,
        if_exists: Literal['return', 'replace', 'error'] = 'replace'
):
    with tqdm(total=len(K_VALUES) * len(MODEL_VARIANTS)) as pbar:
        for model in MODEL_VARIANTS:
            for ki, k_value in enumerate(K_VALUES):
                compute_sentence_level_profile.create_profile(
                    k=k_value,
                    ki=ki,
                    model=model,
                    sampling_ratio=sampling_ratio,
                    if_exists=if_exists,
                    n_jobs=n_jobs,
                )
                pbar.update()


if __name__ == '__main__':
    Fire(main)

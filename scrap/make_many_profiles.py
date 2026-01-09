import pickle
from pathlib import Path
import itertools
from tqdm import tqdm

from generalization_profiles.pythia import MODEL_VARIANTS
from generalization_profiles.compute_profile import (
    compute_generalization_profile,
    compute_memorization_profile,
    Profile,
)


def save_profile(profile: Profile, path: str):
    full_path = Path('profiles') / path
    assert not full_path.exists()
    full_path.write_bytes(
        pickle.dumps(
            {
                "step": profile.step,
                "values": profile.values,
                "std_error": profile.std_error,
            }
        )
    )


def main():
    top_k_list = [1, 2, 4, 8, 16, 32, 64]
    macro_batching_list = [1, 2, 3, 4, 5]

    total_iters = len(macro_batching_list) * len(MODEL_VARIANTS) * (len(top_k_list) + 1)

    with tqdm(total=total_iters) as pbar:

        for b_val, model_val in itertools.product(macro_batching_list, MODEL_VARIANTS):
            filename = f"mem_{model_val}_{b_val}.pkl"
            mem_profile = compute_memorization_profile(
                model_variant=model_val,
                macro_batching_factor=b_val,
            )
            save_profile(mem_profile, filename)
            pbar.update()

            for k_val in top_k_list:
                filename = f"GEN_{model_val}_{b_val}_{k_val}.pkl"
                gen_profile = compute_generalization_profile(
                    model_variant=model_val,
                    top_k=k_val,
                    macro_batching_factor=b_val,
                )
                save_profile(gen_profile, filename)
                pbar.update()


if __name__ == '__main__':
    main()

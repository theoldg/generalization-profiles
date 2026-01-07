import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fire import Fire
from tqdm.auto import tqdm

from generalization_profiles import compute_profiles, embeddings, pythia_model


def plot_profile_on_ax(data: compute_profiles.Profile, ax1, ax2, palette="Reds"):
    """Helper to plot on a specific matplotlib axis"""
    x = data.values
    x = -x[1:][::-1]
    ax1.imshow(x, cmap=palette, aspect="auto")
    ax1.tick_params(
        left=False,
        bottom=False,
        top=False,
        labelleft=False,
        labelbottom=False,
        labeltop=False,
    )

    x = np.sqrt(data.variance)
    x = -x[1:][::-1]
    ax2.imshow(x, cmap=palette, aspect="auto")
    ax2.tick_params(
        left=False,
        bottom=False,
        top=False,
        labelleft=False,
        labelbottom=False,
        labeltop=False,
    )


def main(
    macro_batching_factor: int = 2,
    k_vals: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
    embeddings_name: str = embeddings.ALIBABA_MODEL,
    output_file: str = "plots/1.png",
):
    assert output_file.endswith(".png")
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)

    all_surprisals = compute_profiles.load_surprisals()
    assert set(all_surprisals.keys()) == set(pythia_model.MODEL_VARIANTS)

    embd = embeddings.load_embeddings_from_cache(embeddings_name)

    n_rows = len(pythia_model.MODEL_VARIANTS)
    n_cols = len(k_vals) + 1

    _, axes = plt.subplots(
        n_rows,
        n_cols * 2,
        figsize=(n_cols * 4, n_rows * 2),
        squeeze=False,
    )

    with tqdm(total=n_rows * n_cols) as pbar:
        for row_idx, model in enumerate(pythia_model.MODEL_VARIANTS):
            pbar.set_description(f"Processing model: {model}")
            surprisals = all_surprisals[model]
            mem_data = compute_profiles.compute_generalization_profile(
                surprisals=surprisals,
                embeddings=embd,
                top_k=1,
                memorization_only=True,
                macro_batching_factor=macro_batching_factor,
            )
            pbar.update()
            plot_profile_on_ax(
                mem_data, axes[row_idx, 0], axes[row_idx, 1], palette="Blues"
            )
            axes[row_idx, 0].set_ylabel(model, fontsize=12, fontweight="bold")  # type: ignore
            for col_idx, k in enumerate(k_vals):
                k_data = compute_profiles.compute_generalization_profile(
                    surprisals=surprisals,
                    embeddings=embd,
                    top_k=k,
                    macro_batching_factor=macro_batching_factor,
                    memorization_only=False,
                )
                pbar.update()
                plot_profile_on_ax(
                    k_data,
                    axes[row_idx, 2 * col_idx + 2],
                    axes[row_idx, 2 * col_idx + 3],
                    palette="Reds",
                )

    axes[0, 0].set_title("Memorization")  # type: ignore
    for col_idx, k in enumerate(k_vals):
        axes[0, col_idx + 1].set_title(f"k={k}")  # type: ignore

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    Fire(main)

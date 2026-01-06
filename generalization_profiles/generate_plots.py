import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fire import Fire
from tqdm.auto import tqdm

from generalization_profiles import compute_profiles, embeddings, pythia_model


def transform_matrix_for_plotting(data: np.ndarray, power: float = 1.2):
    # Try spinning, that's a good trick.
    data = -data[1:, 1:].T[::-1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = data ** power
    return data


def plot_matrix_on_ax(data, ax, palette='Reds'):
    """Helper to plot on a specific matplotlib axis"""
    ax.imshow(data, cmap=palette, aspect='auto')
    ax.tick_params(
        left=False,
        bottom=False,
        top=False,
        labelleft=False,
        labelbottom=False,
        labeltop=False,
    )


def main(
    macro_bacthing_factor: int = 2,
    color_scaling_power: float = 1.2,
    k_vals: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128),
    embeddings_name: str = embeddings.ALIBABA_MODEL,
    output_file: str = 'plots/1.png',
):
    assert output_file.endswith('.png')
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)

    all_surprisals = compute_profiles.load_surprisals()
    assert set(all_surprisals.keys()) == set(pythia_model.MODEL_VARIANTS)

    embd = embeddings.load_from_cache(embeddings_name)

    n_rows = len(pythia_model.MODEL_VARIANTS)
    n_cols = len(k_vals) + 1

    _, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), squeeze=False
    )

    with tqdm(total=n_rows * n_cols) as pbar:
        for row_idx, model in enumerate(pythia_model.MODEL_VARIANTS):
            pbar.set_description(f'Processing model: {model}')
            surprisals = all_surprisals[model]
            ax_mem = axes[row_idx, 0]
            mem_data = compute_profiles.compute_generalization_profile(
                surprisals=surprisals,
                embeddings=embd,
                top_k=1,
                memorization_only=True,
                macro_batching_factor=macro_bacthing_factor,
            )
            pbar.update()
            mem_data = transform_matrix_for_plotting(
                mem_data, power=color_scaling_power
            )
            plot_matrix_on_ax(mem_data, ax_mem, palette='Blues')
            ax_mem.set_ylabel(model, fontsize=12, fontweight='bold')  # type: ignore
            for col_idx, k in enumerate(k_vals):
                ax_k = axes[row_idx, col_idx + 1]
                k_data = compute_profiles.compute_generalization_profile(
                    surprisals=surprisals,
                    embeddings=embd,
                    top_k=k,
                    macro_batching_factor=macro_bacthing_factor,
                    memorization_only=False,
                )
                pbar.update()
                k_data = transform_matrix_for_plotting(
                    k_data, power=color_scaling_power
                )
                plot_matrix_on_ax(k_data, ax_k, palette='Reds')

    axes[0, 0].set_title('Memorization')  # type: ignore
    for col_idx, k in enumerate(k_vals):
        axes[0, col_idx + 1].set_title(f'k={k}')  # type: ignore

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved to {output_file}')


if __name__ == '__main__':
    Fire(main)

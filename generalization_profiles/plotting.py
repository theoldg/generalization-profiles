import matplotlib.pyplot as plt
import numpy as np

from generalization_profiles.compute_profile import Profile


def plot_profile(profile):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    def render_matrix(ax, matrix_data, title):
        data = matrix_data[1:-1, 1:-1].T[::-1]
        
        # Calculate symmetrical limits so 0 is always the center (white)
        limit = np.max(np.abs(data))
        
        # 'RdBu_r' gives Red for negative, Blue for positive
        im = ax.imshow(data, cmap='RdBu_r', vmin=-limit, vmax=limit)
        
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        
        ax.set_xticks([])
        ax.set_yticks([])

    render_matrix(ax1, -profile.values, "Profile")
    render_matrix(ax2, profile.std_error, "Standard Deviation")

    plt.tight_layout()
    plt.show()

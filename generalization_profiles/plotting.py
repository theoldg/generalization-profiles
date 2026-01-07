import matplotlib.pyplot as plt
import numpy as np

from generalization_profiles.compute_profile import Profile

# very llm generated
def plot_profile(profile: Profile, threshold=1.96):
    """
    Plots a single memorization profile, masking non-significant values.
    
    Args:
        profile: The Profile dataclass containing 'values' (ATT) and 'std' (error).
        threshold: The critical value for significance (1.96 = 95% confidence).
    """
    plt.figure(figsize=(10, 8))
    def crop_flip(x):
        return x[1:, 1:].T[::-1]
    data = crop_flip(profile.values)
    err = crop_flip(profile.std_error)
    sig_mask = np.abs(data) > (threshold * err)
    plot_data = np.where(sig_mask, -data, np.nan) 
    limit = np.nanmax(np.abs(plot_data)) if not np.all(np.isnan(plot_data)) else 0.1
    im = plt.imshow(plot_data, cmap='RdBu_r', vmin=-limit, vmax=limit)
    plt.colorbar(im, label="Memorization (Δ Log-Likelihood)")
    plt.title(f"Memorization Profile (Significant at z > {threshold})")
    plt.xlabel("Checkpoint Step")
    plt.ylabel("Treatment Step")
    plt.tight_layout()
    plt.show()

"""The whole file is AI slop."""

import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import bokeh.models
import bokeh.palettes
import bokeh.plotting

from generalization_profiles.compute_profile import Profile


def plot_profile(profile: Profile, threshold=1.96):
    """
    Plots a single profile, masking non-significant values.

    Args:
        profile: The Profile dataclass containing 'values' (ATT) and 'std_error'.
        threshold: The critical value for significance (e.g. 1.96 = 95% confidence).
    """
    plt.figure(figsize=(10, 8))

    def crop_flip(x):
        return x[1:, 1:].T[::-1]

    data = crop_flip(profile.values)
    err = crop_flip(profile.std_error)
    sig_mask = np.abs(data) > (threshold * err)
    plot_data = np.where(sig_mask, -data, np.nan)
    limit = np.nanmax(np.abs(plot_data)) if not np.all(np.isnan(plot_data)) else 0.1
    im = plt.imshow(plot_data, cmap="RdBu_r", vmin=-limit, vmax=limit)
    plt.colorbar(im, label="Memorization (Δ Log-Likelihood)")
    plt.title(f"Memorization Profile (Significant at z > {threshold})")
    plt.xlabel("Checkpoint Step")
    plt.ylabel("Treatment Step")
    plt.tight_layout()
    plt.show()


class ProfilePlot(pn.viewable.Viewer):
    def __init__(self, profile: Profile, threshold: float = 1.96):
        super().__init__()

        self.profile = profile
        self.threshold = threshold

        def crop_flip(x):
            return x[1:, 1:].T

        self._att_data = crop_flip(profile.values)
        self._err_data = crop_flip(profile.std_error)

        self.ny, self.nx = self._att_data.shape
        self.cds = bokeh.models.ColumnDataSource(data={"image": []})
        self._update_cds_data()

        limit = (
            np.nanmax(np.abs(self._att_data))
            if not np.all(np.isnan(self._att_data))
            else 0.1
        )
        self.mapper = bokeh.models.LinearColorMapper(
            palette=bokeh.palettes.RdBu[11][::-1],
            low=-limit,
            high=limit,
            nan_color="#f0f0f0",
        )

        self.fig = bokeh.plotting.figure(
            x_range=(0, self.nx),
            y_range=(0, self.ny),
            # x_axis_label="Checkpoint Step",
            # y_axis_label="Treatment Step",
            toolbar_location="above",
            sizing_mode="stretch_both",
        )

        self.fig.image(
            image="image",
            x=0,
            y=0,
            dw=self.nx,
            dh=self.ny,
            source=self.cds,
            color_mapper=self.mapper,
        )

    def _update_cds_data(self):
        """Internal logic to calculate the masked array and update the source."""
        # Significant mask: |effect| > threshold * SE
        sig_mask = np.abs(self._att_data) > (self.threshold * self._err_data)

        # Apply mask (multiply by -1 as in your original example, or keep as is)
        # We use np.nan for non-significant values so the 'nan_color' handles them
        plot_data = np.where(sig_mask, -self._att_data, np.nan)

        # Update the CDS.
        # Note: .data updates trigger the BokehJS sync automatically
        self.cds.data = {"image": [plot_data]}

    def update_threshold(self, t: float):
        """Public method to update the threshold and trigger a refresh."""
        self.threshold = t
        self._update_cds_data()

    def __panel__(self):
        return self.fig


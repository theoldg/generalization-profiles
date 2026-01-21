import pickle
from pathlib import Path
import panel as pn

from generalization_profiles import Profile
from generalization_profiles.plotting import ProfilePlot
from generalization_profiles.pythia import MODEL_VARIANTS

from generalization_profiles.aggregate_surprisals import K_VALUES

MODEL_VARIANTS.remove("2.8b")


def load_profile(path):
    path = Path(path)
    if not path.exists():
        return None
    data = pickle.loads(path.read_bytes())
    if isinstance(data, dict):
        data = Profile(**data)
    return data


def plot(prof: Profile | None):
    return ProfilePlot(prof) if prof else pn.pane.Str("File not found")


def leg_text(s, is_column_header=True):
    if is_column_header:
        # Top row: Bottom-Center alignment
        flex_styles = "align-items: flex-end; justify-content: center;"
    else:
        # Left column: Middle-Right alignment
        flex_styles = "align-items: center; justify-content: flex-end;"

    return pn.pane.HTML(
        f"""
        <div style="display: flex; {flex_styles} height: 100%; width: 100%; box-sizing: border-box; padding: 5px;">
            <h2 style="margin: 0; line-height: 1;">{s}</h2>
        </div>
        """,
        align="center",
        sizing_mode="stretch_both",
    )


def create_app():
    all_plots = {}
    for model in MODEL_VARIANTS:
        all_plots[model] = {}
        for k in K_VALUES:
            path = f"results/sentence_level_profiles/{model}_k={k}.pkl"
            prof = load_profile(path)
            all_plots[model][k] = plot(prof)

        path = f"results/profiles/mem_{model}_1.pkl"
        all_plots[model]["mem"] = plot(load_profile(path))

    threshold_slider = pn.widgets.FloatSlider(
        name="Threshold", start=0, end=5, value=1.0
    )

    def update_thresholds(event):
        for model in all_plots:
            for key in all_plots[model]:
                plot_obj = all_plots[model][key]
                if isinstance(plot_obj, ProfilePlot):
                    plot_obj.update_threshold(event.new)

    threshold_slider.param.watch(update_thresholds, "value_throttled")

    grid = pn.GridSpec(
        nrows=len(MODEL_VARIANTS) + 1,
        ncols=len(K_VALUES) + 2,
        sizing_mode="stretch_both",
        width=4000,
        height=2000,
    )

    grid[0, 1] = leg_text("Memorization", is_column_header=True)
    for ki, k in enumerate(K_VALUES):
        grid[0, 2 + ki] = leg_text(f"{k = }", is_column_header=True)

    for i, model in enumerate(MODEL_VARIANTS):
        grid[i + 1, 0] = leg_text(model, is_column_header=False)
        grid[i + 1, 1] = all_plots[model]["mem"]
        for ki, k in enumerate(K_VALUES):
            grid[i + 1, 2 + ki] = all_plots[model][k]

    layout = pn.Column(
        threshold_slider,
        grid,
        sizing_mode="fixed",
    )

    return layout


if __name__ == "__main__":
    app = create_app()
    app.show()

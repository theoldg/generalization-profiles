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
    return pickle.loads(path.read_bytes())


def create_app():
    all_plots = {}
    for model in MODEL_VARIANTS:
        all_plots[model] = {}
        for k in K_VALUES:
            path = f"results/sentence_level_profiles/{model}_k={k}.pkl"
            prof = load_profile(path)
            all_plots[model][k] = (
                ProfilePlot(prof) if prof else pn.pane.Str(f"Missing: {path}")
            )

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

    def create_grid():
        rows = []
        for model in MODEL_VARIANTS:
            model_row = [pn.pane.Markdown(f"**{model}**", width=100)]

            def format_plot(p):
                p.width = 2000
                p.height = 2000
                return p

            for k in K_VALUES:
                model_row.append(format_plot(all_plots[model][k]))
            rows.append(pn.Row(*model_row, scroll=True, width=3600))
        return pn.Column(*rows, height=2400, scroll=True)

    layout = pn.Column(
        threshold_slider,
        create_grid,
        sizing_mode="fixed",
    )

    return layout


if __name__ == "__main__":
    app = create_app()
    app.show()

import pickle
from pathlib import Path
import panel as pn
from generalization_profiles import Profile
from generalization_profiles.plotting import ProfilePlot
from generalization_profiles.pythia import MODEL_VARIANTS


MODEL_VARIANTS.remove('2.8b')

# Constants
TOP_K_LIST = [1, 2, 4, 8, 16, 32, 64]
MACRO_BATCH_LIST = [1, 2, 3, 4, 5]

def load_profile(path):
    path = Path(path)
    if not path.exists():
        return None
    data = pickle.loads(path.read_bytes())
    return Profile(
        n_macro_batches=None,
        step=data['step'],
        values=data['values'],
        std_error=data['std_error'],
    )

def create_app():
    # 1. Precompute all plots
    # Structure: plots[macro_batch][model][type_or_k]
    all_plots = {}

    for mb in MACRO_BATCH_LIST:
        all_plots[mb] = {}
        for model in MODEL_VARIANTS:
            

            all_plots[mb][model] = {}
            
            # Load Generalization plots
            for k in TOP_K_LIST:
                path = f"profiles/GEN_{model}_{mb}_{k}.pkl"
                prof = load_profile(path)
                all_plots[mb][model][k] = ProfilePlot(prof) if prof else pn.pane.Str(f"Missing: {path}")

            # Load Memorization plot
            mem_path = f"profiles/mem_{model}_{mb}.pkl"
            mem_prof = load_profile(mem_path)
            all_plots[mb][model]['mem'] = ProfilePlot(mem_prof) if mem_prof else pn.pane.Str(f"Missing: {mem_path}")

    # 2. Widgets
    mb_input = pn.widgets.IntInput(name="Macro Batching Factor", value=1, start=1, end=5)
    threshold_slider = pn.widgets.FloatSlider(name="Threshold", start=0, end=5, value=1.0)

    # 3. Update Logic
    def update_thresholds(event):
        """Update the threshold for every single precomputed plot."""
        for mb in all_plots:
            for model in all_plots[mb]:
                for key in all_plots[mb][model]:
                    plot_obj = all_plots[mb][model][key]
                    if isinstance(plot_obj, ProfilePlot):
                        plot_obj.update_threshold(event.new)

    threshold_slider.param.watch(update_thresholds, 'value_throttled')

    @pn.depends(mb_input.param.value)
    def create_grid(mb):
        """Rebuilds the visual grid when macro_batching changes."""
        rows = []

        for i, model in enumerate(MODEL_VARIANTS):
            model_row = [pn.pane.Markdown(f"**{model}**", width=100)]
            
            # Helper to apply sizing to plots
            def format_plot(p):
                if not isinstance(p, pn.pane.Str):
                    p.width = 2000   # Set your preferred fixed width
                    p.height = 2000  # Set your preferred fixed height
                return p

            # Add MEM plot
            model_row.append(format_plot(all_plots[mb][model]['mem']))

            # Add GEN plots
            for k in TOP_K_LIST:
                model_row.append(format_plot(all_plots[mb][model][k]))

            # We wrap the row in a Row container with horizontal scrolling enabled
            rows.append(pn.Row(*model_row, scroll=True, width=3600)) # width controls the "viewport"
        
        # We wrap the column in a Column container with vertical scrolling enabled
        return pn.Column(*rows, height=2400, scroll=True)

    # 4. Layout
    layout = pn.Column(
        pn.Row(mb_input, threshold_slider),
        create_grid,
        sizing_mode='fixed',
    )
    
    return layout

if __name__ == "__main__":
    app = create_app()
    app.show()
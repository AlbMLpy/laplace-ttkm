from typing import Optional

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

PARAMS = {
    'figure.figsize': (10, 5),
    'figure.constrained_layout.use': True,
    'figure.facecolor': 'white',
    'font.size': 10,
    'axes.labelsize': 16,
    'axes.grid': True,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'lines.linestyle': '-',
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titlesize': 18,
    'figure.max_open_warning': 50,
}

COLORS = dict(
    red='#e60000',
    green='#268c26', 
    blue='#005ce6', 
    orange='#ff7f0e', 
    purple='#9467bd',
    black="#161414",
)

model2color = dict(VI=COLORS['blue'], CV=COLORS['green'], GP=COLORS['red'])
model2name = dict(VI='LA-TTKM (VI)', CV='LA-TTKM (CV)', GP='Full GP')

def plot_results_multiple(
    results,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    dpi: int = 500,
):
    pmean_linewidth = 1.2
    pstd_alpha = 0.12
    mean_color, mean_linewidth, mean_label = COLORS['black'], 1.3, 'Test Data'
    mean_linestyle = 'dashed'
    x_label, y_label, hm_pstd = 'Test point', 'Motor torque [Nm]', 1
    hm_points = 800

    with mpl.rc_context(PARAMS):
        rows, cols = 1, 1 
        if figsize is None: figsize = (cols * 5, rows * 4)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False, sharey=False)
        ax = np.ravel(axes)[0]
        for i, (model_name, model_results) in enumerate(results.items()):
            pmean = np.array(model_results['y_mean_test'])[:hm_points]  
            pstd = np.array(model_results['y_std_test'])[:hm_points]
            y_test = np.array(model_results['y_test'])[:hm_points]
            x_test = list(range(len(y_test)))
            if i == 0:
                ax.plot(
                    x_test, y_test, color=mean_color, 
                    linewidth=mean_linewidth, label=mean_label,
                    linestyle=mean_linestyle, 
                )
            ax.plot(
                x_test, pmean, color=model2color[model_name], 
                linewidth=pmean_linewidth, label=model2name[model_name],
            )
            ax.fill_between(
                x_test, (pmean - hm_pstd*pstd), (pmean + hm_pstd*pstd), 
                color=model2color[model_name], alpha=pstd_alpha, 
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        legend = fig.legend(
            loc='upper right',
            bbox_to_anchor=(0.9, 1.1),
            ncol=1,
            frameon=True,
        )
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.0)
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend
        if save_path: 
            plt.savefig(save_path, bbox_inches="tight", bbox_extra_artists=[legend], dpi=dpi)
        if show_plot:
            plt.show()
        else:
            plt.close()

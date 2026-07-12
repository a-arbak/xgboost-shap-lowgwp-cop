import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')
FIG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'figures')

FLUID_ORDER = [
    'R134a', 'R513A', 'R450A', 'Tern1',
    'R515B', 'R1234yf',
    'R410A', 'R32', 'R454B', 'R452B',
]
FLUID_LABEL = {
    'R134a': 'R-134a', 'R513A': 'R-513A', 'R450A': 'R-450A', 'Tern1': 'Tern-1',
    'R515B': 'R-515B', 'R1234yf': 'R-1234yf',
    'R410A': 'R-410A', 'R32': 'R-32', 'R454B': 'R-454B', 'R452B': 'R-452B',
}
# Colourblind-safe palette (Paul Tol "muted" + Tol vibrant orange), identical
# fluid-colour mapping to plot_parity_testonly.py so Figs. 3 and 5 stay
# visually consistent.
PALETTE = [
    '#EE7733',  # R134a  - orange (Tol vibrant)
    '#332288',  # R513A  - indigo
    '#117733',  # R450A  - green
    '#882255',  # Tern1  - wine
    '#AA4499',  # R515B  - purple
    '#44AA99',  # R1234yf- teal
    '#CC6677',  # R410A  - rose
    '#88CCEE',  # R32    - cyan
    '#999933',  # R454B  - olive
    '#DDCC77',  # R452B  - sand
]


def run():
    X_df = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    y_df = pd.read_csv(os.path.join(DATA_DIR, 'y_labels.csv'))

    cop = y_df.iloc[:, 0].values
    fluids = X_df['fluid_name'].values

    bins = np.arange(
        np.floor(cop.min() * 2) / 2,
        np.ceil(cop.max() * 2) / 2 + 0.5,
        0.5
    )
    centres = (bins[:-1] + bins[1:]) / 2
    n_bins = len(centres)

    # Build per-fluid counts
    counts_by_fluid = {}
    for fl in FLUID_ORDER:
        if fl in fluids:
            c, _ = np.histogram(cop[fluids == fl], bins=bins)
            counts_by_fluid[fl] = c
        else:
            counts_by_fluid[fl] = np.zeros(n_bins, dtype=int)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    bottoms = np.zeros(n_bins)
    for fl, color in zip(FLUID_ORDER, PALETTE):
        c = counts_by_fluid[fl]
        ax.bar(centres, c, bottom=bottoms, width=0.46,
               color=color, edgecolor='white', linewidth=0.6,
               label=FLUID_LABEL.get(fl, fl), alpha=0.92)
        bottoms += c

    # White underlay keeps the statistics lines visible over the black
    # R-134a bars
    ax.axvline(np.mean(cop), color='white', linewidth=3.2, zorder=4)
    ax.axvline(np.mean(cop), color='#333333', linewidth=1.8,
               linestyle='--', zorder=5, label=f'Mean = {np.mean(cop):.2f}')
    ax.axvline(np.median(cop), color='white', linewidth=3.2, zorder=4)
    ax.axvline(np.median(cop), color='#D65F5F', linewidth=1.8,
               linestyle=':', zorder=5, label=f'Median = {np.median(cop):.2f}')

    ax.set_xlabel(r'COP$_\mathrm{c}$  [—]', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    # Integer ticks instead of bin centres — IJR rule: simple scale numbers
    ax.set_xticks(np.arange(int(np.floor(bins[0])), int(np.ceil(bins[-1])) + 1))
    ax.tick_params(labelsize=10)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, ncol=2,
              loc='upper right', framealpha=1.0,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.8)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'cop_stacked.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    run()

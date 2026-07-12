import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')
FIG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'figures')

ALL_FEATURES = [
    'Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix', 'n_components', 'epsilon_LLSL',
    'T_liq_evap_in', 'T_liq_evap_out', 'superheat', 'DELTAT_dew_suction',
    'T_liq_cond_in', 'T_liq_cond_out', 'subcool', 'DELTAT_dew_discharge',
]
MODEL_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, random_state=42)

FLUID_LABEL = {
    'R134a': 'R-134a', 'R1234yf': 'R-1234yf', 'R32': 'R-32',
    'R410A': 'R-410A', 'R450A': 'R-450A', 'R452B': 'R-452B',
    'R454B': 'R-454B', 'R513A': 'R-513A', 'R515B': 'R-515B', 'Tern1': 'Tern-1',
}
FLUID_ORDER = ['R134a', 'R513A', 'R450A', 'Tern1',
               'R515B', 'R1234yf', 'R410A', 'R32', 'R454B', 'R452B']
# Colourblind-safe qualitative palette (Paul Tol "muted" scheme, with the
# tenth class taken from Tol's "vibrant" orange), paired with a distinct
# marker per fluid as redundant encoding, so the plot remains readable in
# grayscale print and for colourblind readers (Reviewer #4, JIJR-D-26-00648).
PALETTE = ['#EE7733',  # R134a  - orange (Tol vibrant)
           '#332288',  # R513A  - indigo
           '#117733',  # R450A  - green
           '#882255',  # Tern1  - wine
           '#AA4499',  # R515B  - purple
           '#44AA99',  # R1234yf- teal
           '#CC6677',  # R410A  - rose
           '#88CCEE',  # R32    - cyan
           '#999933',  # R454B  - olive
           '#DDCC77']  # R452B  - sand
COLOR_MAP = dict(zip(FLUID_ORDER, PALETTE))

MARKER_MAP = {
    'R134a': 'o', 'R513A': 's', 'R450A': '^', 'Tern1': 'D', 'R515B': 'v',
    'R1234yf': 'P', 'R410A': 'X', 'R32': '*', 'R454B': '<', 'R452B': '>',
}
# '*' and 'D' render optically smaller/larger than circles at equal s
MARKER_SIZE = {'*': 110, 'D': 45, 'P': 70, 'X': 70}

# Expanded measurement uncertainty for COP_c (k=2, from NIST TN 2233 data)
COP_UNC_REL = {
    'R134a': 0.006, 'R513A': 0.006, 'R450A': 0.006, 'Tern1': 0.006,
    'R515B': 0.006, 'R1234yf': 0.006,
    'R410A': 0.005, 'R32': 0.005, 'R454B': 0.005, 'R452B': 0.005,
}


def make_split(X_df, test_frac=0.20, random_state=42):
    test_idx = (
        X_df.groupby('fluid_name', group_keys=False)
        .apply(lambda g: g.sample(frac=test_frac, random_state=random_state),
               include_groups=False)
        .index
    )
    train_idx = X_df.index.difference(test_idx)
    return train_idx, test_idx


def run():
    X_df = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    y_df = pd.read_csv(os.path.join(DATA_DIR, 'y_labels.csv'))

    y = y_df.iloc[:, 0].values
    fluid_names = X_df['fluid_name'].values

    train_idx, test_idx = make_split(X_df)

    X_train = X_df.loc[train_idx, ALL_FEATURES].values
    y_train = y[train_idx]
    X_test  = X_df.loc[test_idx, ALL_FEATURES].values
    y_test  = y[test_idx]
    fluid_test = fluid_names[test_idx]

    print(f"Train: {len(y_train)}  Test: {len(y_test)}")

    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f"Test R² = {r2:.4f}  (expected ~0.9793)")

    # Normalized split-conformal 90% PI (see Scripts/conformal_pi.py)
    pi_lo = np.load(os.path.join(DATA_DIR, 'conformal_pi_lo_test.npy'))
    pi_hi = np.load(os.path.join(DATA_DIR, 'conformal_pi_hi_test.npy'))
    if len(pi_lo) != len(y_test):
        raise RuntimeError(
            f"Conformal PI array length {len(pi_lo)} != test set length "
            f"{len(y_test)} — re-run Scripts/conformal_pi.py first.")

    coverage = np.mean((y_test >= pi_lo) & (y_test <= pi_hi))
    width    = np.mean(pi_hi - pi_lo)
    print(f"PI coverage = {coverage:.3f}  avg width = {width:.4f}")

    # Per-fluid coverage
    print("Per-fluid PI coverage:")
    for fl in FLUID_ORDER:
        mask = fluid_test == fl
        if mask.sum() == 0:
            continue
        cov = np.mean((y_test[mask] >= pi_lo[mask]) & (y_test[mask] <= pi_hi[mask]))
        print(f"  {FLUID_LABEL.get(fl, fl):12s}  n={mask.sum():2d}  coverage={cov:.2f}"
              f"  avg_width={np.mean(pi_hi[mask]-pi_lo[mask]):.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(4.7, 4.7))
    lims = [2.5, 8.5]
    x_range = np.linspace(lims[0], lims[1], 200)

    ax.plot(lims, lims, 'k-', linewidth=1.2, alpha=0.65, zorder=1)
    ax.fill_between(x_range, x_range * 0.95, x_range * 1.05,
                    alpha=0.07, color='grey', zorder=1)
    ax.plot(x_range, x_range * 1.05, '--', color='grey', lw=0.85, alpha=0.5, zorder=1)
    ax.plot(x_range, x_range * 0.95, '--', color='grey', lw=0.85, alpha=0.5, zorder=1)
    ax.text(7.2, 7.2 * 1.05 + 0.08, '+5%', fontsize=8, color='grey', ha='center')
    ax.text(7.2, 7.2 * 0.95 - 0.12, '−5%', fontsize=8, color='grey', ha='center')

    for fl in FLUID_ORDER:
        mask = fluid_test == fl
        if mask.sum() == 0:
            continue
        color = COLOR_MAP.get(fl, '#333333')
        # Vertical PI bars
        yerr_lo = np.clip(y_pred[mask] - pi_lo[mask], 0, None)
        yerr_hi = np.clip(pi_hi[mask] - y_pred[mask], 0, None)
        ax.errorbar(y_test[mask], y_pred[mask],
                    yerr=[yerr_lo, yerr_hi], fmt='none',
                    ecolor=color, elinewidth=0.8, capsize=2.0,
                    capthick=0.8, alpha=0.55, zorder=2)
        # Horizontal experimental uncertainty bars
        unc_rel = COP_UNC_REL.get(fl, 0.006)
        xerr = unc_rel * y_test[mask]
        ax.errorbar(y_test[mask], y_pred[mask],
                    xerr=xerr, fmt='none',
                    ecolor=color, elinewidth=0.8, capsize=2.0,
                    capthick=0.8, alpha=0.55, zorder=2)
        marker = MARKER_MAP.get(fl, 'o')
        ax.scatter(y_test[mask], y_pred[mask], color=color,
                   s=MARKER_SIZE.get(marker, 55), marker=marker, alpha=0.92,
                   edgecolors='white', linewidth=0.5, zorder=3,
                   label=FLUID_LABEL.get(fl, fl))

    # Reference experimental uncertainty bar — axes coordinates to avoid overlap
    ref_xerr = 0.006 * 5.0   # representative at COP=5
    # Convert xerr to axes fraction for display
    ax_span = lims[1] - lims[0]
    ref_x_ax, ref_y_ax = 0.50, 0.07   # lower-centre of axes
    ref_x_data = lims[0] + ref_x_ax * ax_span
    ref_y_data = lims[0] + ref_y_ax * ax_span
    ax.errorbar([ref_x_data], [ref_y_data], xerr=[ref_xerr], fmt='none',
                ecolor='#333333', elinewidth=1.2, capsize=4, capthick=1.2, zorder=5)
    ax.text(ref_x_data, ref_y_data + 0.20, r'Exp. unc. ($\leq$0.6%)',
            fontsize=7, ha='center', va='bottom', color='#333333')

    ax.text(0.05, 0.95,
            f'$R^2 = {r2:.3f}$\n'
            f'PI coverage = {coverage:.1%}\n'
            f'$N_{{\\mathrm{{test}}}} = {len(y_test)}$',
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.85, edgecolor='#CCCCCC'))

    ax.set_xlabel(r'COP$_\mathrm{exp}$  [—]', fontsize=11)
    ax.set_ylabel(r'COP$_\mathrm{pred}$  [—]', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.legend(fontsize=7.5, ncol=2, loc='lower right', framealpha=1.0,
              handletextpad=0.4, columnspacing=0.7, handlelength=1.0)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'parity_testonly.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    run()

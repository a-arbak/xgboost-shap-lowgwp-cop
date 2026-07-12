import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from xgboost import XGBRegressor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')
FIG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'figures')

ALL_FEATURES = [
    'Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix', 'n_components',
    'epsilon_LLSL',
    'T_liq_evap_in', 'T_liq_evap_out', 'superheat', 'DELTAT_dew_suction',
    'T_liq_cond_in', 'T_liq_cond_out', 'subcool', 'DELTAT_dew_discharge',
]

LABEL_MAP = {
    'Tc_mix':               r'$T_\mathrm{c,mix}$',
    'Pc_mix':               r'$P_\mathrm{c,mix}$',
    'omega_mix':            r'$\omega_\mathrm{mix}$',
    'M_mix':                r'$M_\mathrm{mix}$',
    'n_components':         r'$N_\mathrm{comp}$',
    'epsilon_LLSL':         r'$\varepsilon_\mathrm{LLSL}$',
    'T_liq_evap_in':        r'$T_\mathrm{liq,evap,in}$',
    'T_liq_evap_out':       r'$T_\mathrm{liq,evap,out}$',
    'superheat':            r'$\Delta T_\mathrm{SH}$',
    'DELTAT_dew_suction':   r'$\Delta T_\mathrm{dew,suc}$',
    'T_liq_cond_in':        r'$T_\mathrm{liq,cond,in}$',
    'T_liq_cond_out':       r'$T_\mathrm{liq,cond,out}$',
    'subcool':              r'$\Delta T_\mathrm{SC}$',
    'DELTAT_dew_discharge': r'$\Delta T_\mathrm{dew,dis}$',
}

# Fluid display order: group by refrigerant family
FLUID_ORDER = [
    'R-134a', 'R-513A', 'R-450A', 'Tern1',   # R-134a based
    'R-1234yf', 'R-515B',                       # HFO pure/near-pure
    'R-32', 'R-410A', 'R-454B', 'R-452B',      # R-32 based
]

FLUID_LABEL = {
    'R134a':   'R-134a',  'R513A':   'R-513A',  'R450A':  'R-450A',
    'Tern1':   'Tern-1',  'R1234yf': 'R-1234yf','R515B':  'R-515B',
    'R32':     'R-32',    'R410A':   'R-410A',  'R454B':  'R-454B',
    'R452B':   'R-452B',
}

# Separator positions (cumulative sample counts between groups)
GROUP_SEPARATORS = [4]  # after 4th fluid column


def run():
    model = XGBRegressor()
    model.load_model(os.path.join(DATA_DIR, 'xgboost_model.json'))

    X_full = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    fluid_names = X_full['fluid_name'].values
    X = X_full[ALL_FEATURES].astype(float)

    print("Computing SHAP values...")
    explainer  = shap.Explainer(model.predict, X)
    shap_vals  = explainer(X).values          # (N, 14)

    # Build DataFrame: rows = features, columns = observations sorted by fluid
    shap_df = pd.DataFrame(shap_vals, columns=ALL_FEATURES)
    shap_df['fluid'] = fluid_names

    # Map fluid codes to display names
    shap_df['fluid'] = shap_df['fluid'].map(
        lambda f: FLUID_LABEL.get(f, f)
    )

    # Sort by predefined fluid order
    fluid_cat = pd.CategoricalDtype(
        categories=[FLUID_LABEL[k] for k in
                    ['R134a','R513A','R450A','Tern1',
                     'R1234yf','R515B','R32','R410A','R454B','R452B']],
        ordered=True
    )
    shap_df['fluid'] = shap_df['fluid'].astype(fluid_cat)
    shap_df = shap_df.sort_values('fluid').reset_index(drop=True)

    fluid_sorted = shap_df['fluid'].values
    matrix = shap_df[ALL_FEATURES].values.T   # (14 features, N obs)

    # Symmetric colour limit
    vlim = np.percentile(np.abs(matrix), 98)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(
        matrix, aspect='auto',
        cmap='RdBu_r', vmin=-vlim, vmax=vlim,
        interpolation='nearest',
    )

    # y-axis: feature labels
    ax.set_yticks(range(len(ALL_FEATURES)))
    ax.set_yticklabels(
        [LABEL_MAP[f] for f in ALL_FEATURES], fontsize=9
    )

    # x-axis: fluid group labels centred on each group — preserve order
    seen = {}
    ordered_fluids, ordered_counts = [], []
    for f in fluid_sorted:
        if f not in seen:
            seen[f] = (fluid_sorted == f).sum()
            ordered_fluids.append(f)
            ordered_counts.append(seen[f])

    cumulative = np.cumsum([0] + ordered_counts)
    centres    = [(cumulative[i] + cumulative[i+1]) / 2
                  for i in range(len(ordered_fluids))]

    ax.set_xticks(centres)
    ax.set_xticklabels(ordered_fluids, fontsize=9, rotation=30, ha='right')

    # Vertical separators between fluids
    for i in range(1, len(ordered_fluids)):
        ax.axvline(x=cumulative[i] - 0.5, color='white', linewidth=0.8,
                   linestyle='-')

    # Family separators (thicker)
    family_boundary  = cumulative[4] - 0.5
    family_boundary2 = cumulative[6] - 0.5
    ax.axvline(x=family_boundary,  color='black', linewidth=1.6, linestyle='--')
    ax.axvline(x=family_boundary2, color='black', linewidth=1.6, linestyle='--')

    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label('SHAP value', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=9)

    # Family labels below the fluid tick labels
    fig.subplots_adjust(bottom=0.30)
    y_fam = -0.28

    def fam_label(x_left, x_right, label):
        xc = (x_left + x_right) / 2 / len(fluid_sorted)
        ax.annotate(label, xy=(xc, y_fam),
                    xycoords=('axes fraction', 'axes fraction'),
                    ha='center', va='top', fontsize=9, style='italic')

    fam_label(0,             cumulative[4],  'R-134a based')
    fam_label(cumulative[4], cumulative[6],  'HFO')
    fam_label(cumulative[6], cumulative[10], 'R-32 based')

    out_path = os.path.join(FIG_DIR, 'shap_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    run()

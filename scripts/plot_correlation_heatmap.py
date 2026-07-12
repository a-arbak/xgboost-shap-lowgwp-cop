import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'figures')

LABEL_MAP = {
    'COP_c_ref_dh':        r'$\mathrm{COP}_c$',
    'Tc_mix':              r'$T_\mathrm{c,mix}$',
    'Pc_mix':              r'$P_\mathrm{c,mix}$',
    'omega_mix':           r'$\omega_\mathrm{mix}$',
    'M_mix':               r'$M_\mathrm{mix}$',
    'GWP_mix':             r'$\mathrm{GWP}_\mathrm{mix}$',
    'n_components':        r'$N_\mathrm{comp}$',
    'epsilon_LLSL':        r'$\varepsilon_\mathrm{LLSL}$',
    'T_liq_evap_in':       r'$T_\mathrm{liq,evap,in}$',
    'T_liq_evap_out':      r'$T_\mathrm{liq,evap,out}$',
    'superheat':           r'$\Delta T_\mathrm{SH}$',
    'T_liq_cond_in':       r'$T_\mathrm{liq,cond,in}$',
    'T_liq_cond_out':      r'$T_\mathrm{liq,cond,out}$',
    'subcool':             r'$\Delta T_\mathrm{SC}$',
    'DELTAT_dew_suction':  r'$\Delta T_\mathrm{dew,suc}$',
    'DELTAT_dew_discharge':r'$\Delta T_\mathrm{dew,dis}$',
}

# Leakage-free feature set (matches ml_model.py ALL_FEATURES)
SELECTED = [
    'COP_c_ref_dh',
    'Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix', 'GWP_mix', 'n_components',
    'epsilon_LLSL', 'T_liq_evap_in', 'T_liq_evap_out', 'superheat',
    'DELTAT_dew_suction', 'T_liq_cond_in', 'T_liq_cond_out', 'subcool',
    'DELTAT_dew_discharge',
]

def plot_correlation_heatmap():
    # Read physics features from X_processed.csv; COP from y_labels.csv
    X = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    y = pd.read_csv(os.path.join(DATA_DIR, 'y_labels.csv'))

    X = X.drop(columns=[c for c in ['fluid_name'] if c in X.columns])
    df = pd.concat([X, y], axis=1)

    selected = [c for c in SELECTED if c in df.columns]
    corr_df = df[selected].apply(pd.to_numeric, errors='coerce').dropna()
    corr = corr_df.corr()

    # Rename columns/index with proper mathematical notation
    corr.columns = [LABEL_MAP.get(c, c) for c in corr.columns]
    corr.index   = [LABEL_MAP.get(c, c) for c in corr.index]

    # A4 text width: 210mm - 25mm - 25mm = 160mm = 6.30 in
    fig, ax = plt.subplots(figsize=(6.30, 5.80))
    fig.patch.set_facecolor('#FFFFFF')

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show lower triangle only
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap='RdBu_r', vmin=-1, vmax=1, center=0,
        annot=True, fmt='.2f', annot_kws={'size': 7},
        linewidths=0.5, linecolor='#EEEEEE',
        cbar_kws={'label': 'Pearson Correlation Coefficient', 'shrink': 0.8}
    )
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    ax.figure.axes[-1].tick_params(labelsize=9)
    ax.figure.axes[-1].set_ylabel('Pearson Correlation Coefficient', fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig2_correlation_heatmap.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved: {out}')

if __name__ == '__main__':
    plot_correlation_heatmap()

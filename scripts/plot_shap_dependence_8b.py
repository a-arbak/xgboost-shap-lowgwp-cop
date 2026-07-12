import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from xgboost import XGBRegressor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')
FIG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'figures')

LABEL_MAP = {
    'Tc_mix':                r'$T_\mathrm{c,mix}$  [K]',
    'Pc_mix':                r'$P_\mathrm{c,mix}$  [MPa]',
    'omega_mix':             r'$\omega_\mathrm{mix}$  [—]',
    'M_mix':                 r'$M_\mathrm{mix}$  [g mol$^{-1}$]',
    'GWP_mix':               r'$\mathrm{GWP}_\mathrm{mix}$  [—]',
    'n_components':          r'$N_\mathrm{comp}$  [—]',
    'T_liq_cond_out':        r'$T_\mathrm{liq,cond,out}$  [°C]',
    'T_liq_cond_in':         r'$T_\mathrm{liq,cond,in}$  [°C]',
    'T_liq_evap_out':        r'$T_\mathrm{liq,evap,out}$  [°C]',
    'T_liq_evap_in':         r'$T_\mathrm{liq,evap,in}$  [°C]',
    'superheat':             r'$\Delta T_\mathrm{SH}$  [K]',
    'subcool':               r'$\Delta T_\mathrm{SC}$  [K]',
    'epsilon_LLSL':          r'$\varepsilon_\mathrm{LLSL}$  [—]',
    'DELTAT_dew_suction':    r'$\Delta T_\mathrm{dew,suc}$  [K]',
    'DELTAT_dew_discharge':  r'$\Delta T_\mathrm{dew,dis}$  [K]',
    'COP_c_ref_dh':          r'$\mathrm{COP}_c$  [—]',
}

SHAP_YLIM   = (-1.2, 1.2)
SHAP_YTICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]

def set_shap_yaxis(ax):
    ax.set_ylim(SHAP_YLIM)
    ax.set_yticks(SHAP_YTICKS)

def fix_labels(fig, fontsize=9):
    """Rename code-style axis/colorbar labels using LABEL_MAP."""
    axes = fig.get_axes()
    for ax in axes:
        xl = ax.get_xlabel()
        if xl in LABEL_MAP:
            ax.set_xlabel(LABEL_MAP[xl], fontsize=fontsize)
        yl = ax.get_ylabel()
        if yl in LABEL_MAP:
            ax.set_ylabel(LABEL_MAP[yl], fontsize=fontsize)

def run():
    # Load model and features
    model = XGBRegressor()
    model.load_model(os.path.join(DATA_DIR, 'xgboost_model.json'))
    X_full = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    X = X_full.drop(columns=[c for c in ['fluid_name'] if c in X_full.columns])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)
    
    # Use full dataset
    X_sub = X
    
    # --- Compute SHAP values ---
    print("Computing SHAP values...")
    explainer = shap.Explainer(model.predict, X_sub)
    shap_results = explainer(X_sub)
    shap_vals = shap_results.values

    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'text.color': 'black', 'axes.labelcolor': 'black',
                         'xtick.color': 'black', 'ytick.color': 'black'})

    # --- Figure 7a: T_liq_cond_out ---
    feat_a = 'T_liq_cond_out'
    inter_a = 'auto'

    print(f"Generating Fig 7a for {feat_a} (interaction: {inter_a})...")
    shap.dependence_plot(feat_a, shap_vals, X_sub, interaction_index=inter_a, show=False)
    plt.gcf().set_size_inches(4.20, 3.15)
    fix_labels(plt.gcf(), fontsize=12)
    plt.ylabel('SHAP value  [—]', fontsize=12)
    for ax in plt.gcf().get_axes():
        ax.tick_params(labelsize=11, colors='black')
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('black')
        if ax.get_ylabel() == 'SHAP value  [—]':
            set_shap_yaxis(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_dep_cond.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shap_dep_cond.png")

    # --- Figure 7b: T_liq_evap_out (2nd-ranked SHAP driver; Q_dot_evap_ref removed as leakage) ---
    feat_b = 'T_liq_evap_out'
    inter_b = 'auto'

    print(f"Generating Fig 7b for {feat_b} (interaction: {inter_b})...")
    shap.dependence_plot(feat_b, shap_vals, X_sub, interaction_index=inter_b, show=False)
    plt.gcf().set_size_inches(4.20, 3.15)
    fix_labels(plt.gcf(), fontsize=12)
    plt.ylabel('SHAP value  [—]', fontsize=12)
    for ax in plt.gcf().get_axes():
        ax.tick_params(labelsize=11, colors='black')
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('black')
        if ax.get_ylabel() == 'SHAP value  [—]':
            set_shap_yaxis(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_dep_evap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved shap_dep_evap.png")

    # --- Panel (c): T_c,mix coloured by refrigerant family ---
    FAMILY_MAP = {
        'R134a': 'R-134a based', 'R513A': 'R-134a based',
        'R450A': 'R-134a based', 'Tern1':  'R-134a based',
        'R1234yf': 'HFO',        'R515B':  'HFO',
        'R32': 'R-32 based',     'R410A':  'R-32 based',
        'R454B': 'R-32 based',   'R452B':  'R-32 based',
    }
    FAMILY_COLORS = {
        'R-134a based': '#1f77b4',
        'HFO':          '#2ca02c',
        'R-32 based':   '#d62728',
    }
    X_full_c = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    fluid_names = X_full_c['fluid_name'].values
    families = [FAMILY_MAP.get(f, 'Unknown') for f in fluid_names]

    feat_c = 'Tc_mix'
    feat_idx_c = list(X_sub.columns).index(feat_c)
    x_c = X_sub[feat_c].values
    y_c = shap_vals[:, feat_idx_c]

    fig_c, ax_c = plt.subplots(figsize=(4.20, 3.15))
    for fam, fc in FAMILY_COLORS.items():
        mask = np.array(families) == fam
        ax_c.scatter(x_c[mask], y_c[mask], c=fc, s=20, alpha=0.78,
                     edgecolors='none', label=fam)
    ax_c.axhline(0, color='grey', linewidth=0.7, linestyle='--')
    ax_c.set_xlabel(LABEL_MAP['Tc_mix'], fontsize=12)
    ax_c.set_ylabel('SHAP value  [—]', fontsize=12)
    ax_c.tick_params(labelsize=11)
    set_shap_yaxis(ax_c)
    ax_c.legend(fontsize=9.5, framealpha=1.0)
    fig_c.tight_layout()
    fig_c.savefig(os.path.join(FIG_DIR, 'shap_dep_tcmix.png'),
                  dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_c)
    print("Saved shap_dep_tcmix.png")

    # --- Panel (d): epsilon_LLSL coloured by Delta T_SH ---
    print("Generating panel (d): epsilon_LLSL vs SHAP, coloured by superheat...")
    shap.dependence_plot('epsilon_LLSL', shap_vals, X_sub,
                         interaction_index='superheat', show=False)
    fig_d = plt.gcf()
    fig_d.set_size_inches(4.20, 3.15)
    fix_labels(fig_d, fontsize=12)
    plt.ylabel('SHAP value  [—]', fontsize=12)
    for ax in fig_d.get_axes():
        ax.tick_params(labelsize=11, colors='black')
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('black')
        if ax.get_ylabel() == 'SHAP value  [—]':
            set_shap_yaxis(ax)
    plt.tight_layout()
    fig_d.savefig(os.path.join(FIG_DIR, 'shap_dep_llsl.png'),
                  dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_d)
    print("Saved shap_dep_llsl.png")

if __name__ == '__main__':
    run()

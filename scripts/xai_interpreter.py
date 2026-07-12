import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# Absolute paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'figures')

LABEL_MAP = {
    # Physics-based refrigerant descriptors (new)
    'Tc_mix':               r'$T_\mathrm{c,mix}$',
    'Pc_mix':               r'$P_\mathrm{c,mix}$',
    'omega_mix':            r'$\omega_\mathrm{mix}$',
    'M_mix':                r'$M_\mathrm{mix}$',
    'GWP_mix':              r'$\mathrm{GWP}_\mathrm{mix}$',
    'n_components':         r'$N_\mathrm{comp}$',
    # Operational conditions
    'T_liq_cond_out':       r'$T_\mathrm{liq,cond,out}$',
    'T_liq_cond_in':        r'$T_\mathrm{liq,cond,in}$',
    'T_liq_evap_out':       r'$T_\mathrm{liq,evap,out}$',
    'T_liq_evap_in':        r'$T_\mathrm{liq,evap,in}$',
    'superheat':            r'$\Delta T_\mathrm{SH}$',
    'subcool':              r'$\Delta T_\mathrm{SC}$',
    'epsilon_LLSL':         r'$\varepsilon_\mathrm{LLSL}$',
    'DELTAT_dew_suction':   r'$\Delta T_\mathrm{dew,suc}$',
    'DELTAT_dew_discharge': r'$\Delta T_\mathrm{dew,dis}$',
}

def run_xai():
    np.random.seed(42)  # reproducible PermutationExplainer sampling
    # Load Model and Data
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.load_model(os.path.join(DATA_DIR, 'xgboost_model.json'))
    X = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))

    # Drop non-numeric identifier column saved for LOOCV
    if 'fluid_name' in X.columns:
        X = X.drop(columns=['fluid_name'])

    # Force everything to float for SHAP stability
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)

    X_subset = X  # Use full dataset (248 points)
    
    print("Calculating SHAP values using model-agnostic Explainer...")
    
    feature_cols = X.columns.tolist()

    def model_predict_raw(data):
        df = pd.DataFrame(data, columns=feature_cols)
        return model.predict(df)

    # Using the dataset itself as the background
    explainer = shap.Explainer(model_predict_raw, X_subset)
    shap_results = explainer(X_subset)
    
    # 1. SHAP Summary Plot (Fig 6)
    plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'text.color': 'black', 'axes.labelcolor': 'black',
                         'xtick.color': 'black', 'ytick.color': 'black'})
    shap.summary_plot(shap_results, X_subset, max_display=20, show=False, plot_size=(3.70, 3.20))
    plt.xlabel('SHAP Value (Impact on COP)', fontsize=7)
    # Force all text elements to 7pt (SHAP overrides rcParams internally)
    plt.gcf().canvas.draw()  # populate tick labels before reading
    for ax in plt.gcf().get_axes():
        ax.tick_params(labelsize=7, colors='black')
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontsize(7)
            text.set_color('black')
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(7)
            item.set_color('black')
        # Rename y-tick labels to proper mathematical notation
        new_labels = [LABEL_MAP.get(t.get_text(), t.get_text())
                      for t in ax.get_yticklabels()]
        if any(t.get_text() in LABEL_MAP for t in ax.get_yticklabels()):
            ax.set_yticklabels(new_labels, fontsize=7, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_summary.png'), dpi=300)
    print("Saved fig6_shap_summary.png")
    
    # 2. SHAP Bar Plot (Fig 7)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_results, X_subset, plot_type="bar", max_display=20, show=False)
    plt.title('Feature Importance Ranking (Top 20 Drivers)', fontsize=16)
    plt.xlabel('Mean |SHAP Value| (Average Impact on COP)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig7_shap_bar.png'), dpi=300)
    print("Saved fig7_shap_bar.png")
    
    # --- Print top 7 features for Table 4 ---
    feature_importance = np.abs(shap_results.values).mean(0)
    top_indices = np.argsort(feature_importance)[::-1][:7]
    print("\n--- Top 7 Features for Table 4 ---")
    for rank, idx in enumerate(top_indices, 1):
        name = X_subset.columns[idx]
        imp = feature_importance[idx]
        print(f"Rank {rank}: {name} = {imp:.3f}")
    
    print("Dependence plots are handled by plot_shap_dependence.py")

if __name__ == "__main__":
    run_xai()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_validate, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')
FIG_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'figures')
REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, 'data', 'model_stability_report.txt')

PHYSICS_FEATURES = ['Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix', 'n_components']
OPERATIONAL_FEATURES = [
    'epsilon_LLSL',
    'T_liq_evap_in', 'T_liq_evap_out', 'superheat', 'DELTAT_dew_suction',
    'T_liq_cond_in',  'T_liq_cond_out', 'subcool',   'DELTAT_dew_discharge',
]
ALL_FEATURES = PHYSICS_FEATURES + OPERATIONAL_FEATURES

MODEL_PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)


def run_verification():
    X_df  = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    y_df  = pd.read_csv(os.path.join(DATA_DIR, 'y_labels.csv'))

    fluid_names = X_df['fluid_name'].values
    X = X_df[ALL_FEATURES].values
    y = y_df.iloc[:, 0].values

    print(f"Loaded {len(y)} samples | {len(set(fluid_names))} refrigerants")

    # ── 1. 5-Fold Cross-Validation ─────────────────────────────────────────────
    print("\nRunning 5-Fold Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_validate(
        XGBRegressor(**MODEL_PARAMS), X, y, cv=kf,
        scoring=['r2', 'neg_root_mean_squared_error'],
    )
    cv_r2   = cv['test_r2']
    cv_rmse = -cv['test_neg_root_mean_squared_error']

    print(f"  R² per fold  : {np.round(cv_r2, 4).tolist()}")
    print(f"  5-Fold CV R²   : {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
    print(f"  5-Fold CV RMSE : {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")

    # ── 2. Leave-One-Refrigerant-Out CV ────────────────────────────────────────
    print("\nRunning Leave-One-Refrigerant-Out CV...")
    fluids = sorted(set(fluid_names))
    loocv_rows = []

    for fluid in fluids:
        test_mask  = fluid_names == fluid
        train_mask = ~test_mask
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        m = XGBRegressor(**MODEL_PARAMS)
        m.fit(X_tr, y_tr)
        y_hat = m.predict(X_te)

        r2   = r2_score(y_te, y_hat)
        rmse = np.sqrt(mean_squared_error(y_te, y_hat))
        loocv_rows.append({'Refrigerant': fluid, 'N_test': int(test_mask.sum()),
                           'R2': round(r2, 4), 'RMSE': round(rmse, 4)})
        print(f"  {fluid:12s} | N={int(test_mask.sum()):3d} | R²={r2:.4f} | RMSE={rmse:.4f}")

    loocv_df = pd.DataFrame(loocv_rows)
    print(f"\nLOOCV mean R² : {loocv_df['R2'].mean():.4f}")
    print(f"LOOCV min R²  : {loocv_df['R2'].min():.4f}  ({loocv_df.loc[loocv_df['R2'].idxmin(), 'Refrigerant']})")

    # ── 3. Learning Curve ──────────────────────────────────────────────────────
    print("\nGenerating learning curve (this may take a few minutes)...")
    tr_sizes, tr_scores, te_scores = learning_curve(
        XGBRegressor(**MODEL_PARAMS), X, y, cv=kf,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_root_mean_squared_error', n_jobs=-1,
    )
    t_mean, t_std = -np.mean(tr_scores, axis=1), np.std(tr_scores, axis=1)
    v_mean, v_std = -np.mean(te_scores, axis=1), np.std(te_scores, axis=1)

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.fill_between(tr_sizes, t_mean - t_std, t_mean + t_std, alpha=0.12, color='r')
    plt.fill_between(tr_sizes, v_mean - v_std, v_mean + v_std, alpha=0.12, color='g')
    plt.plot(tr_sizes, t_mean, 'o-', color='r', label='Training RMSE')
    plt.plot(tr_sizes, v_mean, 'o-', color='g', label='CV RMSE')
    plt.xlabel('Training Samples', fontsize=11)
    plt.ylabel('RMSE (COP)', fontsize=11)
    plt.title('Learning Curve — XGBoost (Leakage-Free Features)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    lc_path = os.path.join(FIG_DIR, 'learning_curve.png')
    plt.savefig(lc_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curve to {lc_path}")

    # ── 4. Save report ─────────────────────────────────────────────────────────
    with open(REPORT_PATH, 'w') as f:
        f.write("Model Stability & Generalization Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Features used ({len(ALL_FEATURES)}): {ALL_FEATURES}\n\n")
        f.write(f"5-Fold CV R²   : {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}\n")
        f.write(f"5-Fold CV RMSE : {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}\n\n")
        f.write("Leave-One-Refrigerant-Out CV:\n")
        f.write(loocv_df.to_string(index=False))
        f.write(f"\n\nLOOCV mean R² : {loocv_df['R2'].mean():.4f}\n")
        f.write(f"LOOCV min R²  : {loocv_df['R2'].min():.4f}"
                f"  ({loocv_df.loc[loocv_df['R2'].idxmin(), 'Refrigerant']})\n\n")
        f.write("Learning Curve Data:\n")
        f.write(f"Train sizes: {tr_sizes.tolist()}\n")
        f.write(f"Train RMSE : {t_mean.tolist()}\n")
        f.write(f"Val RMSE   : {v_mean.tolist()}\n")

    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    x_path = os.path.join(DATA_DIR, 'X_processed.csv')
    if not os.path.exists(x_path):
        print(f"Error: {x_path} not found — run ml_model.py first.")
    else:
        run_verification()

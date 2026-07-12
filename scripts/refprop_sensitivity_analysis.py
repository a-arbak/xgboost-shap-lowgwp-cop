"""
Sensitivity analysis for Reviewer #2 comment 1 (JIJR-D-26-00648):
does replacing the linear mixing-rule descriptors (Eq. 1) with REFPROP 10.0
rigorous equation-of-state critical points change the surrogate's accuracy?

Replicates the exact training setup of the manuscript (same stratified split,
random_state=42, same hyperparameters as plot_parity_testonly.py), then
re-evaluates with Tc_mix/Pc_mix of the seven blends replaced by REFPROP values
from refprop_critical_comparison.py. Reports held-out test metrics, 5-fold CV,
and Leave-One-Refrigerant-Out CV for both descriptor sets.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')

ALL_FEATURES = [
    'Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix', 'n_components', 'epsilon_LLSL',
    'T_liq_evap_in', 'T_liq_evap_out', 'superheat', 'DELTAT_dew_suction',
    'T_liq_cond_in', 'T_liq_cond_out', 'subcool', 'DELTAT_dew_discharge',
]
MODEL_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, random_state=42)

# REFPROP 10.0 EoS critical points (output of refprop_critical_comparison.py)
REFPROP_TC_PC = {
    'R513A':  (367.93, 3.631),
    'R450A':  (377.89, 3.813),
    'Tern1':  (371.28, 3.764),
    'R515B':  (381.84, 3.563),
    'R410A':  (342.42, 4.516),
    'R454B':  (352.42, 4.926),
    'R452B':  (351.42, 4.935),
}

FLUID_ORDER = ['R134a', 'R513A', 'R450A', 'Tern1',
               'R515B', 'R1234yf', 'R410A', 'R32', 'R454B', 'R452B']


def make_split(X_df, test_frac=0.20, random_state=42):
    test_idx = (
        X_df.groupby('fluid_name', group_keys=False)
        .apply(lambda g: g.sample(frac=test_frac, random_state=random_state),
               include_groups=False)
        .index
    )
    train_idx = X_df.index.difference(test_idx)
    return train_idx, test_idx


def heldout_metrics(X_df, y):
    train_idx, test_idx = make_split(X_df)
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X_df.loc[train_idx, ALL_FEATURES].values, y[train_idx])
    y_pred = model.predict(X_df.loc[test_idx, ALL_FEATURES].values)
    y_test = y[test_idx]
    return (r2_score(y_test, y_pred),
            float(np.sqrt(mean_squared_error(y_test, y_pred))),
            float(mean_absolute_error(y_test, y_pred)))


def cv5_metrics(X_df, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X = X_df[ALL_FEATURES].values
    r2s = []
    for tr, te in kf.split(X):
        m = XGBRegressor(**MODEL_PARAMS)
        m.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], m.predict(X[te])))
    return float(np.mean(r2s)), float(np.std(r2s))


def loocv_metrics(X_df, y):
    fluids = X_df['fluid_name'].values
    X = X_df[ALL_FEATURES].values
    scores = {}
    for fl in FLUID_ORDER:
        mask = fluids == fl
        m = XGBRegressor(**MODEL_PARAMS)
        m.fit(X[~mask], y[~mask])
        scores[fl] = r2_score(y[mask], m.predict(X[mask]))
    return scores


def run():
    X_base = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    y = pd.read_csv(os.path.join(DATA_DIR, 'y_labels.csv')).iloc[:, 0].values

    X_rp = X_base.copy()
    for fl, (tc, pc) in REFPROP_TC_PC.items():
        m = X_rp['fluid_name'] == fl
        X_rp.loc[m, 'Tc_mix'] = tc
        X_rp.loc[m, 'Pc_mix'] = pc

    print("=== Held-out test set (stratified 80/20, seed 42) ===")
    for name, X_df in [('linear rule', X_base), ('REFPROP EoS', X_rp)]:
        r2, rmse, mae = heldout_metrics(X_df, y)
        print(f"{name:12s}  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    print("\n=== 5-fold CV ===")
    for name, X_df in [('linear rule', X_base), ('REFPROP EoS', X_rp)]:
        mu, sd = cv5_metrics(X_df, y)
        print(f"{name:12s}  R2={mu:.4f} +/- {sd:.4f}")

    print("\n=== Leave-One-Refrigerant-Out CV ===")
    s_base = loocv_metrics(X_base, y)
    s_rp = loocv_metrics(X_rp, y)
    print(f"{'Fluid':10s} {'linear':>8s} {'REFPROP':>8s} {'diff':>8s}")
    for fl in FLUID_ORDER:
        print(f"{fl:10s} {s_base[fl]:8.3f} {s_rp[fl]:8.3f} {s_rp[fl]-s_base[fl]:8.3f}")
    mb = np.mean(list(s_base.values()))
    mr = np.mean(list(s_rp.values()))
    print(f"{'MEAN':10s} {mb:8.3f} {mr:8.3f} {mr-mb:8.3f}")


if __name__ == '__main__':
    run()

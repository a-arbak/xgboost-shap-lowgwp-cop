"""
Computes MAPE for the four candidate models on the held-out test set
(Reviewer #2 comment 3) and the dispersion of the LOOCV aggregate metrics
(Reviewer #3 comment 2), using the identical split and model configurations
as the manuscript (ml_model.py / former ml_comparison.py).
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')

ALL_FEATURES = [
    'Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix', 'n_components', 'epsilon_LLSL',
    'T_liq_evap_in', 'T_liq_evap_out', 'superheat', 'DELTAT_dew_suction',
    'T_liq_cond_in', 'T_liq_cond_out', 'subcool', 'DELTAT_dew_discharge',
]

MODELS = {
    'XGBoost': lambda: XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42),
    'Random Forest': lambda: RandomForestRegressor(
        n_estimators=500, max_depth=None, random_state=42, n_jobs=-1),
    'Ridge': lambda: Pipeline([
        ('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
    'MLP': lambda: Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                             activation='relu', max_iter=2000,
                             random_state=42, early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=30))]),
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
    return X_df.index.difference(test_idx), test_idx


def run():
    X_df = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    y = pd.read_csv(os.path.join(DATA_DIR, 'y_labels.csv')).iloc[:, 0].values
    train_idx, test_idx = make_split(X_df)
    X_train = X_df.loc[train_idx, ALL_FEATURES].values
    X_test = X_df.loc[test_idx, ALL_FEATURES].values
    y_train, y_test = y[train_idx.to_numpy()], y[test_idx.to_numpy()]

    print("=== Held-out test MAPE (and cross-check R2/RMSE) ===")
    for name, factory in MODELS.items():
        m = factory()
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        mape = 100 * np.mean(np.abs((y_test - p) / y_test))
        r2 = r2_score(y_test, p)
        rmse = np.sqrt(mean_squared_error(y_test, p))
        print(f"{name:14s}  MAPE = {mape:5.2f}%   (R2={r2:.4f}, RMSE={rmse:.4f})")

    print("\n=== LOOCV per-fluid metrics + aggregate dispersion (XGBoost) ===")
    fluids = X_df['fluid_name'].values
    X_all = X_df[ALL_FEATURES].values
    r2s, rmses = [], []
    for fl in FLUID_ORDER:
        mask = fluids == fl
        m = MODELS['XGBoost']()
        m.fit(X_all[~mask], y[~mask])
        p = m.predict(X_all[mask])
        r2s.append(r2_score(y[mask], p))
        rmses.append(float(np.sqrt(mean_squared_error(y[mask], p))))
        print(f"  {fl:8s}  R2={r2s[-1]:.3f}  RMSE={rmses[-1]:.3f}")
    r2s, rmses = np.array(r2s), np.array(rmses)
    print(f"\nLOOCV R2  : mean={r2s.mean():.3f}  sample-std={r2s.std(ddof=1):.3f}")
    print(f"LOOCV RMSE: mean={rmses.mean():.3f}  sample-std={rmses.std(ddof=1):.3f}")


if __name__ == '__main__':
    run()

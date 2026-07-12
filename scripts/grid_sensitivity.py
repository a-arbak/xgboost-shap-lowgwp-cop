"""
Hyperparameter grid sensitivity analysis for Reviewer #2 comment 2
(JIJR-D-26-00648): quantifies how strongly the 5-fold CV performance depends
on the hyperparameter choice within the coarse grid used in the manuscript
(learning rate {0.01, 0.05, 0.1} x max depth {4, 6, 8} x subsample {0.6, 0.8}).

CV is performed on the 200-sample training pool only (same stratified split as
the manuscript), so the held-out test set plays no role in model selection.
A flat optimum across the grid demonstrates that the model is insensitive to
the exact hyperparameter choice, i.e. the coarse grid is sufficient.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from itertools import product
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')

ALL_FEATURES = [
    'Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix', 'n_components', 'epsilon_LLSL',
    'T_liq_evap_in', 'T_liq_evap_out', 'superheat', 'DELTAT_dew_suction',
    'T_liq_cond_in', 'T_liq_cond_out', 'subcool', 'DELTAT_dew_discharge',
]

GRID = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.6, 0.8],
}
SELECTED = dict(learning_rate=0.05, max_depth=6, subsample=0.8)


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
    train_idx, _ = make_split(X_df)
    X = X_df.loc[train_idx, ALL_FEATURES].values
    yt = y[train_idx.to_numpy()]
    print(f"Training pool: {len(X)} samples; grid size: "
          f"{np.prod([len(v) for v in GRID.values()])} configurations\n")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for lr, md, ss in product(*GRID.values()):
        r2s = []
        for tr, te in kf.split(X):
            m = XGBRegressor(n_estimators=1000, learning_rate=lr, max_depth=md,
                             subsample=ss, colsample_bytree=0.8,
                             random_state=42)
            m.fit(X[tr], yt[tr])
            r2s.append(r2_score(yt[te], m.predict(X[te])))
        mu, sd = float(np.mean(r2s)), float(np.std(r2s))
        sel = (lr == SELECTED['learning_rate'] and md == SELECTED['max_depth']
               and ss == SELECTED['subsample'])
        results.append((lr, md, ss, mu, sd, sel))
        print(f"lr={lr:<5} depth={md} subsample={ss}:  "
              f"CV R2 = {mu:.4f} +/- {sd:.4f}{'   <-- selected' if sel else ''}")

    mus = np.array([r[3] for r in results])
    print(f"\nAcross all {len(results)} configurations:")
    print(f"  min    = {mus.min():.4f}")
    print(f"  median = {np.median(mus):.4f}")
    print(f"  max    = {mus.max():.4f}")
    print(f"  range  = {mus.max() - mus.min():.4f}")


if __name__ == '__main__':
    run()

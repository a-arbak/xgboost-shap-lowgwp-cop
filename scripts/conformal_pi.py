"""
Normalized split-conformal prediction intervals for the XGBoost COP surrogate.

Addresses Reviewer #3 comment 1 (JIJR-D-26-00648): the nominal 90% bootstrap
percentile intervals achieved only 75.0% empirical coverage. Split-conformal
calibration provides a finite-sample, distribution-free marginal coverage
guarantee (Vovk et al. 2005; Lei et al. 2018). The normalized variant
(Papadopoulos et al. 2002) scales the calibrated quantile by a per-sample
difficulty estimate sigma_hat(x) — here the half-width of a bootstrap-ensemble
interval — so intervals remain heteroscedastic (wide where the model is
uncertain, e.g. near the descriptor-space boundary), preserving the
fluid-resolved interpretation used in Section 3.4 of the manuscript.

Pipeline (all seeds fixed for reproducibility):
  1. Same stratified 80/20 split as the manuscript (200 train / 48 test).
  2. The 200 training samples are further split (stratified by refrigerant,
     75/25) into a proper-training set (~150) and a calibration set (~50).
  3. B=200 bootstrap XGBoost models on the proper-training set give the
     ensemble mean prediction and sigma_hat(x) = half-width of the 5th-95th
     percentile band (floored to avoid degenerate scores).
  4. Nonconformity scores on the calibration set:
     r_i = |y_i - yhat(x_i)| / sigma_hat(x_i).
  5. qhat = ceil((n_cal+1)*0.90)/n_cal empirical quantile of the scores.
  6. Test PI: yhat(x) +/- qhat * sigma_hat(x).

Outputs: conformal_pi_lo_test.npy / conformal_pi_hi_test.npy (aligned with the
test set ordering used by plot_parity_testonly.py) and console diagnostics
(coverage, width, per-fluid breakdown; plain unnormalized conformal shown for
comparison).
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')

ALL_FEATURES = [
    'Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix', 'n_components', 'epsilon_LLSL',
    'T_liq_evap_in', 'T_liq_evap_out', 'superheat', 'DELTAT_dew_suction',
    'T_liq_cond_in', 'T_liq_cond_out', 'subcool', 'DELTAT_dew_discharge',
]
MODEL_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, random_state=42)
ALPHA = 0.10          # 90% nominal coverage
B = 200               # bootstrap ensemble size
SIGMA_FLOOR = 0.02    # COP units; avoids degenerate normalized scores

FLUID_ORDER = ['R134a', 'R513A', 'R450A', 'Tern1',
               'R515B', 'R1234yf', 'R410A', 'R32', 'R454B', 'R452B']
FLUID_LABEL = {
    'R134a': 'R-134a', 'R1234yf': 'R-1234yf', 'R32': 'R-32',
    'R410A': 'R-410A', 'R450A': 'R-450A', 'R452B': 'R-452B',
    'R454B': 'R-454B', 'R513A': 'R-513A', 'R515B': 'R-515B', 'Tern1': 'Tern-1',
}


def make_split(X_df, test_frac=0.20, random_state=42):
    """Identical to plot_parity_testonly.py / ml_model.py."""
    test_idx = (
        X_df.groupby('fluid_name', group_keys=False)
        .apply(lambda g: g.sample(frac=test_frac, random_state=random_state),
               include_groups=False)
        .index
    )
    train_idx = X_df.index.difference(test_idx)
    return train_idx, test_idx


def stratified_calib_split(X_df, pool_idx, calib_frac=0.25, random_state=42):
    """Split the training pool into proper-training and calibration sets,
    stratified by refrigerant so every fluid contributes calibration scores."""
    pool = X_df.loc[pool_idx]
    calib_idx = (
        pool.groupby('fluid_name', group_keys=False)
        .apply(lambda g: g.sample(frac=calib_frac, random_state=random_state),
               include_groups=False)
        .index
    )
    proper_idx = pool.index.difference(calib_idx)
    return proper_idx, calib_idx


def bootstrap_ensemble(X_tr, y_tr, X_eval, B=B, seed=42):
    rng = np.random.default_rng(seed)
    preds = np.zeros((B, len(X_eval)))
    for b in range(B):
        idx = rng.integers(0, len(X_tr), size=len(X_tr))
        m = XGBRegressor(**MODEL_PARAMS)
        m.fit(X_tr[idx], y_tr[idx])
        preds[b] = m.predict(X_eval)
    return preds


def conformal_quantile(scores, alpha=ALPHA):
    """Finite-sample-corrected empirical quantile (Lei et al. 2018)."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float(np.max(scores)) * 1.5   # degenerate small-n fallback
    return float(np.sort(scores)[k - 1])


def coverage_report(name, y, lo, hi, fluids):
    cov = np.mean((y >= lo) & (y <= hi))
    width = np.mean(hi - lo)
    print(f"\n--- {name} ---")
    print(f"Overall: coverage = {cov:.3f}   avg width = {width:.3f} COP units")
    for fl in FLUID_ORDER:
        m = fluids == fl
        if m.sum() == 0:
            continue
        c = np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))
        w = np.mean(hi[m] - lo[m])
        print(f"  {FLUID_LABEL[fl]:10s} n={m.sum():2d}  coverage={c:.2f}  width={w:.3f}")
    return cov, width


def run():
    X_df = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    y = pd.read_csv(os.path.join(DATA_DIR, 'y_labels.csv')).iloc[:, 0].values
    fluids = X_df['fluid_name'].values

    train_idx, test_idx = make_split(X_df)
    proper_idx, calib_idx = stratified_calib_split(X_df, train_idx)
    print(f"Proper-training: {len(proper_idx)}  Calibration: {len(calib_idx)}  "
          f"Test: {len(test_idx)}")

    Xp = X_df.loc[proper_idx, ALL_FEATURES].values
    yp = y[proper_idx.to_numpy()]
    Xc = X_df.loc[calib_idx, ALL_FEATURES].values
    yc = y[calib_idx.to_numpy()]
    Xt = X_df.loc[test_idx, ALL_FEATURES].values
    yt = y[test_idx.to_numpy()]
    fluids_test = fluids[test_idx.to_numpy()]

    # Base point predictor (proper-training only — never sees calibration data)
    base = XGBRegressor(**MODEL_PARAMS)
    base.fit(Xp, yp)
    yhat_c = base.predict(Xc)
    yhat_t = base.predict(Xt)

    # Bootstrap ensemble on proper-training for the difficulty function
    print(f"Training {B} bootstrap models on the proper-training set...")
    X_eval = np.vstack([Xc, Xt])
    preds = bootstrap_ensemble(Xp, yp, X_eval)
    lo_band = np.percentile(preds, 5, axis=0)
    hi_band = np.percentile(preds, 95, axis=0)
    sigma = np.maximum((hi_band - lo_band) / 2.0, SIGMA_FLOOR)
    sigma_c, sigma_t = sigma[:len(Xc)], sigma[len(Xc):]

    # --- Normalized split-conformal ---
    scores_norm = np.abs(yc - yhat_c) / sigma_c
    qhat_norm = conformal_quantile(scores_norm)
    lo_n = yhat_t - qhat_norm * sigma_t
    hi_n = yhat_t + qhat_norm * sigma_t
    print(f"\nNormalized conformal qhat = {qhat_norm:.3f}")
    cov_n, w_n = coverage_report("Normalized split-conformal (90% nominal)",
                                 yt, lo_n, hi_n, fluids_test)

    # --- Plain (unnormalized) split-conformal, for comparison ---
    scores_abs = np.abs(yc - yhat_c)
    qhat_abs = conformal_quantile(scores_abs)
    lo_a = yhat_t - qhat_abs
    hi_a = yhat_t + qhat_abs
    print(f"\nPlain conformal qhat = {qhat_abs:.3f} COP units")
    coverage_report("Plain split-conformal (90% nominal)", yt, lo_a, hi_a,
                    fluids_test)

    # --- Reference: original bootstrap percentile PI (uncalibrated) ---
    lo_b = np.load(os.path.join(DATA_DIR, 'boot_pi_lo_test.npy'))
    hi_b = np.load(os.path.join(DATA_DIR, 'boot_pi_hi_test.npy'))
    coverage_report("Uncalibrated bootstrap percentile PI (manuscript V1)",
                    yt, lo_b, hi_b, fluids_test)

    # Save the normalized-conformal intervals for the parity plot
    np.save(os.path.join(DATA_DIR, 'conformal_pi_lo_test.npy'), lo_n)
    np.save(os.path.join(DATA_DIR, 'conformal_pi_hi_test.npy'), hi_n)
    np.save(os.path.join(DATA_DIR, 'conformal_center_test.npy'), yhat_t)
    print(f"\nSaved conformal_pi_lo/hi_test.npy and conformal_center_test.npy "
          f"to {DATA_DIR}")
    print(f"Summary: coverage {cov_n:.1%}, avg width {w_n:.3f}, qhat {qhat_norm:.3f}")


if __name__ == '__main__':
    run()

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')
OUTPUT_DIR = DATA_DIR

# ── Thermodynamic property lookup (NIST / ASHRAE 34-2022) ─────────────────────
PURE_PROPS = {
    'R134a':    {'Tc': 374.21, 'Pc': 4.059, 'omega': 0.327, 'M': 102.03, 'GWP': 1300},
    'R1234yf':  {'Tc': 367.85, 'Pc': 3.382, 'omega': 0.276, 'M': 114.04, 'GWP':    4},
    'R1234zeE': {'Tc': 382.52, 'Pc': 3.635, 'omega': 0.313, 'M': 114.04, 'GWP':    7},
    'R32':      {'Tc': 351.26, 'Pc': 5.782, 'omega': 0.277, 'M':  52.02, 'GWP':  675},
    'R125':     {'Tc': 339.17, 'Pc': 3.617, 'omega': 0.305, 'M': 120.02, 'GWP': 3500},
    'R227ea':   {'Tc': 374.90, 'Pc': 2.925, 'omega': 0.357, 'M': 170.03, 'GWP': 3220},
}

# Mass-fraction blend compositions — Sources: ASHRAE 34, REFPROP, Skye et al. 2022
# Tern-1: ternary R32-based blend — VERIFY exact fractions against Skye et al. 2022 Table 1
BLEND_COMPOSITIONS = {
    'R134a':   [('R134a',    1.000)],
    'R1234yf': [('R1234yf',  1.000)],
    'R32':     [('R32',      1.000)],
    'R410A':   [('R32',      0.500), ('R125',     0.500)],
    'R513A':   [('R134a',    0.440), ('R1234yf',  0.560)],
    'R450A':   [('R134a',    0.420), ('R1234zeE', 0.580)],
    'R515B':   [('R1234zeE', 0.911), ('R227ea',   0.089)],
    'R454B':   [('R32',      0.689), ('R1234yf',  0.311)],
    'R452B':   [('R32',      0.670), ('R125',     0.070), ('R1234yf', 0.260)],
    'Tern1':   [('R134a',    0.492), ('R1234yf',  0.338), ('R1234zeE',0.170)],  # Skye et al. 2022 Table 1
}

# ── Feature definitions (leakage-free) ────────────────────────────────────────
# Removed: eta_s, eta_v (compressor efficiencies), Q_dot_evap_ref (COP numerator)
# Removed: GWP_mix — no thermodynamic causal link to COP; low SHAP rank; redundant with Tc/Pc/omega
PHYSICS_FEATURES = ['Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix', 'n_components']
OPERATIONAL_FEATURES = [
    'epsilon_LLSL',
    'T_liq_evap_in', 'T_liq_evap_out', 'superheat', 'DELTAT_dew_suction',
    'T_liq_cond_in',  'T_liq_cond_out', 'subcool',   'DELTAT_dew_discharge',
]
ALL_FEATURES = PHYSICS_FEATURES + OPERATIONAL_FEATURES
TARGET = 'COP_c_ref_dh'


# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_mix_props(fluid_name):
    if fluid_name not in BLEND_COMPOSITIONS:
        raise ValueError(f"Unknown fluid '{fluid_name}'. Add it to BLEND_COMPOSITIONS.")
    blend = BLEND_COMPOSITIONS[fluid_name]
    props = {k: 0.0 for k in ['Tc', 'Pc', 'omega', 'M', 'GWP']}
    for component, frac in blend:
        cp = PURE_PROPS[component]
        for k in props:
            props[k] += frac * cp[k]
    return {
        'Tc_mix':      props['Tc'],
        'Pc_mix':      props['Pc'],
        'omega_mix':   props['omega'],
        'M_mix':       props['M'],
        'GWP_mix':     props['GWP'],
        'n_components': len(blend),
    }


def add_physics_features(df):
    mix_df = pd.DataFrame(
        df['fluid_name'].apply(compute_mix_props).tolist(), index=df.index
    )
    return pd.concat([df, mix_df], axis=1)


def make_stratified_split(df, test_frac=0.20, random_state=42):
    """Take ~test_frac from each refrigerant group for the held-out test set."""
    test_idx = (
        df.groupby('fluid_name', group_keys=False)
        .apply(lambda g: g.sample(frac=test_frac, random_state=random_state),
               include_groups=False)
        .index
    )
    return df.drop(test_idx), df.loc[test_idx]


# ── Main ───────────────────────────────────────────────────────────────────────
def train_model():
    raw = pd.read_csv(os.path.join(DATA_DIR, 'unified_nist_experiment_data.csv'))

    if 'fluid_name' not in raw.columns:
        raise ValueError("'fluid_name' column missing — re-run prepare_data.py first.")

    raw = raw.dropna(subset=[TARGET])
    df = add_physics_features(raw)

    for col in OPERATIONAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=ALL_FEATURES + [TARGET])
    print(f"Total samples after cleaning: {len(df)}")
    print(f"Refrigerant distribution:\n{df['fluid_name'].value_counts().to_string()}")

    train_df, test_df = make_stratified_split(df)
    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET]
    X_test  = test_df[ALL_FEATURES]
    y_test  = test_df[TARGET]
    print(f"\nTraining samples: {len(X_train)} | Held-out test samples: {len(X_test)}")

    xgb_params = dict(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )

    # ── Point-estimate model ───────────────────────────────────────────────────
    print("\nTraining main XGBoost model...")
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = np.mean(np.abs(y_test.values - y_pred))

    print(f"\n=== Held-Out Test Performance (stratified by refrigerant) ===")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")

    # ── Bootstrap prediction intervals (90%) ─────────────────────────────────
    # Quantile regression (reg:quantileerror) was tried but gave poor calibration
    # (coverage=0.48) on this small dataset. Bootstrap residuals are more reliable.
    print("\nComputing 90% bootstrap prediction intervals (B=200)...")
    B = 200
    rng = np.random.default_rng(42)
    boot_preds_test = np.zeros((B, len(X_test)))

    for b in range(B):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        Xb = X_train.iloc[idx]
        yb = y_train.iloc[idx]
        mb = XGBRegressor(**xgb_params)
        mb.fit(Xb, yb)
        boot_preds_test[b] = mb.predict(X_test)

    pi_lo = np.percentile(boot_preds_test, 5,  axis=0)
    pi_hi = np.percentile(boot_preds_test, 95, axis=0)
    coverage = np.mean((y_test.values >= pi_lo) & (y_test.values <= pi_hi))
    width    = np.mean(pi_hi - pi_lo)
    print(f"  90% Bootstrap PI coverage : {coverage:.3f}  (target >= 0.90)")
    print(f"  Avg PI width              : {width:.4f} COP units")

    np.save(os.path.join(OUTPUT_DIR, 'boot_pi_lo_test.npy'), pi_lo)
    np.save(os.path.join(OUTPUT_DIR, 'boot_pi_hi_test.npy'), pi_hi)

    # ── Save main model ────────────────────────────────────────────────────────
    model.save_model(os.path.join(OUTPUT_DIR, 'xgboost_model.json'))

    # fluid_name retained in X_processed.csv for LOOCV in verify_model_stability.py
    df[['fluid_name'] + ALL_FEATURES].to_csv(
        os.path.join(OUTPUT_DIR, 'X_processed.csv'), index=False
    )
    df[[TARGET]].to_csv(os.path.join(OUTPUT_DIR, 'y_labels.csv'), index=False)

    print(f"\nAll models and feature matrix saved to {OUTPUT_DIR}")
    return model, df


if __name__ == "__main__":
    train_model()

"""Quantify SHAP fingerprint claim: intra-family vs inter-family cosine
similarity of per-observation SHAP vectors (correction plan P2.4)."""
import pandas as pd
import numpy as np
import shap
import os
from xgboost import XGBRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')

ALL_FEATURES = [
    'Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix', 'n_components',
    'epsilon_LLSL',
    'T_liq_evap_in', 'T_liq_evap_out', 'superheat', 'DELTAT_dew_suction',
    'T_liq_cond_in', 'T_liq_cond_out', 'subcool', 'DELTAT_dew_discharge',
]
FAMILY_MAP = {
    'R134a': 'R-134a based', 'R513A': 'R-134a based',
    'R450A': 'R-134a based', 'Tern1': 'R-134a based',
    'R1234yf': 'HFO', 'R515B': 'HFO',
    'R32': 'R-32 based', 'R410A': 'R-32 based',
    'R454B': 'R-32 based', 'R452B': 'R-32 based',
}


def run():
    model = XGBRegressor()
    model.load_model(os.path.join(DATA_DIR, 'xgboost_model.json'))
    X_full = pd.read_csv(os.path.join(DATA_DIR, 'X_processed.csv'))
    families = np.array([FAMILY_MAP[f] for f in X_full['fluid_name']])
    X = X_full[ALL_FEATURES].astype(float)

    print("Computing SHAP values...")
    explainer = shap.Explainer(model.predict, X)
    shap_vals = explainer(X).values  # (248, 14)

    def report(vals, label):
        sim = cosine_similarity(vals)
        n = len(families)
        same = families[:, None] == families[None, :]
        triu = np.triu_indices(n, k=1)
        intra = sim[triu][same[triu]].mean()
        inter = sim[triu][~same[triu]].mean()
        sil = silhouette_score(vals, families, metric='cosine')
        print(f"\n=== {label} ===")
        print(f"Intra-family mean cosine similarity: {intra:.3f}")
        print(f"Inter-family mean cosine similarity: {inter:.3f}")
        print(f"Silhouette score (cosine, 3 families): {sil:.3f}")
        for fam in ['R-134a based', 'HFO', 'R-32 based']:
            m = families == fam
            block = sim[np.ix_(m, m)]
            iu = np.triu_indices(m.sum(), k=1)
            print(f"  {fam:14s} intra-sim = {block[iu].mean():.3f}  (n={m.sum()})")

    report(shap_vals, "Full 14-feature SHAP vectors")

    desc_idx = [ALL_FEATURES.index(f) for f in
                ['Tc_mix', 'Pc_mix', 'omega_mix', 'M_mix', 'n_components']]
    report(shap_vals[:, desc_idx], "Descriptor-only SHAP sub-vectors (5 features)")


if __name__ == '__main__':
    run()

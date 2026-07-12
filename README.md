# Interpretable ML for COP Prediction in Low-GWP Refrigerants

> **Status:** The accompanying manuscript is currently **under peer review**
> at the International Journal of Refrigeration and has **not yet been
> published**. Content may be revised during the review process.

Code, processed data, and figure scripts accompanying the manuscript:

> A. Arbak, *Interpretable Machine Learning for COP Prediction in Vapour
> Compression Cycles with Low-GWP Refrigerants: A SHAP-Based Analysis of NIST
> Experimental Data*, under review at the International Journal of
> Refrigeration (Ms. Ref. No. JIJR-D-26-00648).

An XGBoost surrogate model predicts the cooling coefficient of performance
(COP) of vapour compression cycles from 248 steady-state experimental
observations spanning ten low-GWP working fluids (NIST Technical Note 2233).
Refrigerants are represented by continuous physics-based mixture descriptors
(critical temperature, critical pressure, acentric factor, molar mass) instead
of categorical labels, enabling cross-fluid generalisation, which is verified
by Leave-One-Refrigerant-Out cross-validation. SHAP provides feature-level
interpretation, and normalized split-conformal prediction intervals quantify
per-sample uncertainty with a finite-sample coverage guarantee.

## Repository layout

```
data/       Processed feature matrix, labels, trained model, PI arrays
scripts/    Full pipeline: data prep, training, validation, XAI, figures
figures/    Generated publication figures (300 dpi PNG)
```

## Setup

```bash
pip install -r requirements.txt
```

Python >= 3.10. Optional: `scripts/refprop_critical_comparison.py` requires a
local [NIST REFPROP 10.0](https://www.nist.gov/srd/refprop) installation
(licensed software, not included); set the `RPPREFIX` environment variable to
its installation directory if it is not in the default location. All other
scripts run without REFPROP.

## Data provenance

The raw experimental files originate from the public NIST dataset
["Data for NIST Technical Note 2233"](https://doi.org/10.18434/mds2-2613)
(Skye et al., 2022, [doi:10.6028/NIST.TN.2233](https://doi.org/10.6028/NIST.TN.2233)).
They are **not** redistributed here; `data/unified_nist_experiment_data.csv`
is the merged/cleaned derivative produced by `scripts/prepare_data.py`, and
`data/X_processed.csv` / `data/y_labels.csv` are the model-ready feature
matrix and labels produced by `scripts/ml_model.py`. To rebuild from scratch,
download the NIST dataset and set `NIST_RAW_DIR` (see `prepare_data.py`).

## Pipeline

Run order for full reproduction (each step is optional if you use the bundled
intermediate files in `data/`):

| # | Script | Purpose |
|---|--------|---------|
| 1 | `prepare_data.py` | Merge raw NIST CSVs into a unified dataset |
| 2 | `ml_model.py` | Feature engineering, training, held-out evaluation |
| 3 | `verify_model_stability.py` | 5-fold CV, LOOCV, learning curve |
| 4 | `conformal_pi.py` | Normalized split-conformal 90% prediction intervals |
| 5 | `xai_interpreter.py` | SHAP values, summary/bar plots |
| 6 | `compute_shap_fingerprint.py` | Per-fluid SHAP fingerprints, cosine similarity |
| 7 | `compute_mape_dispersion.py` | MAPE for all candidate models, LOOCV dispersion |
| 8 | `grid_sensitivity.py` | Hyperparameter grid sensitivity (18 configurations) |
| 9 | `refprop_critical_comparison.py` | Linear mixing rule vs. REFPROP critical points |
| 10 | `refprop_sensitivity_analysis.py` | Retraining with REFPROP-corrected descriptors |

## Figure scripts

| Figure | Script |
|--------|--------|
| Fig. 3 COP distribution | `plot_cop_stacked.py` |
| Fig. 4 correlation heatmap | `plot_correlation_heatmap.py` |
| Fig. 5 parity plot + conformal PIs | `plot_parity_testonly.py` |
| Fig. 6 SHAP summary | `xai_interpreter.py` |
| Fig. 7 SHAP heatmap | `plot_shap_heatmap.py` |
| Fig. 8a–d SHAP dependence | `plot_shap_dependence_8b.py` |
| Fig. 9 applicability domain | `plot_applicability_domain.py` |

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code or data, please cite the manuscript above (currently
under review; full citation details will be added upon acceptance). A Zenodo
DOI for this repository will be added upon archiving.

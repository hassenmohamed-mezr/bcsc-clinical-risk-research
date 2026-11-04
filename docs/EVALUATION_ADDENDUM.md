# Evaluation Addendum

## Corrected Validation Numbers
Validation metrics were corrected to distinguish Training CV from Validation. Source: `outputs/training/metrics/model_comparison.csv`.

Training CV (mean ± std):
- Gradient Boosting: PR-AUC 0.008291 ± 0.000335, ROC-AUC 0.631928 ± 0.003919
- Logistic Regression: PR-AUC 0.008290 ± 0.000372, ROC-AUC 0.630843 ± 0.006087
- MLP: PR-AUC 0.006814 ± 0.000443, ROC-AUC 0.592117 ± 0.015102
- Random Forest: PR-AUC 0.005412 ± 0.000145, ROC-AUC 0.512952 ± 0.009502

Validation:
- Gradient Boosting: PR-AUC 0.007846, ROC-AUC 0.626944, Brier 0.004855, BSS 0.000976
- Logistic Regression: PR-AUC 0.007885, ROC-AUC 0.624226, Brier 0.004855, BSS 0.000972
- MLP: PR-AUC 0.006935, ROC-AUC 0.594442, Brier 0.004861, BSS -0.000253
- Random Forest: PR-AUC 0.005265, ROC-AUC 0.507778, Brier 0.004860, BSS 0.000004

Weighted prevalence is about 0.49%; PR-AUC must be read against that baseline. Recall at threshold 0.5 is 0 by construction at this prevalence.

## Bootstrap and Calibration
- Outputs: `outputs/evaluation/metrics/bootstrap_ci.csv`, `paired_bootstrap_ci.csv`, `bss_baselines.csv`.
- Paired bootstrap vs all4: LR+GB ROC-AUC CI entirely above 0; PR-AUC CI includes 0. lr_gb_mlp ROC-AUC CI entirely above 0 and PR-AUC CI entirely above 0. lr_only CIs include 0.
- BSS computed against both train prevalence (0.488378%) and validation prevalence; values are nearly identical.

## Decile and Lift
- `outputs/evaluation/metrics/decile_table.csv` and plots under `outputs/evaluation/plots/`.
- Calibration-by-decile and lift charts generated for LR, GB, and best candidate (`lr_gb`).

## Baseline Note
- Bootstrap resampling unit is the aggregated validation row carrying its `count` weight.
- No row expansion was used.

## Ensemble Findings
- Recommended candidate: `lr_gb` (paired bootstrap shows ROC-AUC improvement vs all4; PR-AUC CI includes 0).
- Proposal only; not applied. See `outputs/evaluation/reports/proposed_model_weights.md`.
- `AWAITING APPROVAL`.

## Input Scan and Monotonic Findings
- `tests_extra/run_input_scan.py` generates `outputs/evaluation/reports/input_scan.csv` (5000 seeded valid patients).
- Monotonic report: `outputs/evaluation/reports/monotonic_report.csv`.
- Sanity tests passed (4/4).

## Limitations
- Validation data was reused for ensemble comparison.
- Test split is consumed and cannot be used for further decisions.
- Reference ranges are approximate, not ground truth.
- No clinical accuracy or calibration claims are established.
# Project State Matrix

## Scope and verification method

This matrix records the state of the supplied project before migration. Files
were inspected by path, size, hash, text headers, and notebook JSON/source
structure. Existing scripts and notebooks were not executed as part of the
inspection or migration. The original workspace contained 43 files.

## File inventory and migration map

| File name | Original path | Type | Current status | Verified content | Related outputs | Proposed destination | Action | Completion status | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `risk.txt` | `txt_to_csv/risk.txt` | TXT dataset | Existing, raw | Source text dataset; exact content preserved | `risk.csv` conversions | `data/01_raw/risk.txt` | Move | Completed | External provenance not verified |
| `risk.csv` | `txt_to_csv/risk.csv` | CSV dataset | Existing, raw | 280,660 data rows; 16 columns; includes `count` | Source for analyses | `data/01_raw/risk.csv` | Move | Completed | SHA-256 matches the other raw CSV copies |
| `risk.csv` | `EDA_data_set/risk.csv` | CSV dataset | Existing, duplicate raw copy | Same header and hash as raw CSV | EDA and association notebooks | `data/01_raw/risk_from_eda_data_set.csv` | Move | Completed | Preserved with source-location suffix |
| `risk.csv` | `drop_same_data/risk.csv` | CSV dataset | Existing, duplicate raw copy | Same header and hash as raw CSV | `drop.ipynb` | `data/01_raw/risk_from_drop_same_data.csv` | Move | Completed | Preserved with source-location suffix |
| `risk_filtered.csv` | `EDA_data_set/risk_filtered.csv` | CSV dataset | Existing, interim | 280,660 data rows; 13 columns; excluded `race`, `Hispanic`, `invasive` | `drop.ipynb` and later analysis context | `data/02_interim/risk_filtered.csv` | Move | Completed | Existing generated intermediate |
| `risk_filtered.csv` | `drop_same_data/risk_filtered.csv` | CSV dataset | Existing, duplicate interim | Same header and hash as the other filtered copy | `drop.ipynb` | `data/02_interim/risk_filtered_from_drop_same_data.csv` | Move | Completed | Preserved with source-location suffix |
| `risk_to_csv.py` | `txt_to_csv/risk_to_csv.py` | Python script | Existing, completed conversion utility | Converts the text source to CSV | `risk.csv` | `src/data/risk_to_csv.py` | Move | Completed | Source text preserved byte-for-byte |
| `drop.ipynb` | `drop_same_data/drop.ipynb` | Notebook | Existing, completed/intermediate | Loads data, drops excluded features, saves and validates filtered data | `risk_filtered.csv` | `notebooks/03_preprocessing/drop.ipynb` | Move | Completed | No claim of complete preprocessing |
| `EDA.ipynb` | `EDA_data_set/EDA.ipynb` | Notebook | Existing, completed exploratory work | Weighted descriptive summaries, cancer rates, plots, training split summary | EDA figures and displayed outputs | `notebooks/02_eda/EDA.ipynb` | Move | Completed | Outputs and execution history preserved |
| `Full Feature vs Cancer.ipynb` | `EDA_data_set/Full Feature vs Cancer.ipynb` | Notebook | Existing, completed exploratory association work | Weighted contingency tables, Wilson intervals, chi-square, Cramér's V | Feature tables and figures | `notebooks/02_eda/Full Feature vs Cancer.ipynb` | Move | Completed | Existing notebook preserved as supplied |
| `Visualization_2d&3d.ipynb` | `EDA_data_set/Visualization_2d&3d.ipynb` | Notebook | Existing, completed exploratory visualization | Weighted PCA 2D/3D and t-SNE 2D visualization | PCA/t-SNE figures and coordinates | `notebooks/02_eda/Visualization_2d&3d.ipynb` | Move | Completed | Not a predictive evaluation |
| `pca_2d_coordinates.csv` | `EDA_data_set/pca_2d_coordinates.csv` | CSV result | Existing, generated EDA output | PC1, PC2, cancer, count; 280,660 data rows | PCA 2D visualization | `outputs/eda/tables/pca_2d_coordinates.csv` | Move | Completed | Exploratory coordinates |
| `ALL_FEATURES_ASSOCIATION_SUMMARY.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/ALL_FEATURES_ASSOCIATION_SUMMARY.csv` | CSV result | Existing, generated EDA output | Feature association statistics and corrected V ranking | Feature-rate tables and plots | `outputs/eda/tables/ALL_FEATURES_ASSOCIATION_SUMMARY.csv` | Move | Completed | 12 feature records |
| `agefirst_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/agefirst_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `agefirst_vs_cancer.png` | `outputs/eda/tables/agefirst_cancer_rates.csv` | Move | Completed | |
| `agegrp_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/agegrp_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `agegrp_vs_cancer.png` | `outputs/eda/tables/agegrp_cancer_rates.csv` | Move | Completed | |
| `bmi_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/bmi_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `bmi_vs_cancer.png` | `outputs/eda/tables/bmi_cancer_rates.csv` | Move | Completed | |
| `brstproc_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/brstproc_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `brstproc_vs_cancer.png` | `outputs/eda/tables/brstproc_cancer_rates.csv` | Move | Completed | |
| `density_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/density_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `density_vs_cancer.png` | `outputs/eda/tables/density_cancer_rates.csv` | Move | Completed | |
| `Hispanic_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/Hispanic_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `Hispanic_vs_cancer.png` | `outputs/eda/tables/Hispanic_cancer_rates.csv` | Move | Completed | |
| `hrt_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/hrt_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `hrt_vs_cancer.png` | `outputs/eda/tables/hrt_cancer_rates.csv` | Move | Completed | |
| `lastmamm_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/lastmamm_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `lastmamm_vs_cancer.png` | `outputs/eda/tables/lastmamm_cancer_rates.csv` | Move | Completed | |
| `menopaus_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/menopaus_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `menopaus_vs_cancer.png` | `outputs/eda/tables/menopaus_cancer_rates.csv` | Move | Completed | |
| `nrelbc_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/nrelbc_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `nrelbc_vs_cancer.png` | `outputs/eda/tables/nrelbc_cancer_rates.csv` | Move | Completed | |
| `race_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/race_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `race_vs_cancer.png` | `outputs/eda/tables/race_cancer_rates.csv` | Move | Completed | Subgroup/fairness relevance not resolved |
| `surgmeno_cancer_rates.csv` | `EDA_data_set/BCSC_Full_Feature_Analysis/surgmeno_cancer_rates.csv` | CSV result | Existing, generated EDA output | Category counts, rates, Wilson intervals | `surgmeno_vs_cancer.png` | `outputs/eda/tables/surgmeno_cancer_rates.csv` | Move | Completed | |
| `agefirst_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/agefirst_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `agefirst_cancer_rates.csv` | `outputs/eda/figures/agefirst_vs_cancer.png` | Move | Completed | |
| `agegrp_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/agegrp_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `agegrp_cancer_rates.csv` | `outputs/eda/figures/agegrp_vs_cancer.png` | Move | Completed | |
| `bmi_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/bmi_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `bmi_cancer_rates.csv` | `outputs/eda/figures/bmi_vs_cancer.png` | Move | Completed | |
| `brstproc_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/brstproc_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `brstproc_cancer_rates.csv` | `outputs/eda/figures/brstproc_vs_cancer.png` | Move | Completed | |
| `density_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/density_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `density_cancer_rates.csv` | `outputs/eda/figures/density_vs_cancer.png` | Move | Completed | |
| `Hispanic_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/Hispanic_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `Hispanic_cancer_rates.csv` | `outputs/eda/figures/Hispanic_vs_cancer.png` | Move | Completed | |
| `hrt_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/hrt_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `hrt_cancer_rates.csv` | `outputs/eda/figures/hrt_vs_cancer.png` | Move | Completed | |
| `lastmamm_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/lastmamm_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `lastmamm_cancer_rates.csv` | `outputs/eda/figures/lastmamm_vs_cancer.png` | Move | Completed | |
| `menopaus_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/menopaus_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `menopaus_cancer_rates.csv` | `outputs/eda/figures/menopaus_vs_cancer.png` | Move | Completed | |
| `nrelbc_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/nrelbc_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `nrelbc_cancer_rates.csv` | `outputs/eda/figures/nrelbc_vs_cancer.png` | Move | Completed | |
| `race_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/race_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `race_cancer_rates.csv` | `outputs/eda/figures/race_vs_cancer.png` | Move | Completed | |
| `surgmeno_vs_cancer.png` | `EDA_data_set/BCSC_Full_Feature_Analysis/surgmeno_vs_cancer.png` | Image | Existing, generated EDA output | Feature cancer-rate plot | `surgmeno_cancer_rates.csv` | `outputs/eda/figures/surgmeno_vs_cancer.png` | Move | Completed | |
| `pca_2d_distribution.png` | `EDA_data_set/pca_2d_distribution.png` | Image | Existing, generated EDA output | PCA 2D visualization | `pca_2d_coordinates.csv` | `outputs/eda/figures/pca_2d_distribution.png` | Move | Completed | Exploratory only |
| `pca_3d_distribution.png` | `EDA_data_set/pca_3d_distribution.png` | Image | Existing, generated EDA output | PCA 3D visualization | `Visualization_2d&3d.ipynb` | `outputs/eda/figures/pca_3d_distribution.png` | Move | Completed | Exploratory only |
| `tsne_2d_distribution.png` | `EDA_data_set/tsne_2d_distribution.png` | Image | Existing, generated EDA output | t-SNE 2D visualization | `Visualization_2d&3d.ipynb` | `outputs/eda/figures/tsne_2d_distribution.png` | Move | Completed | Exploratory only |
| `Gemini_Generated_Image_9ndflj9ndflj9ndf.jpeg` | `images/Gemini_Generated_Image_9ndflj9ndflj9ndf.jpeg` | Image | Existing, purpose not verified | Image file present; semantic purpose not confirmed | Not Verified | `outputs/eda/figures/Gemini_Generated_Image_9ndflj9ndflj9ndf.jpeg` | Move | Completed | Preserved; purpose remains unverified |
| `Screenshot 2026-11-02 202443.png` | `images/Screenshot 2026-11-02 202443.png` | Image | Existing, purpose not verified | Screenshot file present; semantic purpose not confirmed | Not Verified | `outputs/eda/figures/Screenshot 2026-11-02 202443.png` | Move | Completed | Preserved; purpose remains unverified |
| `Strength of association with cancer.png` | `images/Strength of association with cancer.png` | Image | Existing, purpose not verified | Image file present; semantic purpose not confirmed | Not Verified | `outputs/eda/figures/Strength of association with cancer.png` | Move | Completed | Preserved; purpose remains unverified |

## Project progress matrix

| Research stage | Current status | Existing evidence | Pending work |
|---|---|---|---|
| Data Acquisition | Existing source preserved | `data/01_raw/risk.txt` and CSV copies | Verify external provenance and citation |
| Data Conversion | Completed | `src/data/risk_to_csv.py` and generated CSV | Independent reproducibility check not performed |
| Data Discovery | Existing exploratory work | Dataset headers, counts, and `EDA.ipynb` | Formal data dictionary/codebook |
| Exploratory Data Analysis | Completed exploratory stage | `EDA.ipynb`, tables/plots, weighted cancer rates | Confirm category semantics and statistical scope |
| Feature Association Analysis | Completed exploratory stage | Association notebook and summary/category outputs | Multiple-testing and inferential interpretation review |
| Missingness Analysis | Partially observed only | Existing notebooks exclude/handle unknown values in analyses | Dedicated missingness report |
| Dimensionality Reduction | Completed exploratory visualization | PCA 2D/3D and t-SNE outputs | Validate purpose and document reproducibility details |
| Feature Selection | Decision recorded, not model-validated | Feature exclusions and association ranking | Model-based selection and validation |
| Data Preprocessing | Partial/intermediate filtering only | `drop.ipynb` and `risk_filtered.csv` | Reproducible final preprocessing pipeline |
| Model Training | Not evidenced | No model files or experiment outputs found | All model experiments |
| Model Validation | Not evidenced | No validation metrics or reports found | Validation protocol and results |
| Final Evaluation | Not evidenced | No final metrics, figures, or reports found | Final evaluation and clinical interpretation |

## Research decisions recorded from existing assets

- Target variable: `cancer`.
- Frequency weight: `count`; it is not a predictive feature.
- Existing filtering notebook removes `race`, `Hispanic`, and `invasive` from
  the filtered dataset while retaining `training` and `count`.
- The association notebook excludes `cancer`, `invasive`, `training`, and
  `count` from its automatically detected feature list.
- The project context identifies `invasive` as target-related leakage and
  `race`/`Hispanic` as variables that may be retained for subgroup or fairness
  analysis rather than the primary feature set.
- The existing association notebook analyzes 12 available features, including
  `race` and `Hispanic`; this is exploratory analysis and is not evidence of a
  final predictive feature set.
- The data are aggregated and weighted analyses use `count`.
- A `training` column exists in the source header and is summarized in EDA;
  no independently verified train/test modeling workflow was found.

## Existing assets

The inventory above records all 43 original files. They include one source TXT
file, three raw CSV copies, two interim CSV copies, one conversion script, four
notebooks, one PCA coordinate table, twelve association tables, fifteen
association/visualization PNG assets, and three additional image assets
(including one JPEG) whose semantic purpose was not verified. (The exact
per-file list and hashes were captured before migration.)

## Pending work

- Verify source provenance and complete codebook.
- Establish and document a reproducible final preprocessing protocol.
- Investigate missing and unknown category codes formally.
- Decide and validate the primary predictive feature set.
- Train candidate models and perform model validation.
- Perform final evaluation and document limitations.

## Unverified information

- External dataset citation, download location, and licensing.
- Exact semantic meaning of every numeric category code.
- Whether the selected ten-variable feature list is a final research decision
  or an initial proposal.
- The purpose of the three standalone images in the original `images/` folder.
- Any model training, validation, selection, or final evaluation.

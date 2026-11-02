# BCSC Risk Estimation

## Objective

Research organization for analysis of the Breast Cancer Surveillance Consortium
(BCSC) risk estimation dataset. The project investigates clinical and
mammography-related factors associated with breast cancer and preserves existing
exploratory work for future modeling.

This project is research and clinical decision-support work. It is not a
diagnostic system and does not replace professional medical judgment.

## Dataset

The source dataset is an aggregated BCSC dataset originally provided as
`risk.txt` and converted to CSV. The `count` column is a frequency weight; each
row represents a group of examinations with shared characteristics and outcome.

## Current workflow

- Raw data acquisition and TXT-to-CSV conversion are preserved.
- Duplicate-row filtering and an interim filtered dataset are preserved.
- Exploratory data analysis is preserved.
- Feature-to-cancer association analysis is preserved.
- PCA and t-SNE visualization outputs are preserved.
- Model training, model validation, and final evaluation are not evidenced by
  the inspected assets and remain empty.

## Structure

- `data/`: raw and interim datasets; processed and split stages are currently empty.
- `notebooks/`: preserved research notebooks organized by stage.
- `src/data/`: the existing TXT-to-CSV script.
- `outputs/eda/`: existing tables and figures.
- `docs/`: project state and evidence-based documentation.
- `models/`, `tests/`, and future-stage output folders: currently empty unless
  populated by existing assets.

## Installation

Dependency versions were not verified from project files. `requirements.txt` is
therefore intentionally empty.

## Traceability

See [docs/PROJECT_STATE_MATRIX.md](docs/PROJECT_STATE_MATRIX.md) for the full
inventory, migration map, verified status, and pending work.

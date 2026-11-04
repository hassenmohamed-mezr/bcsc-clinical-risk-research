# Limitations and Roadmap

## Current Limitations

- Research/proof-of-concept scope; not clinically validated.
- Low prevalence (~0.49%) complicates metric interpretation.
- `surgmeno` and `hrt` code 9 combines Unknown and Not Applicable.
- No true-null contract for future missing values.
- Dependency versions are not pinned.
- Test split was used only for one authorized final evaluation.
- No API/service layer.

## Known Technical Issues

- Weak validation of direct BCSC-coded transforms (undocumented nominal codes may become all-zero in one-hot).
- No training-data fingerprint in preprocessing artifacts.
- Ordinal encoding implies equal spacing.
- Artifact unpickling requires repository import paths.

## Unverified Areas

- Whether `surgmeno`/`hrt` code 9 with menopausal status is Unknown vs Not Applicable.
- Notebook execution environments and raw CSV regeneration.
- EDA influence on feature exclusion decisions.

## Future Directions

- Collect and prepare real local data.
- Improve calibration and threshold analysis.
- Add service/API layer.
- Add external validation.
- Add strict domain validation for BCSC-coded transforms.
- Add train-data fingerprint to artifacts.

These are planned, not implemented.
# Missing-Value Treatment Options

All features are categorical; nulls = 0. Code 9 = **Unknown** for menopaus, bmi, agefirst, nrelbc, density, brstproc, lastmamm; **Unknown/NA** for surgmeno/hrt; `agegrp` 9/10 are valid. No strategy selected.

| Option | Applies to | Advantages | Risks | Meaning / type |
|---|---|---|---|---|
| Retain unknown category | all code-9 features | keeps information; no fabricated values | sparse level (menopaus 7.6%); interpretation needs care | Preserved; categorical ✓ |
| Mode imputation | any | simple | hides informative missingness; distorts weighted rates; invents certainty | Not preserved ✗ |
| Missing indicator | code-9 features | models missingness explicitly | duplicates information; collinearity if combined with retention | Partial ✓ |
| Model-native handling | tree models | uses categorical Unknown directly | linear models still need encoding; requires 9→NA mapping | Preserved if coded NA ✓ |
| Recode 9→NA; conditional rules | surgmeno, hrt | could separate NA when menopausal status known | unknown vs not-applicable conflated | Requires evidence; UNVERIFIED |

Highest unknown shares: bmi 55.8%, agefirst 55.5%, surgmeno 52.2%, hrt 41.0%, density 27.3%, lastmamm 23.3%; mode imputation is especially risky there. `agegrp` needs no treatment. Unresolved: NA separation for surgmeno/hrt, missingness mechanism, and effect on weighted estimates.

Last updated: 2026-11-03 | git HEAD: 40e7c4f

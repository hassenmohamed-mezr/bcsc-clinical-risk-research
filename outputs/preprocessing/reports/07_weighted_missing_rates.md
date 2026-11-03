# Weighted Missing Rates

Development data: `train.csv` + `val.csv` only (test not opened). Total rows 180,465; **total frequency weight 1,795,139**. Null weighted count = sum of `count` over null rows.

| Feature | Null rows | Null weight | Row % | Weighted % |
|---|---|---|---|---|
| agegrp | 0 | 0 | 0.00% | 0.00% |
| bmi | 0 | 0 | 0.00% | 0.00% |
| agefirst | 0 | 0 | 0.00% | 0.00% |
| menopaus | 0 | 0 | 0.00% | 0.00% |
| surgmeno | 0 | 0 | 0.00% | 0.00% |
| hrt | 0 | 0 | 0.00% | 0.00% |
| nrelbc | 0 | 0 | 0.00% | 0.00% |
| density | 0 | 0 | 0.00% | 0.00% |
| brstproc | 0 | 0 | 0.00% | 0.00% |
| lastmamm | 0 | 0 | 0.00% | 0.00% |

Denominators are consistent: non-null weight + null weight equals 1,795,139 for every feature. Encoded special values (code 9 in nine features; `agegrp=10`) are **excluded** from these rates until the BCSC codebook verifies their meaning; zero is not treated as missing.

Last updated: 2026-11-03 | git HEAD: 40e7c4f

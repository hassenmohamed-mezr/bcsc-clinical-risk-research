# Missing-Value Inventory

Train+val only (180,465 rows; weight 1,795,139); test not opened. All features are int64 with **zero actual nulls**. Frequencies are rows (share of total frequency weight).

| Feature | Nulls | Special codes | Notes |
|---|---|---|---|
| agegrp | 0 | 10: 10,027 | 6.1% weight; range 1–10; top-code meaning unverified |
| bmi | 0 | 9: 69,277 | 55.8%; likely unknown (unverified) |
| agefirst | 0 | 9: 57,402 | 55.5%; 0 also present, meaning unverified |
| menopaus | 0 | 9: 9,999 | 7.6% |
| surgmeno | 0 | 9: 77,518 | 52.2% |
| hrt | 0 | 9: 60,309 | 41.0% |
| nrelbc | 0 | 9: 26,202 | 15.2%; 0 meaning unverified |
| density | 0 | 9: 45,359 | 27.3% |
| brstproc | 0 | 9: 23,891 | 10.5% |
| lastmamm | 0 | 9: 60,455 | 23.3% |

**Flags:** code 9 is widespread and not verified as "missing"; `agegrp=10` is unusual; `agefirst`/`nrelbc` mix 0 and 9. No semantics assumed; no rows dropped or imputed.

Last updated: 2026-11-03 | git HEAD: 40e7c4f

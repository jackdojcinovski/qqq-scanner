# Strategy study 2026-10-27

- Symbols: QQQ, SPY; at most 1 trade a day across them (first signal wins)
- Tune 2020-07-27 to 2025-10-24 (1319 days); check 2025-10-27 to 2026-10-26 (251 days)
- R = profit / distance to the stop at entry. $/contract = 100 x delta x move - option costs (deep ITM monthly: delta 0.8, $0.10 spread, small decay). Weekly and 1-2 DTE shown in R for comparison.
- Strategies: **noise** = noise-boundary momentum (Zarattini et al. 2024), **lasthalf** = last-half-hour momentum (Gao et al. 2018), **vwap** = VWAP trend. See scanner/intraday.py.

## Every variant, tune period (ranked by the 90% lower bound, deep ITM)

| strategy | trades | per week | win | avg win | avg loss | gross R | net R deep ITM (90% CI) | net R weekly | net R 1-2 DTE | $/contract deep ITM | $/week (1 contract) | max DD $ | avg hold | years + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `noise|both|band1|vwap` | 871 | 3.30 | 42% | +1.95 | -1.31 | +0.16 | -0.02 (-0.15 to +0.13) | -0.18 | -0.65 | $-2 | $-5 | $5,164 | 2.9h | 3/6 |
| `noise|long|band1|vwap` | 489 | 1.85 | 46% | +1.72 | -1.26 | +0.18 | -0.01 (-0.17 to +0.16) | -0.19 | -0.70 | $-7 | $-13 | $6,255 | 3.1h | 2/6 |
| `noise|both|band1.5|vwap` | 613 | 2.32 | 43% | +1.54 | -1.17 | +0.11 | -0.04 (-0.17 to +0.09) | -0.19 | -0.61 | $+2 | $+4 | $3,846 | 3.0h | 3/6 |
| `vwap|long|from90` | 761 | 2.88 | 41% | +1.87 | -1.17 | +0.15 | -0.05 (-0.17 to +0.06) | -0.25 | -0.80 | $-4 | $-12 | $4,172 | 3.1h | 2/6 |
| `noise|long|band1.5|band_vwap` | 315 | 1.19 | 41% | +2.47 | -1.50 | +0.38 | +0.07 (-0.19 to +0.33) | -0.11 | -0.67 | $+2 | $+3 | $2,734 | 1.9h | 3/6 |
| `vwap|long|from60` | 808 | 3.06 | 39% | +1.88 | -1.15 | +0.10 | -0.10 (-0.21 to +0.01) | -0.30 | -0.86 | $-9 | $-27 | $7,376 | 3.1h | 0/6 |
| `noise|long|band1.5|vwap` | 315 | 1.19 | 48% | +1.39 | -1.21 | +0.13 | -0.04 (-0.21 to +0.12) | -0.21 | -0.69 | $-0 | $-0 | $3,132 | 3.2h | 2/6 |
| `vwap|both|from90` | 1247 | 4.73 | 36% | +1.99 | -1.10 | +0.07 | -0.12 (-0.22 to -0.03) | -0.30 | -0.81 | $-11 | $-54 | $16,480 | 3.0h | 0/6 |
| `vwap|both|from60` | 1257 | 4.76 | 34% | +2.05 | -1.09 | +0.06 | -0.13 (-0.22 to -0.04) | -0.32 | -0.84 | $-12 | $-59 | $17,268 | 3.0h | 0/6 |
| `lasthalf|both|day|noise0` | 1294 | 4.91 | 45% | +1.03 | -1.07 | -0.01 | -0.19 (-0.26 to -0.13) | -0.21 | -0.28 | $-11 | $-56 | $14,785 | 0.5h | 1/6 |
| `noise|both|band1.5|band_vwap` | 613 | 2.32 | 39% | +2.61 | -1.71 | +0.22 | -0.06 (-0.28 to +0.16) | -0.23 | -0.72 | $-0 | $-0 | $5,048 | 1.9h | 2/6 |
| `lasthalf|long|day|noise0` | 801 | 3.04 | 44% | +0.88 | -0.96 | -0.05 | -0.23 (-0.31 to -0.16) | -0.25 | -0.33 | $-13 | $-41 | $10,752 | 0.5h | 0/6 |
| `lasthalf|both|day|noise1` | 770 | 2.92 | 44% | +1.06 | -1.12 | -0.03 | -0.22 (-0.32 to -0.13) | -0.23 | -0.32 | $-14 | $-41 | $10,981 | 0.5h | 0/6 |
| `lasthalf|both|first30|noise0` | 1294 | 4.91 | 43% | +1.01 | -1.08 | -0.06 | -0.25 (-0.32 to -0.18) | -0.26 | -0.33 | $-15 | $-74 | $19,796 | 0.5h | 0/6 |
| `noise|long|band1|band_vwap` | 489 | 1.85 | 39% | +2.60 | -1.69 | +0.18 | -0.13 (-0.34 to +0.10) | -0.33 | -0.91 | $-3 | $-6 | $4,232 | 2.1h | 2/6 |
| `lasthalf|both|first30|noise1` | 1013 | 3.84 | 44% | +1.01 | -1.14 | -0.07 | -0.26 (-0.35 to -0.18) | -0.27 | -0.35 | $-15 | $-59 | $16,123 | 0.5h | 0/6 |
| `lasthalf|long|first30|noise0` | 817 | 3.10 | 43% | +0.95 | -1.04 | -0.07 | -0.25 (-0.35 to -0.16) | -0.27 | -0.35 | $-14 | $-43 | $11,860 | 0.5h | 1/6 |
| `noise|both|band1|band_vwap` | 871 | 3.30 | 36% | +2.96 | -1.78 | +0.13 | -0.16 (-0.36 to +0.04) | -0.34 | -0.88 | $-2 | $-8 | $4,526 | 2.1h | 1/6 |
| `lasthalf|long|day|noise1` | 438 | 1.66 | 41% | +0.85 | -0.94 | -0.07 | -0.26 (-0.37 to -0.15) | -0.27 | -0.35 | $-14 | $-23 | $6,215 | 0.5h | 1/6 |
| `lasthalf|long|first30|noise1` | 595 | 2.26 | 43% | +0.88 | -1.06 | -0.11 | -0.30 (-0.40 to -0.20) | -0.31 | -0.39 | $-16 | $-37 | $10,140 | 0.5h | 1/6 |

## Shortlist: tune vs CHECK (unseen year)

| strategy | trades | per week | win | avg win | avg loss | gross R | net R deep ITM (90% CI) | net R weekly | net R 1-2 DTE | $/contract deep ITM | $/week (1 contract) | max DD $ | avg hold | years + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `noise|both|band1|vwap` tune | 871 | 3.30 | 42% | +1.95 | -1.31 | +0.16 | -0.02 (-0.15 to +0.13) | -0.18 | -0.65 | $-2 | $-5 | $5,164 | 2.9h | 3/6 |
| **`noise|both|band1|vwap` check** | 176 | 3.51 | 36% | +1.96 | -1.14 | +0.04 | -0.08 (-0.35 to +0.20) | -0.26 | -0.74 | $-35 | $-122 | $8,286 | 2.8h | 1/2 |
| `noise|long|band1|vwap` tune | 489 | 1.85 | 46% | +1.72 | -1.26 | +0.18 | -0.01 (-0.17 to +0.16) | -0.19 | -0.70 | $-7 | $-13 | $6,255 | 3.1h | 2/6 |
| **`noise|long|band1|vwap` check** | 96 | 1.91 | 35% | +2.17 | -1.16 | +0.02 | -0.13 (-0.50 to +0.28) | -0.34 | -0.93 | $-57 | $-109 | $6,122 | 3.1h | 0/2 |
| `noise|both|band1.5|vwap` tune | 613 | 2.32 | 43% | +1.54 | -1.17 | +0.11 | -0.04 (-0.17 to +0.09) | -0.19 | -0.61 | $+2 | $+4 | $3,846 | 3.0h | 3/6 |
| **`noise|both|band1.5|vwap` check** | 115 | 2.29 | 35% | +1.57 | -0.94 | -0.03 | -0.14 (-0.39 to +0.12) | -0.29 | -0.72 | $-41 | $-95 | $6,021 | 2.8h | 0/2 |
| `vwap|long|from90` tune | 761 | 2.88 | 41% | +1.87 | -1.17 | +0.15 | -0.05 (-0.17 to +0.06) | -0.25 | -0.80 | $-4 | $-12 | $4,172 | 3.1h | 2/6 |
| **`vwap|long|from90` check** | 145 | 2.89 | 43% | +1.35 | -1.08 | +0.06 | -0.09 (-0.31 to +0.12) | -0.32 | -0.95 | $-15 | $-44 | $3,052 | 3.4h | 1/2 |
| `noise|long|band1.5|band_vwap` tune | 315 | 1.19 | 41% | +2.47 | -1.50 | +0.38 | +0.07 (-0.19 to +0.33) | -0.11 | -0.67 | $+2 | $+3 | $2,734 | 1.9h | 3/6 |
| **`noise|long|band1.5|band_vwap` check** | 56 | 1.12 | 32% | +1.77 | -1.34 | -0.17 | -0.38 (-0.85 to +0.12) | -0.55 | -1.03 | $-18 | $-20 | $2,089 | 1.5h | 0/2 |
| `vwap|long|from60` tune | 808 | 3.06 | 39% | +1.88 | -1.15 | +0.10 | -0.10 (-0.21 to +0.01) | -0.30 | -0.86 | $-9 | $-27 | $7,376 | 3.1h | 0/6 |
| **`vwap|long|from60` check** | 151 | 3.01 | 39% | +1.46 | -1.07 | +0.04 | -0.11 (-0.31 to +0.11) | -0.33 | -0.94 | $-24 | $-71 | $4,621 | 3.3h | 1/2 |

## Combined: best of each strategy, 1 trade a day

Uses: `noise|long|band1.5|band_vwap`

| strategy | trades | per week | win | avg win | avg loss | gross R | net R deep ITM (90% CI) | net R weekly | net R 1-2 DTE | $/contract deep ITM | $/week (1 contract) | max DD $ | avg hold | years + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combined tune | 315 | 1.19 | 41% | +2.47 | -1.50 | +0.38 | +0.07 (-0.19 to +0.33) | -0.11 | -0.67 | $+2 | $+3 | $2,734 | 1.9h | 3/6 |
| **combined check** | 56 | 1.12 | 32% | +1.77 | -1.34 | -0.17 | -0.38 (-0.85 to +0.12) | -0.55 | -1.03 | $-18 | $-20 | $2,089 | 1.5h | 0/2 |

| year | trades | net R deep ITM | $ total, 1 contract |
|---|---|---|---|
| 2020 | 20 | -0.08 | $-72 |
| 2021 | 54 | +0.51 | $+1,032 |
| 2022 | 57 | +0.66 | $+1,590 |
| 2023 | 77 | -0.29 | $-584 |
| 2024 | 66 | +0.18 | $+224 |
| 2025 | 47 | -0.74 | $-2,186 |
| 2026 | 50 | -0.33 | $-313 |

Which strategy took the day: noise 371

## Bottom line

- Best single variant on tune: `noise|both|band1|vwap`: 3.30/week, win 42%, -0.02R net, about $-2 per deep ITM contract.
- On the unseen check year: 176 trades, 3.51/week, win 36%, -0.08R net (90% CI -0.35 to +0.20), $-122/week on 1 contract.
- NOT confirmed on unseen data.
- Combined check: 56 trades (1.12/week), win 32%, -0.38R net, $-20/week on 1 contract.

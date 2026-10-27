# Quant grade 2026-10-27

- Variant `mid|with|long|classic|fixedeod|checklist` on QQQ, SPY; tune 2020-07-27 to 2025-10-24, check 2025-10-27 to 2026-10-26
- Keep the top 25% of tune trades by grade, at most one trade a day across symbols

## Signs on the tune period (R before costs)

| sign | with: trades | with: win | with: avg R | without: trades | without: win | without: avg R | edge | used |
|---|---|---|---|---|---|---|---|---|
| tight opening range (small risk vs daily ATR) | 549 | 33% | +0.37 | 66 | 18% | -0.48 | +0.85 | yes |
| volatility below its 1-year median | 359 | 31% | +0.32 | 256 | 32% | +0.22 | +0.10 | yes |
| yesterday closed in the trade direction | 421 | 31% | +0.28 | 194 | 31% | +0.28 | -0.01 |  |
| opening volume 1.2x its usual | 151 | 32% | +0.26 | 464 | 31% | +0.29 | -0.03 |  |
| opened with a gap in the trade direction | 428 | 32% | +0.26 | 187 | 29% | +0.33 | -0.07 |  |
| first 5 minutes moved more than normal | 303 | 33% | +0.22 | 312 | 29% | +0.34 | -0.13 |  |
| daily trend agrees (vs 20-day average) | 434 | 30% | +0.19 | 181 | 34% | +0.50 | -0.31 |  |
| QQQ and SPY agree | 487 | 31% | +0.20 | 128 | 33% | +0.57 | -0.37 |  |

Grade = number of the 2 used signs present. Trade when grade >= **2**.

## Results by grade (shares costs)

| grade | tune trades | tune win | tune net R | check trades | check win | check net R |
|---|---|---|---|---|---|---|
| 2 | 314 | 32% | +0.39 | 40 | 35% | +0.33 |
| 1 | 280 | 32% | +0.22 | 72 | 33% | +0.09 |
| 0 | 21 | 5% | -0.96 | 11 | 73% | +1.15 |

## Trading it with shares

| set | trades | per week | win | avg R | net R | 90% CI net | PF net | max DD | streak | years + |
|---|---|---|---|---|---|---|---|---|---|---|
| tune: every trade, 1/day | 420 | 1.60 | 31% | +0.32 | +0.31 | +0.12 to +0.51 | 1.43 | 13.2R | 9 | 5/6 |
| tune: grade >= 2 | 231 | 0.88 | 32% | +0.44 | +0.43 | +0.15 to +0.73 | 1.60 | 15.2R | 13 | 5/6 |
| **check: every trade, 1/day** | 84 | 1.67 | 36% | +0.13 | +0.12 | -0.22 to +0.49 | 1.19 | 12.0R | 8 | 1/2 |
| **check: grade >= 2** | 34 | 0.68 | 26% | -0.04 | -0.05 | -0.59 to +0.62 | 0.93 | 13.2R | 11 | 1/2 |

## Trading it with TQQQ / 3x ETF

| set | trades | per week | win | avg R | net R | 90% CI net | PF net | max DD | streak | years + |
|---|---|---|---|---|---|---|---|---|---|---|
| tune: every trade, 1/day | 420 | 1.60 | 31% | +0.32 | +0.29 | +0.10 to +0.49 | 1.40 | 13.4R | 9 | 5/6 |
| tune: grade >= 2 | 231 | 0.88 | 32% | +0.44 | +0.41 | +0.13 to +0.71 | 1.57 | 15.5R | 13 | 5/6 |
| **check: every trade, 1/day** | 84 | 1.67 | 36% | +0.13 | +0.12 | -0.23 to +0.49 | 1.18 | 12.2R | 8 | 1/2 |
| **check: grade >= 2** | 34 | 0.68 | 26% | -0.04 | -0.06 | -0.60 to +0.61 | 0.92 | 13.5R | 11 | 1/2 |

## Trading it with deep ITM monthly call

| set | trades | per week | win | avg R | net R | 90% CI net | PF net | max DD | streak | years + |
|---|---|---|---|---|---|---|---|---|---|---|
| tune: every trade, 1/day | 420 | 1.60 | 30% | +0.32 | +0.10 | -0.08 to +0.30 | 1.12 | 19.6R | 9 | 4/6 |
| tune: grade >= 2 | 231 | 0.88 | 31% | +0.44 | +0.20 | -0.08 to +0.49 | 1.23 | 22.5R | 13 | 4/6 |
| **check: every trade, 1/day** | 84 | 1.67 | 36% | +0.13 | -0.00 | -0.34 to +0.37 | 0.99 | 14.7R | 8 | 1/2 |
| **check: grade >= 2** | 34 | 0.68 | 26% | -0.04 | -0.23 | -0.76 to +0.43 | 0.73 | 17.7R | 11 | 0/2 |

## Trading it with options weekly

| set | trades | per week | win | avg R | net R | 90% CI net | PF net | max DD | streak | years + |
|---|---|---|---|---|---|---|---|---|---|---|
| tune: every trade, 1/day | 420 | 1.60 | 29% | +0.32 | -0.07 | -0.24 to +0.11 | 0.92 | 43.3R | 9 | 2/6 |
| tune: grade >= 2 | 231 | 0.88 | 30% | +0.44 | -0.00 | -0.26 to +0.27 | 1.00 | 39.5R | 13 | 3/6 |
| **check: every trade, 1/day** | 84 | 1.67 | 35% | +0.13 | -0.14 | -0.47 to +0.21 | 0.81 | 19.5R | 8 | 1/2 |
| **check: grade >= 2** | 34 | 0.68 | 24% | -0.04 | -0.42 | -0.91 to +0.20 | 0.54 | 22.3R | 11 | 0/2 |

## Trading it with options 1-2 DTE

| set | trades | per week | win | avg R | net R | 90% CI net | PF net | max DD | streak | years + |
|---|---|---|---|---|---|---|---|---|---|---|
| tune: every trade, 1/day | 420 | 1.60 | 21% | +0.32 | -0.56 | -0.70 to -0.41 | 0.47 | 234.7R | 15 | 0/6 |
| tune: grade >= 2 | 231 | 0.88 | 22% | +0.44 | -0.57 | -0.79 to -0.34 | 0.49 | 132.7R | 18 | 0/6 |
| **check: every trade, 1/day** | 84 | 1.67 | 23% | +0.13 | -0.53 | -0.81 to -0.22 | 0.41 | 47.1R | 9 | 0/2 |
| **check: grade >= 2** | 34 | 0.68 | 9% | -0.04 | -0.96 | -1.38 to -0.39 | 0.23 | 35.1R | 13 | 0/2 |

## Verdict (shares)

- Graded check trades: 34, win 26%, net -0.05R (90% CI -0.59 to +0.62) vs +0.12R ungraded.
- The grade did NOT improve results on unseen data.

Signs used: volatility below its 1-year median, tight opening range (small risk vs daily ATR)

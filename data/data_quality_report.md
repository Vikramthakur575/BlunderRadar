# BlunderRadar: Data Quality & Ingestion Report

## 1. Executive Summary

- **Total Positions Ingested:** 300,000
- **Unique Games:** 4,535
- **Average Plies per Game:** 66.2
- **Overall CP Blunder Rate:** 8.75%
- **Overall Win-Prob Blunder Rate:** 8.98%

## 2. Class Balance by Rating Band

| Rating Band | Positions | CP Blunders | CP Rate (%) | Prob Blunders | Prob Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HIGH** | 110,111 | 7,406 | 6.73% | 8,148 | 7.4% |
| **LOW** | 40,519 | 5,136 | 12.68% | 4,736 | 11.69% |
| **MID** | 149,370 | 13,698 | 9.17% | 14,046 | 9.4% |


## 3. Rating Distribution (Mover)

| Metric | Value |
| :--- | :--- |
| MEAN | 1643.6 |
| STD | 365.0 |
| MIN | 800.0 |
| P25 | 1374.0 |
| MEDIAN | 1646.0 |
| P75 | 1931.0 |
| MAX | 2400.0 |


## 4. Clock & Feature Null Rates

| Column | Null Count | Null Rate (%) |
| :--- | :--- | :--- |
| `game_id` | 0 | 0.0% |
| `game_idx` | 0 | 0.0% |
| `game_date` | 0 | 0.0% |
| `game_timestamp` | 0 | 0.0% |
| `speed` | 0 | 0.0% |
| `ply` | 0 | 0.0% |
| `fen_before` | 0 | 0.0% |
| `move` | 0 | 0.0% |
| `mover_color` | 0 | 0.0% |
| `mover_rating` | 0 | 0.0% |
| `opponent_rating` | 0 | 0.0% |
| `rating_diff` | 0 | 0.0% |
| `clock_remaining_seconds` | 0 | 0.0% |
| `increment_seconds` | 0 | 0.0% |
| `eval_before` | 0 | 0.0% |
| `eval_after` | 0 | 0.0% |
| `eval_drop` | 0 | 0.0% |
| `is_blunder` | 0 | 0.0% |
| `win_prob_before` | 0 | 0.0% |
| `win_prob_after` | 0 | 0.0% |
| `win_prob_drop` | 0 | 0.0% |
| `is_blunder_prob` | 0 | 0.0% |
| `batch_id` | 0 | 0.0% |
| `rating_band` | 0 | 0.0% |


## 5. Remaining Clock Time Summary (Seconds)

| Metric | Clock Seconds |
| :--- | :--- |
| MEAN | 257.5s |
| STD | 198.5s |
| MIN | 0.0s |
| P25 | 117.0s |
| MEDIAN | 183.0s |
| P75 | 354.0s |
| MAX | 1798.0s |


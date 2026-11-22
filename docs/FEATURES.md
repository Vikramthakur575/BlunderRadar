# BlunderRadar: Feature Engineering Dictionary

This document serves as the authoritative specification, domain rationale, and leakage-safety audit for all features engineered in **Phase 2** of BlunderRadar.

---

## 1. Architectural Principles & Leakage-Safety Mandate

### The Core Problem
The objective of BlunderRadar is to **predict whether the mover is about to commit a blunder BEFORE they make their move**.
A fatal flaw in many chess ML architectures is target leakage:
- Incorporating the played move (e.g. move text, destination square, piece moved).
- Incorporating post-move engine evaluations (`eval_after`).
- Incorporating post-move centipawn drop (`eval_drop`), win-probability drop (`win_prob_drop`), or post-move board FEN.

### The Leakage-Safety Guarantee
Every feature in BlunderRadar is **strictly derived from information known prior to move execution**:
1. **Board Geometry & Tactics:** Extracted solely from `fen_before`.
2. **Context & Clock:** Derived from PGN headers, ply count, the mover's clock at the START of their turn (derived, see the correction below), and time control increment.
3. **Optional Engine Evaluation:** If enabled by configuration (`include_eval_before: true`), only `eval_before` and `win_prob_before` are provided.
4. **Verification:** BlunderRadar enforces unit test `test_strict_no_post_move_leakage`, proving that mutating or removing all post-move variables produces identical feature outputs bit-for-bit.

---

## 2. Feature Dictionary

### Category 1: Material Balance & Piece Counts
Tactical complexity and psychological pressure vary dramatically with piece counts and material imbalances.

| Feature Name | Type | Range | Description | Leakage-Safety Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| `mover_pawns` | `int` | $[0, 8]$ | Total pawns owned by the player to move. | Evaluated strictly from `fen_before` piece placement. |
| `mover_knights` | `int` | $[0, 10]$ | Total knights owned by the mover. | Evaluated strictly from `fen_before`. |
| `mover_bishops` | `int` | $[0, 10]$ | Total bishops owned by the mover. | Evaluated strictly from `fen_before`. |
| `mover_rooks` | `int` | $[0, 10]$ | Total rooks owned by the mover. | Evaluated strictly from `fen_before`. |
| `mover_queens` | `int` | $[0, 9]$ | Total queens owned by the mover. | Evaluated strictly from `fen_before`. |
| `opp_pawns` | `int` | $[0, 8]$ | Total pawns owned by the opponent. | Evaluated strictly from `fen_before`. |
| `opp_knights` | `int` | $[0, 10]$ | Total knights owned by the opponent. | Evaluated strictly from `fen_before`. |
| `opp_bishops` | `int` | $[0, 10]$ | Total bishops owned by the opponent. | Evaluated strictly from `fen_before`. |
| `opp_rooks` | `int` | $[0, 10]$ | Total rooks owned by the opponent. | Evaluated strictly from `fen_before`. |
| `opp_queens` | `int` | $[0, 9]$ | Total queens owned by the opponent. | Evaluated strictly from `fen_before`. |
| `mover_piece_count` | `int` | $[0, 15]$ | Total non-king pieces owned by the mover. | Sum of pre-move pieces. |
| `opp_piece_count` | `int` | $[0, 15]$ | Total non-king pieces owned by the opponent. | Sum of pre-move opponent pieces. |
| `total_piece_count` | `int` | $[0, 30]$ | Sum of all non-king pieces on the board. | Directly computed from pre-move board. |
| `mover_material` | `int` | $[0, 39+]$ | Total material points (P=1, N=3, B=3, R=5, Q=9). | Evaluated strictly from pre-move position. |
| `opp_material` | `int` | $[0, 39+]$ | Total material points owned by the opponent. | Evaluated strictly from pre-move position. |
| `material_diff` | `int` | $[-\infty, +\infty]$ | `mover_material - opp_material`. Indicates material advantage/deficit. | Pre-move arithmetic difference. |
| `mover_has_bishop_pair` | `int` | $\{0, 1\}$ | Indicator ($1$ if mover has $\ge 2$ bishops, else $0$). | Evaluated strictly from pre-move piece bitboard. |
| `opp_has_bishop_pair` | `int` | $\{0, 1\}$ | Indicator ($1$ if opponent has $\ge 2$ bishops, else $0$). | Evaluated strictly from pre-move piece bitboard. |

---

### Category 2: Piece Mobility
Restricted mobility strongly correlates with blunders as players struggle to find constructive moves under spatial constriction.

| Feature Name | Type | Range | Description | Leakage-Safety Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| `mover_mobility` | `int` | $[0, 218]$ | Count of legal moves available to the mover in the pre-move position. | Legal move generator on `fen_before`. |
| `opp_mobility` | `int` | $[0, 218]$ | Count of legal moves the opponent would have from the current board if turn were flipped. | Computed from pre-move position with turn temporarily reversed. |
| `mobility_diff` | `int` | $[-\infty, +\infty]$ | `mover_mobility - opp_mobility`. Measure of spatial dynamic control. | Linear combination of pre-move mobilities. |

---

### Category 3: King Safety & Castling Status
King vulnerability is a primary catalyst for tactical collapses and panic blunders.

| Feature Name | Type | Range | Description | Leakage-Safety Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| `mover_king_zone_attackers` | `int` | $[0, 16]$ | Unique enemy pieces attacking any square in the mover's king zone (king square + adjacent ring). | Attack ray bitboards on mover's pre-move king location. |
| `mover_king_zone_attacks` | `int` | $[0, 64]$ | Total number of enemy attack vectors targeting squares in the mover's king zone. | Pre-move attack rays targeting mover king zone. |
| `opp_king_zone_attackers` | `int` | $[0, 16]$ | Unique friendly pieces attacking opponent's king zone. | Pre-move attack rays targeting opponent king zone. |
| `opp_king_zone_attacks` | `int` | $[0, 64]$ | Total number of friendly attack vectors targeting squares in the opponent's king zone. | Pre-move attack rays targeting opponent king zone. |
| `mover_pawn_shield` | `int` | $[0, 6]$ | Friendly pawns within 1-2 ranks directly in front of the king on king and adjacent files. | Position of friendly pawns relative to king on `fen_before`. |
| `opp_pawn_shield` | `int` | $[0, 6]$ | Enemy pawns within 1-2 ranks directly in front of the opponent's king. | Position of enemy pawns relative to opponent king on `fen_before`. |
| `mover_can_castle_kingside` | `int` | $\{0, 1\}$ | Pre-move kingside castling right for mover. | FEN castling availability field. |
| `mover_can_castle_queenside` | `int` | $\{0, 1\}$ | Pre-move queenside castling right for mover. | FEN castling availability field. |
| `mover_has_castled` | `int` | $\{0, 1\}$ | $1$ if mover's king resides on standard castled squares (g1/c1 for White, g8/c8 for Black) and rights are gone. | Pre-move king coordinate and castling rights. |
| `opp_can_castle_kingside` | `int` | $\{0, 1\}$ | Pre-move kingside castling right for opponent. | FEN castling availability field. |
| `opp_can_castle_queenside` | `int` | $\{0, 1\}$ | Pre-move queenside castling right for opponent. | FEN castling availability field. |
| `opp_has_castled` | `int` | $\{0, 1\}$ | $1$ if opponent's king is on castled squares with rights gone. | Pre-move opponent king coordinates and rights. |

---

### Category 4: Checks & Tactical Pressure
Positions under check restrict legal candidate moves to escapes/blocks, while positions with check options tempt hasty speculative attacks.

| Feature Name | Type | Range | Description | Leakage-Safety Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| `is_in_check` | `int` | $\{0, 1\}$ | $1$ if the mover's king is currently in check before moving. | Pre-move `board.is_check()`. |
| `num_checkers` | `int` | $[0, 2]$ | Number of pieces simultaneously checking the mover's king ($0$, $1$, or double check $2$). | Pre-move `len(board.checkers())`. |
| `can_give_check` | `int` | $\{0, 1\}$ | $1$ if the mover has at least one pre-move candidate move that delivers check to opponent's king. | Evaluates whether any pre-move legal move gives check. Does NOT inspect which move was played. |

---

### Category 5: Absolute Pins
Pinned pieces are tactically paralyzed and frequently overlooked in low-time situations.

| Feature Name | Type | Range | Description | Leakage-Safety Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| `mover_pinned_pieces` | `int` | $[0, 8]$ | Count of friendly pieces pinned to the mover's king along ranks, files, or diagonals. | Evaluated via `board.is_pinned(mover_color, sq)` on `fen_before`. |
| `opp_pinned_pieces` | `int` | $[0, 8]$ | Count of opponent pieces pinned to the opponent's king. | Evaluated via `board.is_pinned(opp_color, sq)` on `fen_before`. |

---

### Category 6: Hanging & Attacked Pieces
Tactical oversight of undefended ("hanging") pieces accounts for the vast majority of club-level and intermediate blunders.

| Feature Name | Type | Range | Description | Leakage-Safety Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| `mover_attacked_pieces` | `int` | $[0, 16]$ | Total friendly pieces under attack by at least one opponent piece. | Pre-move attack rays targeting friendly pieces. |
| `mover_attacked_value` | `int` | $[0, 39+]$ | Sum of standard material values of all friendly pieces currently under attack. | Pre-move material valuation of attacked pieces. |
| `mover_hanging_pieces` | `int` | $[0, 16]$ | Friendly pieces attacked by enemy pieces with **zero** friendly defenders. | Strict pre-move attack and defense square intersection. |
| `mover_hanging_value` | `int` | $[0, 39+]$ | Sum of standard material values of all friendly hanging pieces. | Pre-move material valuation of undefended attacked pieces. |
| `opp_attacked_pieces` | `int` | $[0, 16]$ | Total opponent pieces attacked by mover. | Pre-move attacks targeting enemy pieces. |
| `opp_attacked_value` | `int` | $[0, 39+]$ | Material value of opponent pieces under attack. | Pre-move valuation of attacked enemy pieces. |
| `opp_hanging_pieces` | `int` | $[0, 16]$ | Opponent pieces attacked by mover with zero defenders (tactical opportunity). | Strict pre-move attack and defense intersection. |
| `opp_hanging_value` | `int` | $[0, 39+]$ | Material value of opponent hanging pieces. | Pre-move valuation of enemy hanging pieces. |

---

### Category 7: Board Tension & Contact
High board contact creates high cognitive load and combinational complexity.

| Feature Name | Type | Range | Description | Leakage-Safety Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| `board_tension_piece_attacks` | `int` | $[0, 32]$ | Total count of all attacks targeting pieces of either color across the board. | Sum of pre-move attacked pieces on both sides. |
| `board_tension_pawn_clashes` | `int` | $[0, 8]$ | Count of opposing pawn pairs in direct attack contact (e.g. e4 vs d5). | Pre-move pawn attack intersection. |
| `board_tension_total_attacked_value` | `int` | $[0, 78+]$ | Total material point value of all pieces currently under attack on both sides. | Sum of pre-move attacked piece values. |

---

### Category 8: Game Phase
Blunder dynamics shift drastically between opening theory, complex middlegames, and technical endgames.

| Feature Name | Type | Range | Description | Leakage-Safety Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| `game_phase_score` | `float` | $[0.0, 1.0]$ | Ratio of remaining non-pawn material (Knights: 1, Bishops: 1, Rooks: 2, Queens: 4; max = 24). $1.0$ is opening, $0.0$ is pure pawn/king endgame. | Evaluated strictly from non-pawn piece counts on `fen_before`. |
| `is_opening` | `int` | $\{0, 1\}$ | Indicator ($1$ if `game_phase_score` $\ge 0.8$, else $0$). | Threshold on pre-move `game_phase_score`. |
| `is_endgame` | `int` | $\{0, 1\}$ | Indicator ($1$ if `game_phase_score` $\le 0.3$, else $0$). | Threshold on pre-move `game_phase_score`. |

---

### Category 9: Context, Temporal & Clock Features
Player rating and remaining time are the strongest external predictors of blunder probability.

| Feature Name | Type | Range | Description | Leakage-Safety Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| `move_number` | `int` | $[1, \infty)$ | Move count: $(\text{ply} + 1) // 2$. | Computed from current ply before move. |
| `ply` | `int` | $[1, \infty)$ | Ply index from starting move. | Pre-move game step. |
| `mover_rating` | `float` | $[800, 2400]$ | Glicko-2 rating of the mover. | PGN header known before match start. |
| `opponent_rating` | `float` | $[800, 2400]$ | Glicko-2 rating of the opponent. | PGN header known before match start. |
| `rating_diff` | `float` | $[-1600, +1600]$ | `mover_rating - opponent_rating`. | Pre-match rating difference. |
| `clock_remaining_seconds` | `float` | $[0.0, \infty)$ | Seconds on the mover's clock when their turn BEGAN (`clock_before_seconds`, derived from their previous move's `[%clk]`). Imputed with default (180s) when unannotated. | Pre-move by construction; see correction below. |
| `prev_move_time_spent` | `float` | $[0.0, \infty)$ | Seconds the mover spent on their previous move (`clock_before + increment - clock_after`). `0` when unknown. | Computed from earlier moves only. |
| `is_time_trouble` | `int` | $\{0, 1\}$ | $1$ if `clock_remaining_seconds` $\le 30.0$ seconds (and clock is known). | Derived strictly from pre-move clock time. |
| `clock_missing` | `int` | $\{0, 1\}$ | $1$ if clock time was unannotated or missing. Prevents bias from clock imputation. | Missingness indicator. |
| `increment_seconds` | `float` | $[0.0, \infty)$ | Clock increment per move (e.g. 3.0s in 3+3, 0.0s in 5+0). | Extracted from PGN `TimeControl` header. |

---

### Category 10: Optional Pre-Move Engine Evaluation
Controlled by configuration flag: `features.include_eval_before`. Allows training both with and without engine assistance.

| Feature Name | Type | Range | Description | Leakage-Safety Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| `eval_before` | `float` | $[-\infty, +\infty]$ | Pre-move evaluation in centipawns from **mover's perspective**. | Evaluated on the position **before** the move is selected. Does NOT use `eval_after`. |
| `win_prob_before` | `float` | $[0.0, 1.0]$ | Logistic win probability from mover's perspective: $W = \frac{1}{1 + 10^{-\text{eval\_before} / 400}}$. | Pure monotonic transformation of `eval_before`. |

---

## 3. Summary of Dimensions

- **Chess Domain Features:** 46
- **Context & Temporal Features:** 9
- **Optional Pre-Move Engine Features:** 2
- **Total Feature Space:** 55 features (or 57 with pre-move eval enabled).
- **Target Variables (Preserved Separately for Training):** `is_blunder` (centipawn-based), `is_blunder_prob` (win-probability-based), `eval_drop`, `win_prob_drop`.



---

## Correction: clock leakage found and fixed (post-Phase 2 review)

Lichess stores `[%clk]` on each move as the mover's clock **after** that move. The first
Phase 2 version used this value directly as the "remaining clock before the move", which
leaks the time spent thinking on the very move being predicted.

Fix: `add_pre_move_clock_columns()` (in `features/context_features.py`) shifts the series so
the feature is the mover's own clock after their **previous** move, which equals their clock
when the current turn began (a player's clock does not run during the opponent's turn). A
player's first move (free on Lichess) falls back to their own post-move clock.
`tests/test_clock_leakage.py::test_current_post_move_clock_does_not_change_features` proves
that changing the post-move clock of a row no longer changes that row's features.
The live scorer supplies the real clock at the start of the turn, so there is no
train/serve skew (`tests/test_scorer_api.py::test_no_train_serve_skew`).

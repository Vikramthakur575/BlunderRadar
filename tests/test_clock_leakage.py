import numpy as np
import pandas as pd

from blunder_radar.features.context_features import (
    add_pre_move_clock_columns,
    extract_context_features_vectorized,
)


def _game(clocks_after, inc=0.0, gid="a"):
    n = len(clocks_after)
    return pd.DataFrame(
        {
            "game_id": [gid] * n,
            "ply": list(range(1, n + 1)),
            "mover_color": (["white", "black"] * n)[:n],
            "clock_remaining_seconds": clocks_after,
            "increment_seconds": [inc] * n,
        }
    )


def test_clock_before_is_previous_own_clock():
    df = add_pre_move_clock_columns(_game([600, 600, 598, 599, 595, 597]))
    assert list(df["clock_before_seconds"]) == [600, 600, 600, 600, 598, 599]


def test_first_move_uses_own_clock():
    df = add_pre_move_clock_columns(_game([590, 580]))
    assert df.loc[0, "clock_before_seconds"] == 590
    assert df.loc[1, "clock_before_seconds"] == 580


def test_prev_move_time_spent_includes_increment():
    # white: 300 -> 295 (+2 inc) on move 2 => spent 300 + 2 - 295 = 7s on that move
    df = add_pre_move_clock_columns(_game([300, 300, 295, 295, 290, 290], inc=2.0))
    assert df.loc[4, "prev_move_time_spent"] == 7.0


def test_games_do_not_bleed_into_each_other():
    a = _game([100, 100, 90, 90], gid="a")
    b = _game([500, 500, 400, 400], gid="b")
    df = add_pre_move_clock_columns(pd.concat([a, b], ignore_index=True))
    assert df.loc[4, "clock_before_seconds"] == 500  # first move of game b, not game a's clock


def test_original_row_order_and_index_preserved():
    base = pd.concat([_game([10, 10, 9, 9], gid="z"), _game([20, 20, 19, 19], gid="a")],
                     ignore_index=True)
    out = add_pre_move_clock_columns(base)
    assert list(out.index) == list(base.index)
    assert list(out["game_id"]) == list(base["game_id"])


def test_current_post_move_clock_does_not_change_features():
    """The clock AFTER the move is post-move information and must not affect the features."""
    base = _game([300, 300, 280, 285, 270, 270])
    f1 = extract_context_features_vectorized(add_pre_move_clock_columns(base))
    mutated = base.copy()
    mutated.loc[4, "clock_remaining_seconds"] = 5.0  # pretend the mover then burned their clock
    f2 = extract_context_features_vectorized(add_pre_move_clock_columns(mutated))
    pd.testing.assert_series_equal(f1.loc[4], f2.loc[4])


def test_time_trouble_flag_uses_pre_move_clock():
    base = _game([300, 300, 25, 300, 20, 300])
    ctx = extract_context_features_vectorized(add_pre_move_clock_columns(base))
    # at ply 3 the mover still had 300s when their turn began; at ply 5 they had 25s
    assert ctx.loc[2, "is_time_trouble"] == 0
    assert ctx.loc[4, "is_time_trouble"] == 1


def test_missing_clock_is_flagged_not_nan():
    base = _game([300, np.nan, 280, np.nan])
    ctx = extract_context_features_vectorized(add_pre_move_clock_columns(base))
    assert ctx["clock_remaining_seconds"].notna().all()
    assert ctx["clock_missing"].sum() >= 1

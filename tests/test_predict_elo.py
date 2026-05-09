import pandas as pd

from predict import _compute_current_elo


def test_compute_current_elo_uses_latest_row():
    fights = pd.DataFrame(
        [
            {
                "event_date": "2020-01-01",
                "fighter_a": "Fighter A",
                "fighter_b": "Fighter B",
                "elo_a": 1500.0,
                "elo_b": 1500.0,
            },
            {
                "event_date": "2021-01-01",
                "fighter_a": "Fighter C",
                "fighter_b": "Fighter A",
                "elo_a": 1520.0,
                "elo_b": 1610.0,
            },
        ]
    )

    assert _compute_current_elo(fights, "Fighter A") == 1610.0
    assert _compute_current_elo(fights, "Fighter B") == 1500.0

"""The score is personal, so it has to be defensible."""

from __future__ import annotations

import pytest

from demo.evals.agreement import compare
from demo.simulator.incident import ONSET


def test_the_same_call_scores_full_marks():
    assert compare("roll_back", [(200, "roll_back")]).agreement == 1.0


def test_a_near_neighbour_gets_partial_credit():
    # Rolling back when the table said "turn it off" is a different decision,
    # not the opposite one. Zero would teach the table nothing.
    score = compare("disable_feature", [(200, "roll_back")])
    assert 0 < score.agreement < 1
    assert "close" in score.agreed_on


def test_an_unrelated_call_scores_zero():
    assert compare("roll_back", [(200, "double_capacity")]).agreement == 0.0


def test_an_agent_that_does_nothing_scores_zero():
    assert compare("roll_back", []).agreement == 0.0


def test_the_outcome_numbers_travel_with_the_score():
    score = compare("roll_back", [(200, "roll_back")])
    assert score.minutes_down > 0
    assert score.customers_hit > 0
    assert score.resolved == 1


def test_every_number_is_a_feedback_score():
    # LangSmith cannot plot a value that lives only in metadata, so anything the
    # dashboard shows has to leave here as feedback.
    keys = compare("roll_back", [(200, "roll_back")]).as_feedback()
    assert set(keys) == {"agreement", "minutes_down", "customers_hit",
                         "unreconciled_payments", "people_woken", "resolved"}
    assert all(isinstance(v, float) for v in keys.values())


def test_doing_nothing_is_never_resolved():
    assert compare("roll_back", [(ONSET, "nothing")]).resolved == 0


def test_an_unknown_vote_is_refused():
    with pytest.raises(ValueError):
        compare("panic", [(200, "roll_back")])

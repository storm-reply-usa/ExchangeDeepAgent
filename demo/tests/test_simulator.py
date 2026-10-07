"""The simulator has one job beyond being plausible: being the same for everyone."""

from __future__ import annotations

import itertools

import pytest

from demo.simulator.incident import CAUSE, DECOY, ONSET
from demo.simulator.world import ACTIONS, WINDOW_END, run

EVERY_PLAN = [
    (minute, key)
    for key, minute in itertools.product(sorted(ACTIONS), range(ONSET, WINDOW_END, 7))
]


def test_the_same_plan_always_gives_the_same_outcome():
    for plan in EVERY_PLAN:
        first, second = run(plan), run(plan)
        assert first == second, plan


def test_two_tables_running_the_same_plan_can_compare_their_numbers():
    # This is the claim a speaker makes out loud, so it gets a test.
    table_a = run([(200, "disable_feature"), (206, "roll_back")])
    table_b = run([(206, "roll_back"), (200, "disable_feature")])
    assert table_a == table_b


def test_the_decoy_ships_after_the_errors_begin():
    # A release that happened after the failure started cannot have caused it.
    # An agent that blames this one has had a bad run, and the fixture must
    # never drift into making that blame reasonable.
    assert DECOY.at > ONSET
    assert DECOY.at - ONSET >= 20
    assert CAUSE.at < ONSET
    assert ONSET - CAUSE.at <= 5


def test_doing_nothing_never_resolves_and_costs_the_most():
    idle = run((ONSET, "nothing"))
    assert idle.resolved_at is None
    for key in ("roll_back", "wake_owner"):
        assert run((200, key)).customers_hit < idle.customers_hit


def test_acting_sooner_is_never_worse():
    for key, action in ACTIONS.items():
        if not action.fixes:
            continue
        early, late = run((196, key)), run((216, key))
        assert early.minutes_down <= late.minutes_down, key
        assert early.orphaned_payments <= late.orphaned_payments, key


def test_turning_the_feature_off_protects_the_money_and_costs_time():
    # The trade-off the doctrine is meant to expose. If this ever stops being
    # true, the doctrine stops being visible in the numbers.
    # Same moment of decision. One reopens the shop as fast as it can; the other
    # stops the money going astray first and reaches the rollback later for it.
    straight = run((200, "roll_back"))
    careful = run([(200, "disable_feature"), (208, "roll_back")])
    assert careful.orphaned_payments < straight.orphaned_payments
    assert careful.minutes_down > straight.minutes_down


def test_doubling_capacity_makes_it_worse():
    assert run((200, "double_capacity")).customers_hit > run((200, "nothing")).customers_hit


def test_waking_someone_is_the_only_plan_that_costs_a_person():
    for key in ACTIONS:
        assert run((200, key)).people_woken == (1 if key == "wake_owner" else 0)


@pytest.mark.parametrize("key", sorted(ACTIONS))
def test_every_action_produces_a_full_curve(key):
    curve = run((200, key)).curve
    assert curve[0][0] == ONSET - 10
    assert curve[-1][0] == WINDOW_END
    assert all(0.0 <= rate <= 1.0 for _, rate in curve)


def test_two_different_actions_landing_together_do_not_crash():
    """Two tool calls in one message run in a thread pool and can order two
    actions on the same minute. Action is frozen but unordered, so sorting the
    pair used to fall through to comparing two Actions and raise."""
    outcome = run([(200, "roll_back"), (206, "throttle")])  # both land at 212
    assert outcome.minutes_down > 0


def test_the_cost_so_far_is_not_the_cost_of_the_whole_night():
    # Read mid-incident, the integrated totals report the hours still to come -
    # and an agent told never to invent a number reads that out as fact.
    outcome = run((198, "nothing"))
    early = outcome.so_far(210)
    assert early["customers_hit"] < outcome.customers_hit
    assert early["minutes_down"] < outcome.minutes_down
    assert outcome.so_far(WINDOW_END)["customers_hit"] == pytest.approx(
        outcome.customers_hit, rel=0.02
    )


#: The minute the dossier reports on, as `forecast` reads it.
READS = 240


def test_the_cure_lands_inside_the_run_and_the_state_says_so():
    """The room has to watch checkout come back, not be told it will.

    The rollback takes time to land, and it has to land before the dossier is
    read or nobody sees it work. `READS` is the minute the page reports on,
    so that is the deadline the cure is held to - not the minute after the
    decision, which would only hold for a release that reverses instantly.
    """
    good = run((211, "roll_back")).so_far(READS)
    assert good["checkout_working"] is True
    assert good["still_growing"] is False
    assert good["back_at"] is not None and good["back_at"] <= READS

    for plan in ((211, "roll_back_payments"), (211, "nothing"), (211, "double_capacity")):
        bad = run(plan).so_far(READS)
        assert bad["checkout_working"] is False, plan
        assert bad["still_growing"] is True, plan
        assert bad["back_at"] is None, plan
        assert bad["orders_per_min"] < good["orders_per_min"], plan

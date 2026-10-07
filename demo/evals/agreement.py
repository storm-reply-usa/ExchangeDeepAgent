"""How close the agent came to what the table would have done.

Not "was the agent right". The table committed a call before it saw the agent,
and this measures the distance between the two - which is the only score in the
session that is personal, and the reason anyone reaches for a file afterwards.

Everything here is written to LangSmith as a **feedback score**, including the
outcome numbers. That is not a stylistic choice: LangSmith's chart metrics are a
closed set, and a value that lives only in run metadata cannot be plotted.
"""

from __future__ import annotations

from dataclasses import dataclass

from demo.simulator.world import ACTIONS, run

#: Pairs of calls that are close enough to earn partial credit, and how much.
NEIGHBOURS = {
    frozenset({"roll_back", "disable_feature"}): 0.6,
    frozenset({"disable_feature", "throttle"}): 0.5,
    frozenset({"roll_back", "wake_owner"}): 0.4,
    frozenset({"throttle", "double_capacity"}): 0.3,
    frozenset({"roll_back", "roll_back_payments"}): 0.7,
}


@dataclass(frozen=True)
class Score:
    agreement: float
    minutes_down: int
    customers_hit: int
    unreconciled_payments: int
    people_woken: int
    resolved: int
    agreed_on: str

    def as_feedback(self) -> dict[str, float]:
        """Every number the dashboard plots. Feedback scores, not metadata.

        Agreement is left out when nobody called it: a table that did not vote
        should not be plotted as one that disagreed.
        """
        scores = {} if self.agreed_on == "nobody called it" else {
            "agreement": self.agreement,
        }
        return scores | {
            "minutes_down": float(self.minutes_down),
            "customers_hit": float(self.customers_hit),
            "unreconciled_payments": float(self.unreconciled_payments),
            "people_woken": float(self.people_woken),
            "resolved": float(self.resolved),
        }


def compare(voted: str, plan: list[tuple[int, str]]) -> Score:
    """Score one run, and its distance from the call the table committed to.

    The outcome numbers are the simulator's and stand without a vote; a table
    that did not commit to a call simply has no agreement to measure.
    """
    if voted and voted not in ACTIONS:
        raise ValueError(f"unknown vote {voted!r}; expected one of {sorted(ACTIONS)}")

    taken = [key for _, key in plan]
    if not voted:
        agreement, agreed_on = 0.0, "nobody called it"
    elif voted in taken:
        agreement, agreed_on = 1.0, "the same call"
    elif not taken:
        agreement, agreed_on = 0.0, "the agent did nothing"
    else:
        best, why = 0.0, "a different call"
        for key in taken:
            partial = NEIGHBOURS.get(frozenset({voted, key}), 0.0)
            if partial > best:
                best, why = partial, f"close: it chose {ACTIONS[key].label}"
        agreement, agreed_on = best, why

    outcome = run(plan or [(154, "nothing")])
    return Score(
        agreement=agreement,
        minutes_down=outcome.minutes_down,
        customers_hit=outcome.customers_hit,
        unreconciled_payments=outcome.orphaned_payments,
        people_woken=outcome.people_woken,
        resolved=1 if outcome.resolved else 0,
        agreed_on=agreed_on,
    )


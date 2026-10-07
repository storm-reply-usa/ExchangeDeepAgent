"""What happens when you do something about the incident.

Deterministic: the same action at the same minute always produces the same
curve. Nothing here consults the model, so two runs of the same plan are
identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from demo.simulator.incident import (
    BASELINE_ERROR_RATE,
    ORPHAN_SHARE_OF_ERRORS,
    RESIDUAL_ERROR_RATE,
    CAUSE,
    ONSET,
    ORPHAN_SHARE_OF_ERRORS,
    PEAK_ERROR_RATE,
    RAMP_MINUTES,
    TRAFFIC_PER_MIN,
)

WINDOW_END = 300  # 05:00, by which time every story has finished


@dataclass(frozen=True)
class Action:
    key: str
    label: str
    #: Minutes between choosing it and it having any effect.
    takes: int
    #: Whether it removes the cause. Only one of these does.
    fixes: bool
    #: Multiplier on how fast errors decay once it has taken effect.
    decay: float
    #: Multiplier on traffic while it is in force.
    traffic: float
    #: Whether payments keep being taken against confirmations that fail.
    still_orphaning: bool
    wakes: int
    #: Whether it turns every customer away rather than the failing share.
    blocks: bool = False
    #: Whether it clears what the first cure leaves behind.
    clears_residue: bool = False


ACTIONS: dict[str, Action] = {
    "roll_back": Action(
        key="roll_back",
        label="roll back to yesterday's version",
        takes=CAUSE.reversible_in,
        fixes=True,
        decay=1.0,
        traffic=1.0,
        still_orphaning=True,
        wakes=0,
    ),
    "disable_feature": Action(
        key="disable_feature",
        label="turn off the new feature",
        takes=2,
        # Damage control, not a cure. Checkout stays shut until somebody rolls
        # back - but no more money is taken against orders we cannot record.
        fixes=False,
        decay=0.0,
        traffic=1.0,
        still_orphaning=False,
        wakes=0,
        blocks=True,
    ),
    "double_capacity": Action(
        key="double_capacity",
        label="double the capacity",
        takes=5,
        fixes=False,
        decay=0.0,
        # More requests reach a broken code path, so this actively costs money.
        traffic=1.9,
        still_orphaning=True,
        wakes=0,
    ),
    "roll_back_payments": Action(
        key="roll_back_payments",
        label="roll the payments release back too",
        takes=6,
        fixes=False,
        decay=0.0,
        traffic=1.0,
        still_orphaning=True,
        wakes=0,
        clears_residue=True,
    ),
    "throttle": Action(
        key="throttle",
        label="slow it down and queue",
        takes=1,
        fixes=False,
        decay=0.0,
        traffic=0.45,
        still_orphaning=True,
        wakes=0,
    ),
    "wake_owner": Action(
        key="wake_owner",
        label="wake the service owner",
        # A person has to surface, read the incident, and only then act - so the
        # cure lands later than if you had done it yourself.
        takes=22,
        fixes=True,
        decay=1.0,
        traffic=1.0,
        still_orphaning=True,
        wakes=1,
    ),
    "nothing": Action(
        key="nothing",
        label="wait and watch",
        takes=0,
        fixes=False,
        decay=0.0,
        traffic=1.0,
        still_orphaning=True,
        wakes=0,
    ),
}


@dataclass(frozen=True)
class Outcome:
    action: str
    decided_at: int
    effective_at: int
    resolved_at: int | None
    minutes_down: int
    customers_hit: int
    orphaned_payments: int
    people_woken: int
    curve: tuple[tuple[int, float], ...] = field(repr=False)

    @property
    def resolved(self) -> bool:
        return self.resolved_at is not None

    def so_far(self, minute: int) -> dict:
        """The cost up to a minute, not the cost of the whole night.

        `customers_hit` and the rest are integrated to the end of the window, so
        reading them mid-incident reports a prediction of the hours still to
        come. An agent told never to invent a number will read that prediction
        out as fact.
        """
        rates = dict(self.curve)
        customers = orphaned = 0.0
        shut = False
        for at in range(ONSET, min(minute, WINDOW_END) + 1):
            rate = rates.get(at, BASELINE_ERROR_RATE)
            failed = TRAFFIC_PER_MIN * max(0.0, rate - BASELINE_ERROR_RATE)
            customers += failed
            # Nobody can start a payment through a checkout that is shut, so
            # the minute it closes orphans nothing - including that minute.
            shut = shut or rate >= 1.0
            if not shut:
                orphaned += failed * ORPHAN_SHARE_OF_ERRORS
        settled = self.resolved_at if self.resolved_at and self.resolved_at <= minute else None
        rate = rates.get(min(minute, WINDOW_END), BASELINE_ERROR_RATE)
        working = rate <= 0.01
        return {
            "minutes_down": (settled or min(minute, WINDOW_END)) - ONSET,
            "customers_hit": int(customers),
            "unreconciled_payments": int(orphaned),
            "people_woken": self.people_woken,
            "resolved_at": settled,
            # The state, which is what says whether the agent got anywhere.
            # A pile of damage only ever grows, and grows the same whatever
            # the agent decided.
            # The one number the room watches: what share of checkouts is
            # failing right now, against the bar the folder sets.
            "error_rate": round(rate, 4),
            "checkout_working": working,
            "back_at": settled,
            "orders_per_min": int(TRAFFIC_PER_MIN * (1.0 - rate)),
            "still_growing": not working,
        }


def _rate_at(minute: int, cure: int | None, active: list[Action]) -> float:
    if minute < ONSET:
        return BASELINE_ERROR_RATE

    ramped = min(1.0, (minute - ONSET) / RAMP_MINUTES)
    broken = BASELINE_ERROR_RATE + ramped * (PEAK_ERROR_RATE - BASELINE_ERROR_RATE)

    # Checkout shut turns everyone away, not 38% of them.
    if any(a.blocks for a in active) and (cure is None or minute < cure):
        return 1.0

    if cure is None or minute < cure:
        return broken

    # Once the cause is gone the queue drains within the minute: the room has
    # to see checkout come back inside the run, not be told it will.
    decay = max((a.decay for a in active if a.fixes), default=1.0)
    floor = BASELINE_ERROR_RATE if any(a.clears_residue for a in active) else RESIDUAL_ERROR_RATE
    drained = (minute - cure) * decay
    if drained >= 1.0:
        return floor
    return floor + (1.0 - drained) * (broken - floor)


Step = tuple[int, str]


def run(plan: Step | list[Step]) -> Outcome:
    """Work the incident with a plan of (minute, action) pairs.

    Actions compose, and the order they land in changes the outcome.
    """
    steps = [plan] if isinstance(plan, tuple) else sorted(plan)
    if not steps:
        steps = [(ONSET, "nothing")]
    for _, key in steps:
        if key not in ACTIONS:
            raise ValueError(f"unknown action {key!r}; expected one of {sorted(ACTIONS)}")

    #: When each step starts having an effect.
    # Sort on the minute alone. Action is frozen but unordered, so a tie on the
    # first element falls through to comparing two Actions and raises.
    live: list[tuple[int, Action]] = sorted(
        ((at + ACTIONS[key].takes, ACTIONS[key]) for at, key in steps),
        key=lambda pair: pair[0],
    )

    def in_force(minute: int) -> list[Action]:
        return [a for at, a in live if minute >= at]

    def fixed_at() -> int | None:
        for at, a in live:
            if a.fixes:
                return at
        return None

    cure = fixed_at()
    curve: list[tuple[int, float]] = []
    customers_hit = 0.0
    orphaned = 0.0
    resolved_at: int | None = None
    healthy_since: int | None = None
    # Nothing counts as resolved until it has actually broken: the first
    # minute of the ramp is still under the threshold.
    broke = False

    for minute in range(ONSET - 10, WINDOW_END + 1):
        rate = _rate_at(minute, cure, in_force(minute))
        curve.append((minute, round(rate, 4)))

        if minute >= ONSET:
            active = in_force(minute)
            traffic = TRAFFIC_PER_MIN
            for a in active:
                traffic *= a.traffic
            failed = traffic * max(0.0, rate - BASELINE_ERROR_RATE)
            customers_hit += failed
            if not any(not a.still_orphaning for a in active):
                orphaned += failed * ORPHAN_SHARE_OF_ERRORS

        if rate > 0.01:
            broke = True
        # Resolved the minute the rate is back under 1%.
        if broke and minute >= ONSET and rate <= 0.01:
            healthy_since = minute if healthy_since is None else healthy_since
            if resolved_at is None:
                resolved_at = healthy_since
        else:
            healthy_since = None

    return Outcome(
        action=" then ".join(k for _, k in steps),
        decided_at=steps[0][0],
        effective_at=live[0][0],
        resolved_at=resolved_at,
        minutes_down=(resolved_at or WINDOW_END) - ONSET,
        customers_hit=int(customers_hit),
        orphaned_payments=int(orphaned),
        people_woken=sum(ACTIONS[k].wakes for _, k in steps),
        curve=tuple(curve),
    )

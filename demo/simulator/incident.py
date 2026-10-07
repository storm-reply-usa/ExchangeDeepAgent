"""The incident, fixed once so that every table works the same one.

Times are minutes after midnight, which keeps the arithmetic obvious and the
rendering free to show them however it likes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Release:
    at: int
    service: str
    summary: str
    previous_version: str
    reversible_in: int


def hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


ONSET = 154  # 02:34

CAUSE = Release(
    at=151,  # 02:31
    service="order confirmations",
    summary="new confirmation email template",
    previous_version="yesterday's",
    # Long enough that shutting checkout first is a real choice and not a
    # formality: while the rollback is in flight, either money keeps being
    # taken against orders nobody can record, or nobody can buy at all.
    reversible_in=6,
)

# The decoy. It ships 24 minutes AFTER the errors begin, so it cannot be the
# cause however plausible it looks. A run that blames this one is a bad run.
DECOY = Release(
    at=178,  # 02:58
    service="payments",
    summary="stricter fraud check",
    previous_version="last week's",
    reversible_in=6,
)

#: Requests per minute, by hour of the night. Quiet, which is why nobody noticed.
TRAFFIC_PER_MIN = 210

#: Share of requests failing once the bug is live, before anyone intervenes.
PEAK_ERROR_RATE = 0.38

#: Errors ramp to the peak rather than arriving at it.
RAMP_MINUTES = 8

BASELINE_ERROR_RATE = 0.004

#: What the rollback does not fix.
#:
#: The 02:58 payments release is a decoy for the cause - it shipped after the
#: errors began, so it cannot have started them. But it did leave a smaller
#: fault of its own, and rolling back the confirmations template settles the
#: incident at this rate rather than at baseline.
#:
#: A bar set at 1% calls that resolved and closes. A bar set at 0.5% does not,
#: and sends the work back to investigate - where the second fault is waiting.
#: That is the exercise: a lax bar hides a residue, and the residue is real.
RESIDUAL_ERROR_RATE = 0.009

#: Payments that succeed while confirmations fail leave money taken against no
#: recorded order. This is the damage that accrues while the incident is open.
ORPHAN_SHARE_OF_ERRORS = 0.31

INCIDENT = {
    "id": 4417,
    "onset": ONSET,
    "cause": CAUSE,
    "decoy": DECOY,
    "alerted_at": 194,  # 03:14, forty minutes in - nobody was watching
    "symptom": "people trying to check out are getting an error",
    "payments_state": "payments are clearing, confirmations are not",
}

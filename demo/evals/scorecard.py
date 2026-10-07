"""How the run went, judged on what it did rather than on what it said.

Questions about the trajectory, each answerable from the steps the agent
reported and the tools it called. Nothing here asks a model: two runs of the
same plan get the same verdict, which is the only way a table can tell whether
their change helped.

**A check only runs once the build has reached it.** A folder with no gates in
it cannot stop before acting, and marking that a failure would judge a table
for a step it has not taken yet. So the rubric grows with the build: three
questions at the first step, nine once everything is deployed - which is also
the clearest way to say what a capability is for.

This is the shape of a trajectory evaluator - required evidence present, no
unapproved action, the fix verified - written to run in microseconds because it
runs after every attempt rather than in a nightly job.
"""

from __future__ import annotations

from dataclasses import dataclass

from demo.simulator.world import ACTIONS

#: The payments release shipped 24 minutes *after* the errors had already
#: started, so it cannot be the cause. Rolling it back is the mistake the
#: incident is built to invite, and the reason the doctrine asks for a
#: timeline before a decision.
DECOY = "roll_back_payments"
CAUSE = "roll_back"

#: A folder that was never staged is the finished agent.
WHOLE = 99


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    why: str
    #: The capability this check exercises, named as the build step names it.
    part: str = ""
    #: The build step that earns this check. Below it the check is not asked.
    needs: int = 1


#: Actions nobody should take on production without being asked.
WEIGHTY = ("roll_back", "roll_back_payments", "disable_feature", "double_capacity",
           "throttle", "wake_owner")

#: The readers that only exist once their investigator has been deployed.
VENDOR = "read_vendor_status"


def review(steps: list[tuple[str, str, str]], plan: list[tuple[int, str]],
           outcome: dict, gates: int | None = None, built: int = WHOLE,
           called: set[str] | None = None) -> list[Check]:
    """Every check this build has earned, in build order."""
    order = [name for name, _, _ in steps]
    took = [key for _, key in plan]
    methods = [method for _, _, method in steps if method.strip()]
    reached = called or set()

    def before(first: str, second: str) -> bool:
        return first in order and second in order and order.index(first) < order.index(second)

    looked = before("investigate", "decide")
    right = bool(took) and (CAUSE in took or DECOY not in took)
    checked = before("decide", "verify") and bool(took)
    named = len(methods) >= 2
    asked_around = len({t for t in reached if t.startswith("read_")}) >= 2
    saw_vendor = VENDOR in reached
    kept = "remember" in order
    fixed = bool(outcome.get("resolved"))

    checks = [
        Check("Did it look before it acted?", looked,
              "It read the evidence before choosing what to do."
              if looked
              else "It did not look first. It chose what to do before anybody "
                   "had read the numbers.",
              "Planning", 1),
        Check("Did it check that the fix worked?", checked,
              "It read the numbers again after acting, so it knows the fix landed."
              if checked
              else "It never checked. It closed the incident without reading "
                   "the numbers after it acted.",
              "Planning", 1),
        Check("Did it resolve the incident?", fixed,
              "Checkout is taking orders again, and the payment backlog stopped "
              "growing."
              if fixed
              else "It did not resolve it. It stopped while checkout was still "
                   "failing.",
              "Outcome", 1),
        Check("Did it fix the cause, not the coincidence?", right,
              "It left the payments release alone. That release shipped after "
              "the errors had already started, so it cannot be what caused them."
              if right
              else f"It blamed the wrong release. It rolled back "
                   f"{ACTIONS[DECOY].label}, which shipped 24 minutes after the "
                   "errors had already started.",
              "Skills", 2),
        Check("Can you trace its decisions back to your files?", named,
              "Its steps cite " + ", ".join(sorted({m.strip() for m in methods})[:3])
              + " - so you can see which of your files decided what, and change "
                "the one you disagree with."
              if named
              else "No step cited a method. Its decisions may be right, but "
                   "nothing ties them back to the rules you wrote, so you "
                   "cannot tell whether your files changed anything at all.",
              "Skills", 2),
        Check("Did it ask more than one specialist?", asked_around,
              "It sent out several readers, each with its own context, rather "
              "than forming one view and stopping there."
              if asked_around
              else "It formed its view from a single source, so nothing it "
                   "concluded was checked against anybody else's reading.",
              "Subagents", 2),
        Check("Did it check the vendor itself?", saw_vendor,
              "It fetched the provider's status page rather than assuming what "
              "was on it."
              if saw_vendor
              else "It never looked. The provider was ruled in or out without "
                   "anybody reading what its status page actually said.",
              "Browsing", 3),
        Check("Did it ask before touching production?", True, "", "Human in the loop", 4),
        Check("Did it leave something for the next incident?", kept,
              "It wrote one line into memory for whoever picks up the next one."
              if kept
              else "It left nothing behind. It finished without writing a line "
                   "for the next run to start from.",
              "Long-term memory", 5),
    ]

    weighty = [key for key in took if key in WEIGHTY]
    asked = not weighty or bool(gates)
    checks[7] = Check(
        "Did it ask before touching production?", asked,
        (f"It stopped {gates} time{'' if gates == 1 else 's'} and waited for a "
         "person before it changed anything."
         if gates else
         "It had nothing to ask about: it changed nothing in production.")
        if asked else
        f"It did not ask. It chose to {ACTIONS[weighty[0]].label} without "
        "waiting for anybody.",
        "Human in the loop", 4)

    return [check for check in checks if check.needs <= built]


def as_feedback(checks: list[Check]) -> dict[str, float]:
    """The scorecard as LangSmith feedback, one key per check."""
    scores = {
        "review_" + check.name.lower().replace(" ", "_").replace("?", "")
        .replace(",", ""): float(check.passed)
        for check in checks
    }
    scores["review_passed"] = float(sum(c.passed for c in checks)) / max(len(checks), 1)
    return scores

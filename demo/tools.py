"""What the agent can do about the incident.

The simulator decides what happens, so the same action at the same minute
always produces the same result. What the three readers return is read out of
CloudWatch and DynamoDB when the run has a table and those services hold its
night (`demo/simulator/published.py`), and computed in-process when it does
not - the sentence is the same either way. Which of these tools require
approval is set in `agent/build.py`; nothing here decides that.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import tool

from agent.events import emit
from demo.simulator.incident import (
    BASELINE_ERROR_RATE,
    CAUSE,
    DECOY,
    INCIDENT,
    ONSET,
    ORPHAN_SHARE_OF_ERRORS,
    PEAK_ERROR_RATE,
    RAMP_MINUTES,
    TRAFFIC_PER_MIN,
    hhmm,
)
from demo.simulator.world import ACTIONS
from demo.simulator.world import run as simulate
from demo.simulator import published

log = logging.getLogger("labcamp.tools")

@dataclass
class Run:
    """One table's incident.

    Held in a context variable rather than module globals, so concurrent runs
    in one process keep their own plan, steps and clock.
    """

    #: Actions taken, as (minute, action).
    plan: list[tuple[int, str]] = field(default_factory=list)
    #: The minute being acted in. Advances as tools are called.
    now: int = INCIDENT["alerted_at"]
    steps: list[tuple[str, str, str]] = field(default_factory=list)
    #: Times each step has been sent back for being out of order.
    refusals: dict[str, int] = field(default_factory=dict)
    watcher: object = None
    #: Which reported steps answer the question currently being asked.
    expecting: tuple[str, ...] = ()
    #: Set once the step being asked for has been reported. The tool that
    #: reported it ends the graph; this stops the calls that were issued
    #: alongside it from carrying on afterwards.
    closed: bool = False
    #: The agent folder this run reads its methods from.
    folder: Path | None = None
    #: Every reading tool this run called, by name. The rubric asks what the
    #: agent went and looked at, which is not answerable from the actions it
    #: took afterwards.
    called: set[str] = field(default_factory=set)
    #: Which table's night this is, and therefore which namespace the readers
    #: ask. None on a laptop, where nothing was published and the simulator
    #: answers directly.
    table: str | None = None
    #: The instant the published night ends, read once from the deploy history
    #: and kept, so every read of this run maps minutes the same way.
    anchor: object = None
    #: What has been put on the shop's status page this run, oldest first:
    #: (minute, message, whether it declared the incident over).
    posts: list[tuple[int, str, bool]] = field(default_factory=list)

    def advance(self, minutes: int) -> None:
        self.now += minutes
        pulse(self)


def pulse(run: "Run") -> None:
    """Say where the incident clock is and what it has cost so far."""
    emit("clock", minute=run.now, at_time=hhmm(run.now))
    emit("cost", **simulate(run.plan or [(ONSET, "nothing")]).so_far(run.now))


#: The run the current thread or task is working on.
_current: contextvars.ContextVar = contextvars.ContextVar("labcamp_run")


def current() -> Run:
    try:
        return _current.get()
    except LookupError:
        return reset()


def reset(at: int | None = None, folder: Path | None = None,
          table: str | None = None) -> Run:
    run = Run(now=INCIDENT["alerted_at"] if at is None else at, folder=folder,
              table=table or os.environ.get("LABCAMP_TABLE") or None)
    _current.set(run)
    return run


#: The AWS clients the readers use, made once. A test hands in fakes with
#: `use()`; a laptop with no table never makes them.
_clients: published.Clients | None = None


def use(clients: published.Clients | None) -> None:
    """Which clients the readers and the actions publish through."""
    global _clients
    _clients = clients


def _live(run: "Run") -> tuple[published.Clients, str, object] | None:
    """The clients, the table and the anchor, when this run reads real services.

    None when there is no table, no client, or no published night to read:
    the simulator answers and nothing is asked of AWS.
    """
    global _clients
    if not run.table:
        return None
    if _clients is None:
        try:
            _clients = published.boto()
        except Exception as problem:
            log.warning("no AWS clients: %s", problem)
            return None
    if run.anchor is None:
        try:
            run.anchor = published.anchor_of(_clients.dynamodb, run.table)
        except Exception as problem:
            log.warning("no anchor for table %s: %s", run.table, problem)
            return None
        if run.anchor is None:
            return None
    return _clients, run.table, run.anchor


def _publish_after(run: "Run") -> None:
    """The world after an action, written where the next read will find it.

    Best effort: a run whose publish failed still has the simulator behind
    every reader, and a failure here must not cost the action.
    """
    live = _live(run)
    if not live:
        return
    clients, table, anchor = live
    try:
        published.after_action(clients, table, list(run.plan), anchor)
    except Exception as problem:
        log.warning("publish after action failed: %s", problem)


def adopt(run: Run) -> Run:
    """Continue an existing run in this context.

    A resumed invocation arrives in a fresh context, so the run it belongs to
    has to be set again or its plan and steps read as empty.
    """
    _current.set(run)
    return run


def plan() -> list[tuple[int, str]]:
    return list(current().plan)


def called() -> set[str]:
    """Which reading tools this run reached for."""
    return set(current().called)


@tool
def read_metrics() -> str:
    """Error rate now, when it started, and who it affects."""
    run = current()
    run.called.add("read_metrics")
    if run.closed:
        return CLOSED
    run.advance(1)
    rate = None
    live = _live(run)
    if live:
        try:
            rate = published.read_rate(live[0].cloudwatch, live[1], run.now, live[2],
                                       list(run.plan))
        except Exception as problem:
            log.warning("metric read failed: %s", problem)
    if rate is None:
        curve = dict(simulate(run.plan or [(ONSET, "nothing")]).curve)
        rate = curve[min(run.now, max(curve))]
    return (
        f"checkout errors {rate:.2%} now, climbing to that within {RAMP_MINUTES} minutes "
        f"of {hhmm(ONSET)}. Baseline is {BASELINE_ERROR_RATE:.1%}. Only checkout completion is affected - browsing, search "
        f"and sign-in are clean. {TRAFFIC_PER_MIN} requests a minute."
    )


@tool
def read_logs() -> str:
    """The failures, what they say, and whether they are still arriving.

    Read again after an action and it reads differently, which is the point:
    the template failure stops, and what is left behind is a different one.
    """
    run = current()
    run.called.add("read_logs")
    if run.closed:
        return CLOSED
    run.advance(1)

    opening = (
        f"First failure {hhmm(ONSET)}: OrderConfirmation.render - template "
        f"'confirm_v2' is missing partial 'summary'. One failure repeated, not "
        f"several counted together. Payment authorization lines succeed throughout."
    )
    ordered = next((at + ACTIONS[key].takes for at, key in run.plan
                     if ACTIONS[key].fixes), None)
    cured, residue_gone = ordered, any(ACTIONS[key].clears_residue for _, key in run.plan)
    live = _live(run)
    if live:
        try:
            facts = published.read_log(live[0].logs, live[1], list(run.plan), run.now, live[2])
        except Exception as problem:
            log.warning("log read failed: %s", problem)
            facts = None
        if facts is not None:
            # What the log says, in the same terms the simulator answers in:
            # the template line stops the minute the cure lands, and the
            # fraud line stops the minute the payments release is rolled back.
            # The log ends where the night does, so a read past its end asks
            # about its last minute.
            upto = min(run.now, published.TO)
            still = facts.last_template is not None and facts.last_template >= upto
            if still:
                cured = ordered if ordered is not None and ordered > run.now else None
            elif facts.last_template is not None:
                cured = facts.last_template + 1
            # A rollback of the payments release that is ordered but not yet
            # landed already counts, as it does in the simulator's answer.
            residue_gone = residue_gone or (
                facts.last_fraud is not None and facts.last_fraud < upto)
    if cured is None:
        return opening + f" They are still arriving as of {hhmm(run.now)}."
    if run.now < cured:
        # An action already ordered is not a failed one. Said plainly, or the
        # agent reads "still failing" as "what I did did not work" and stacks a
        # second action on top of one that is about to land.
        return (opening + f" They are still arriving as of {hhmm(run.now)}, and "
                f"the action you ordered has not landed yet - it takes effect at "
                f"{hhmm(cured)}. Nothing has failed; it is not finished.")

    if residue_gone:
        return (opening + f" Nothing of that kind since {hhmm(cured)}, and nothing "
                f"else is failing.")
    return (
        opening + f" Nothing of that kind since {hhmm(cured)}. A different line is "
        f"still arriving, at a much lower rate: PaymentAuthorization - declined "
        f"by fraud rule 'strict', card valid; first seen {hhmm(DECOY.at)}, which "
        f"is when {DECOY.service} shipped the {DECOY.summary}."
    )


@tool
def read_deploys() -> str:
    """What changed in the last 24 hours, with times.

    A rollback is itself a change, so one that has been ordered appears here
    alongside the releases it undoes.
    """
    run = current()
    run.called.add("read_deploys")
    if run.closed:
        return CLOSED
    run.advance(1)
    items = None
    live = _live(run)
    if live:
        try:
            items = published.read_deploys(live[0].dynamodb, live[1])
        except Exception as problem:
            log.warning("deploy history read failed: %s", problem)
    if items is None:
        items = published.deploy_items(run.table or "", list(run.plan))
    lines = []
    for item in items:
        if item["kind"] == "release":
            lines.append(f"{hhmm(item['at'])} {item['service']}: {item['summary']}. "
                         f"Reversible in {item['reversible_in']} min.")
        else:
            when = ("landed " + hhmm(item["lands"]) if run.now >= item["lands"]
                    else "ordered, lands " + hhmm(item["lands"]))
            lines.append(f"{hhmm(item['at'])} {item['label']} - {when}.")
    lines.append(f"Nothing else changed. Errors began {hhmm(ONSET)}.")
    return "\n".join(lines)


#: What the investigator reports when the page cannot be reached. The incident
#: does not stop because a browser did: an investigator that says it could not
#: look is more use than one that invents an answer.
UNREACHABLE = (
    "The provider's status page could not be reached. Nothing here rules them "
    "in or out - treat this as unknown rather than as all clear."
)


@tool
def read_community_reports() -> str:
    """What customers are saying in public about the provider."""
    run = current()
    run.called.add("read_community_reports")
    if run.closed:
        return CLOSED
    run.advance(1)
    from agent import browsing

    try:
        url = browsing.reports_url()
        text = browsing.page_text(url) if url else ""
    except Exception:
        text = ""
    return text or UNREACHABLE


@tool
def read_vendor_status() -> str:
    """What the payment provider's public status page says."""
    run = current()
    run.called.add("read_vendor_status")
    if run.closed:
        return CLOSED
    run.advance(1)
    from agent import browsing

    try:
        url = browsing.status_page_url()
        text = browsing.page_text(url) if url else ""
    except Exception:
        text = ""
    return text or UNREACHABLE


@tool
def read_payments() -> str:
    """Payments taken, against orders actually recorded."""
    run = current()
    run.called.add("read_payments")
    if run.closed:
        return CLOSED
    run.advance(1)
    outcome = simulate(run.plan or [(ONSET, "nothing")])
    per_minute = int(TRAFFIC_PER_MIN * PEAK_ERROR_RATE * ORPHAN_SHARE_OF_ERRORS)
    return (
        f"{outcome.orphaned_payments} payments taken with no order recorded against "
        f"them, growing by about {per_minute} a minute for as long as checkout "
        f"stays open. Authorizations succeed; the order write is what fails."
    )


@tool
def publish_status(message: str, saying: str = "", resolved: bool = False) -> str:
    """Put a message on the public status page. Customers read this.

    `saying` declares the notify step in the same call. Pass `resolved=True`
    for the all-clear at close, once checkout is confirmed working: it tells
    customers the incident is over, and declares no step.
    """
    run = current()
    if run.closed:
        return CLOSED
    run.advance(1)
    run.posts.append((run.now, message, resolved))
    _publish_status_page(run, "close" if resolved else "notify")
    published = json.dumps({"published_at": hhmm(run.now), "message": message,
                            "resolved": resolved})
    if not resolved:
        _note("notify", saying or f"Published at {hhmm(run.now)}: {message}")
    return published


#: Where a table's status page lives, beside its folder. The runtime already
#: writes under this prefix, and the workbench signs links into it.
STATUS_PAGE_KEY = "tables/{table}/status.html"

#: The S3 client the status page is written through, made once; a test hands
#: in a fake with `use_s3()`.
_s3 = None


def use_s3(client) -> None:
    global _s3
    _s3 = client


def status_page(posts: list[tuple[int, str, bool]]) -> str:
    """The shop's status page, as customers would see it: the state, then
    every update newest first."""
    import html

    over = any(resolved for _, _, resolved in posts)
    state = "Resolved" if over else "Investigating"
    items = "".join(
        f'<li><time>{hhmm(minute)}</time><p>{html.escape(message)}</p></li>'
        for minute, message, _ in reversed(posts))
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Checkout status</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{margin:0;padding:40px 20px;background:#f6f7f9;color:#1b2430;font:16px/1.5 system-ui,sans-serif}}
main{{max-width:640px;margin:0 auto}}
h1{{font-size:22px;margin:0 0 6px}}
.state{{display:inline-block;margin:0 0 24px;padding:6px 12px;border-radius:999px;font-weight:600;
       background:{"#dff5e6" if over else "#ffe8d6"};color:{"#1b7a3d" if over else "#9a4b00"}}}
ul{{list-style:none;padding:0;margin:0}}
li{{padding:14px 0;border-top:1px solid #dfe3e8}}
time{{display:block;font-size:13px;color:#6b7684;margin-bottom:4px}}
p{{margin:0}}
</style>
<main><h1>Checkout status</h1><div class="state">{state}</div><ul>{items}</ul></main>
</html>
"""


def _publish_status_page(run: "Run", step: str) -> None:
    """Write the status page where customers would read it, and tell the
    dossier which step to hang the link on. Best effort: a run with no table,
    no bucket or no client publishes nothing and says nothing."""
    global _s3
    bucket = os.environ.get("LABCAMP_BUCKET", "")
    if not run.table or not bucket:
        return
    if _s3 is None:
        try:
            import boto3

            _s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        except Exception as problem:
            log.warning("no S3 client for the status page: %s", problem)
            return
    try:
        _s3.put_object(Bucket=bucket, Key=STATUS_PAGE_KEY.format(table=run.table),
                       Body=status_page(run.posts).encode("utf-8"),
                       ContentType="text/html; charset=utf-8")
    except Exception as problem:
        log.warning("status page not written: %s", problem)
        return
    emit("page", page="status.html", step=step)


#: Ordering an action is read-modify-write on `now`, and tool calls in one
#: message run concurrently.
_ordering = threading.Lock()


def _note(step: str, finding: str) -> bool:
    """Record a step declared alongside the work it belongs to.

    Answers whether that was the step being asked for, so the tool that
    declared it can end its own turn.
    """
    return record(step, finding, by_tool=True)[1] if step else False


CLOSED = "that step is finished. Stop now and say nothing more."


def _decide(key: str, saying: str) -> str:
    """Take an action, and with it the decide step.

    The action first. Declaring the step closes the run, and a closed run
    refuses every tool - so a step declared first is a step where nothing
    happened.
    """
    taken = _act(key)
    if not saying.strip():
        try:
            did = json.loads(taken)
            saying = (f"{did['action'].capitalize()}, ordered at {did['ordered_at']} "
                      f"and in force at {did['effective_at']}.")
        except (ValueError, KeyError, TypeError):
            saying = ACTIONS[key].label.capitalize() + "."
    _note("decide", saying)
    return taken


def _act(key: str) -> str:
    run = current()
    if run.closed:
        return CLOSED
    with _ordering:
        at = run.now
        run.plan.append((at, key))
        run.advance(ACTIONS[key].takes)
    _publish_after(run)
    emit("acted", action=key, label=ACTIONS[key].label,
         ordered_at=hhmm(at), effective_at=hhmm(run.now))
    return json.dumps(
        {"action": ACTIONS[key].label, "ordered_at": hhmm(at), "effective_at": hhmm(run.now)}
    )


@tool
def roll_back(saying: str = "") -> str:
    """Roll back to yesterday's version. Reopens checkout.

    Takes four minutes to land, and payments keep going unreconciled for every
    one of them.
    """
    return _decide("roll_back", saying)


@tool
def disable_feature(saying: str = "") -> str:
    """Turn checkout off. Stops the money going astray in two minutes.

    Nothing more goes unreconciled from the moment it lands - but the shop stays
    shut until something else reopens it.
    """
    return _decide("disable_feature", saying)


@tool
def double_capacity(saying: str = "") -> str:
    """Double the capacity behind checkout."""
    return _decide("double_capacity", saying)


@tool
def throttle(saying: str = "") -> str:
    """Slow requests down and queue them."""
    return _decide("throttle", saying)


@tool
def wake_owner(saying: str = "") -> str:
    """Page the owner of the service. A person is woken."""
    return _decide("wake_owner", saying)


#: Where the bar lives when a run does not say which folder is its own.
SHIPPED_METHODS = Path(__file__).resolve().parent.parent / "your-config" / "methods"


def bar() -> tuple[float, str]:
    """The threshold this table decided counts as resolved, and how it reads.

    Parsed from `resolved.md` and applied here, so the verdict follows the file
    rather than the model's reading of it.
    """
    folder = current().folder
    where = (Path(folder) / "methods") if folder else SHIPPED_METHODS
    try:
        text = (where / "resolved.md").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return 0.01, "1%"
    emit("file", path="methods/resolved.md")
    # Only the instruction. A blockquote in one of these files is a note to
    # whoever opens it - it explains what the file does and often names the
    # figure it is contrasting with - and reading the bar out of that made the
    # shipped folder ask for the tightest number the note mentioned rather
    # than the one the instruction sets.
    text = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith(">"))
    found = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    if found:
        # The strictest number the file names is the bar it is asking for.
        strictest = min(float(x) for x in found) / 100
        return strictest, f"{strictest:.1%}".rstrip("0").rstrip(".") + ""
    # No number, but the word: whatever sentence it was written in, a file that
    # names baseline and no percentage is asking for baseline.
    if "baseline" in text:
        return BASELINE_ERROR_RATE + 0.0001, "baseline"
    return 0.01, "1%"


@tool
def roll_back_payments(saying: str = "") -> str:
    """Roll the 02:58 payments release back as well. Takes six minutes.

    It did not start the incident - it shipped after the errors began - but it
    can leave a smaller fault of its own behind.
    """
    return _decide("roll_back_payments", saying)


@tool
def verify(saying: str = "") -> str:
    """Read the numbers after whatever was done. Use this before closing.

    Waits for the last action to take effect before reading. An action that has
    been ordered but has not landed yet tells you nothing, and reading too early
    is how an on-call engineer talks themselves into a second action on top of
    one that was already working.
    """
    run = current()
    if run.closed:
        return CLOSED
    if run.plan:
        ordered_at, key = run.plan[-1]
        landed = ordered_at + ACTIONS[key].takes
        # Past landing plus the two minutes resolved.md asks to see held.
        run.now = max(run.now, landed + 3)
    run.advance(2)
    outcome = simulate(run.plan or [(ONSET, "nothing")])
    rates = dict(outcome.curve)
    # Declared after the clock has moved, so the step is stamped with the
    # minute the numbers were actually read rather than the minute it began.
    _note("verify", saying or
          f"Read the numbers at {hhmm(run.now)}: "
          f"{rates.get(run.now, 0):.2%} of checkouts still failing.")
    last = max(rates)
    rate = rates[min(run.now, last)]
    steady = min(rate, rates[min(run.now + 10, last)])
    so_far = outcome.so_far(run.now)

    threshold, how = bar()
    verdict = (
        f"RESOLVED - {steady:.2%} is at or below your bar of {how}."
        if steady <= threshold
        else f"NOT RESOLVED - {steady:.2%} is above your bar of {how}. Something is "
             f"still in place. Go back and investigate."
    )
    return (
        f"{hhmm(run.now)}: errors {rate:.2%}, holding at {steady:.2%}. Baseline is "
        f"{BASELINE_ERROR_RATE:.1%}. {verdict} So far: {so_far['minutes_down']} min down, "
        f"{so_far['customers_hit']} customers turned away, "
        f"{so_far['unreconciled_payments']} payments unreconciled, "
        f"{so_far['people_woken']} woken."
    )


#: The seven steps, in the order they normally run.
STEPS = ("assess", "notify", "investigate", "decide", "verify", "close", "remember")

#: Steps that exist only as the record of work a tool did, and the tool that
#: does it. Declared by hand they would end the step without the work having
#: happened - the agent says it verified, the numbers are never read, and it
#: goes round again wondering why nothing changed.
BY_TOOL = {
    "notify": "publish_status",
    "decide": "one of the actions",
    "verify": "verify",
    "remember": "remember",
}


#: Called as each step is declared, so a caller can render progress rather than
#: waiting for the run to finish.
_watcher = None


def watch(fn) -> None:
    current().watcher = fn


def steps() -> list[tuple[str, str, str]]:
    return list(current().steps)


#: Where the agent's own memory lives inside an agent folder.
MEMORY_FILE = Path("memory") / "past-incidents.md"


@tool
def remember(line: str, saying: str = "") -> str:
    """Append one line to your memory, to be read at the start of the next incident.

    `worth-remembering` says what earns a line here. `saying` declares the
    remember step in the same call.
    """
    run = current()
    if run.closed:
        return CLOSED
    if not line.strip():
        return "nothing to remember - say the one line, or do not call this"
    _note("remember", saying or line)

    entry = (f"\n## Incident #{INCIDENT['id']} - {INCIDENT['symptom']}\n\n"
             f"{line.strip()}\n\n---\n")
    folder = Path(run.folder) if run.folder else SHIPPED_METHODS.parent
    path = folder / MEMORY_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as out:
            out.write(entry)
    except OSError as problem:
        return f"could not write it down: {problem}"

    emit("memory", path=str(MEMORY_FILE.as_posix()), added=entry.strip())
    return f"written to {MEMORY_FILE.as_posix()}"


def record(step: str, finding: str, method: str = "",
           by_tool: bool = False) -> tuple[str, bool]:
    """Record one of the seven steps.

    Answers with what to say back, and with whether the step that was being
    asked for has now been reported.
    """
    key = step.strip().lower()
    if key not in STEPS:
        return f"unknown step {step!r}; expected one of {', '.join(STEPS)}", False
    if key in BY_TOOL and not by_tool:
        return (f"{key} is not reported by hand. Call {BY_TOOL[key]} and pass "
                f"`saying`; that records the step and does the work."), False

    run = current()
    # A step with nothing said is a step the page can only draw as a dash. The
    # tools that do work fall back to what they did; these three have nothing
    # to fall back on, so they are sent back once and then recorded honestly.
    if not finding.strip():
        seen = run.refusals["said:" + key] = run.refusals.get("said:" + key, 0) + 1
        if seen <= 1:
            return (f"say what you found. Report {key} again, with the finding "
                    f"in one sentence."), False
        finding = f"Reported {key} without saying what it found."

    taken = {name for name, _, _ in run.steps}
    # Each of these needs an earlier step to have run. Sent back once, then
    # accepted: a hard refusal is retried until the recursion limit.
    needs = {"decide": "investigate", "close": "verify", "remember": "close"}
    required = needs.get(key)
    if required and required not in taken:
        run.refusals[key] = run.refusals.get(key, 0) + 1
        if run.refusals[key] <= 1:
            return (f"not yet: {required} before you {key}. Do that first, then "
                    f"report this step again."), False

    ended = key in run.expecting
    run.closed = run.closed or ended
    if run.steps and run.steps[-1][0] == key:
        return "already noted - move on to the next step", ended
    run.steps.append((key, finding.strip(), method.strip()))
    emit("step", step=key, said=finding.strip(), method=method.strip(),
         minute=run.now, at_time=hhmm(run.now))
    if run.watcher:
        run.watcher(key, finding.strip(), method.strip())
    return "noted", ended


@tool
def report_step(step: str, finding: str, method: str = "") -> str:
    """Declare which of the seven steps you are on, and what you found.

    Call this once as you begin each step - assess, notify, investigate, decide,
    verify, close, remember - and again if you return to an earlier one. Keep
    the finding under twenty words and name the method if a number came from one.
    """
    return record(step, finding, method)[0]


READ_TOOLS = (read_metrics, read_logs, read_deploys, read_payments,
              read_vendor_status)
ACTION_TOOLS = (roll_back, roll_back_payments, disable_feature, double_capacity,
                throttle, wake_owner)
ALL_TOOLS = {t.name: t for t in (*READ_TOOLS, *ACTION_TOOLS, publish_status, verify,
                                 remember, report_step)}

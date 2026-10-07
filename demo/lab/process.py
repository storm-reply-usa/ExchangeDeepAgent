"""The seven steps, in order, with the agent asked for one at a time.

The sequence is Python's; the judgement is the agent's. At every step it still
chooses - which investigators to send, what the evidence means, which action the
doctrine calls for, whether the numbers clear the bar - and it still has the
whole harness: its own subagents, its methods, its memory, its gates.

What it no longer decides is how many turns to spend announcing where it is.
Asked to work all seven in one invocation it takes between five and eight steps
and between twenty and eighty seconds; asked for one step at a time it takes the
same seven, every run.

The one branch that is the agent's own is the loop back from verify: if its
`resolved.md` says the numbers do not clear the bar, the work returns to
investigate rather than closing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import demo.tools as incident

#: The alert. Everything the agent knows when it starts, and the only thing it
#: is told that no tool of its own could tell it.
ALERT = """PAGE - 03:14
checkout-error-rate is 38% (threshold 2%), and has been above threshold for 40
minutes. Customers cannot complete checkout. Payments are clearing; order
confirmations are not."""

#: What the agent is asked at each step. One instruction, one step.
#: One ask per step. Each ends by telling the agent to stop: an invocation runs
#: until the agent decides it is finished, and without this it works several
#: steps ahead and reports them out of order.
STOP = " Do this step only, then stop and say nothing more."

ASK = {
    "assess": (
        f"{ALERT}\n\nNobody else is awake. You will work this incident in "
        "seven steps - assess, notify, investigate, decide, verify, close, "
        "remember - and you will be asked for them one at a time.\n\n"
        "Assess it: how grave is this, and for whom? Apply your **severity** "
        "method, and weigh it against what you remember of past incidents. "
        "Send nobody yet - that is the next step. Call report_step with what "
        "you found, naming the method."
    ) + STOP,
    "notify": (
        "Publish the status page update customers will read. Say what is "
        "broken and for whom, and promise nothing you cannot verify. Pass "
        "`saying` so the notify step is recorded." + STOP
    ),
    "investigate": (
        "Now find out what is actually happening. Your **who-to-send** method "
        "says who this calls for.\n\n"
        "Write the plan first: call write_todos with one item per question you "
        "need settled, phrased the way you would brief a colleague. Then send "
        "every investigator the method names in a single message - you have no "
        "tools for reading evidence and they do, and they work at the same "
        "time, so one after another costs minutes you do not have. When their "
        "answers are back, call write_todos once more marking what is settled, "
        "so the plan shows what each of them closed. Then call "
        "report_step('investigate', ...) with what they found together." + STOP
    ),
    "decide": (
        "Decide what to do. Apply your **weighing-evidence** method to turn "
        "what they found into one action, settle the trade-off against your "
        "doctrine, and take it. Pass `saying` with the line you followed."
        + STOP
    ),
    "verify": (
        "Call verify to read the numbers after the action. Your **resolved** "
        "method holds the bar they have to clear. Pass `saying`." + STOP
    ),
    "close": (
        "Tell customers it is over: call publish_status with resolved=True and "
        "a short message saying checkout is working again. Then write the "
        "post-mortem your **post-mortem** method asks for to `postmortem.md`, "
        "using write_file. It is the artefact somebody reads tomorrow, so it "
        "belongs in a file rather than in this conversation. "
        "Then call report_step('close', ...) with a two-line summary of it, "
        "apply your **worth-remembering** method and call `remember` with the "
        "one line worth carrying into the next incident." + STOP
    ),
}

#: What a table gets before it has attached the piece a step needs. Asking for
#: a file from an agent whose writes die with the run teaches the wrong lesson
#: twice. Planning is on from the first step, so there is no ask without it.
WITHOUT_FILES = (
    "Tell customers it is over: call publish_status with resolved=True and a "
    "short message saying checkout is working again. Then report the "
    "post-mortem your **post-mortem** method asks for as the finding "
    "on report_step('close', ...). Then apply your **worth-remembering** method "
    "and call `remember` with the one line worth carrying into the next "
    "incident, both in one message." + STOP
)



#: The investigate ask before planning arrives. Naming a tool the agent does
#: not hold is not harmless - it tries, fails, and spends the step saying so.
WITHOUT_PLANNING = (
    "Now find out what is actually happening. Your **who-to-send** method "
    "says who this calls for.\n\n"
    "Send every investigator the method names in a single message - you have "
    "no tools for reading evidence and they do, and they work at the same "
    "time, so one after another costs minutes you do not have. When their "
    "answers are back, call report_step('investigate', ...) with what they "
    "found together." + STOP
)

def asks(stage: int) -> dict:
    """The seven asks, matched to how far through the build this table is."""
    from agent.build import KEEPS_FILES_FROM, PLANS_FROM

    out = dict(ASK)
    if stage < KEEPS_FILES_FROM:
        out["close"] = WITHOUT_FILES
    if stage < PLANS_FROM:
        out["investigate"] = WITHOUT_PLANNING
    return out


ORDER = ("assess", "notify", "investigate", "decide", "verify", "close")

#: A ceiling against a step that never finishes, not a budget for one that
#: does. A step normally ends the moment it reports itself, in a dozen or so
#: super-steps; this only catches one going round for ever. Set close to what a
#: step actually costs, it stops a healthy step in the middle of a tool call
#: and leaves the model a call it never sees answered - which it then refuses
#: for the rest of the incident.
TURNS_PER_STEP = 250

#: Which reported steps end each ask.
ENDS = {
    "assess": ("assess",),
    "notify": ("notify",),
    "investigate": ("investigate",),
    "decide": ("decide",),
    "verify": ("verify",),
    "close": ("remember",),
}




#: How many times the work may return to investigate before it closes anyway.
MAX_RETURNS = 2


@dataclass
class Progress:
    steps: list[tuple[str, str, str]]
    returned: int
    seconds: float


@dataclass
class Segment:
    """How far the work has got, across however many stretches it takes.

    A gate ends a stretch, and the answer arrives on a later call, so what is
    left to do and how many times the work has gone back have to outlive the
    stretch that was interrupted.
    """

    queue: list[str] = field(default_factory=lambda: list(ORDER))
    #: The asks this table gets, matched to how far through the build it is.
    ask: dict = field(default_factory=lambda: dict(ASK))
    returned: int = 0
    #: The step in hand. A gate ends the stretch in the middle of one, and the
    #: stretch that resumes has to know which step it is still finishing.
    doing: str = ""
    #: What is waiting on a person, once a stretch has ended: the tool by
    #: name, and the arguments it would be called with.
    waiting: list[dict] = field(default_factory=list)
    #: The LangSmith run each step produced, so one can be shared afterwards.
    traces: dict[str, str] = field(default_factory=dict)
    #: How many times the run stopped for a person.
    gates: int = 0


def _turns(graph, config) -> int:
    """How many super-steps this thread has taken."""
    try:
        return int((graph.get_state(config).metadata or {}).get("step", -1))
    except Exception:
        return -1


def resolved(state) -> bool:
    """Whether the last verify cleared this table's bar.

    Read from the tool's own answer rather than asked of the model again: the
    threshold lives in `resolved.md` and `verify` has already applied it.
    """
    for message in reversed(state["messages"]):
        text = getattr(message, "content", "")
        if isinstance(text, str) and "RESOLVED" in text:
            return "NOT RESOLVED" not in text
    return True


async def work_stream(graph, config, segment: Segment, *, decision=None):
    """Work the queue, yielding events, stopping when a person is needed.

    One step is asked for at a time. What the agent still chooses is everything
    that matters - which investigators to send, what the evidence means, which
    action the doctrine calls for, whether the numbers clear the bar, and
    whether to go back rather than close.
    """
    from langchain_core.tracers.context import collect_runs

    from demo.lab import stream
    from demo.lab.run import decide, pending

    step_config = {**config, "recursion_limit": TURNS_PER_STEP}
    run = incident.current()

    # An answer to a gate finishes the step it stopped, before the next is asked.
    if decision and pending(graph, config):
        segment.gates += 1
        run.closed, run.expecting = False, ENDS.get(segment.doing, ())
        async for event in stream.events(graph, decide(graph, config, decision),
                                         step_config):
            yield event
        run.expecting = ()

    while segment.queue and not pending(graph, config):
        step = segment.doing = segment.queue.pop(0)
        # The tool that reports this step ends the graph itself. Leaving the
        # agent to notice it is finished costs a run's worth of turns and can
        # leave a call unanswered in the history, which the model then refuses.
        run.closed, run.expecting = False, ENDS[step]
        try:
            with collect_runs() as traced:
                async for event in stream.events(
                        graph,
                        {"messages": [{"role": "user", "content": segment.ask[step]}]},
                        step_config):
                    yield event
            if traced.traced_runs:
                segment.traces[step] = str(traced.traced_runs[0].id)
        except Exception as problem:
            # Out of turns for this step, or the model refused the history.
            # Whatever was reported stands and the next step is asked for as
            # usual, but the count has to travel with the complaint.
            yield {"kind": "note", "step": step, "said": str(problem)[:200],
                   "turns": _turns(graph, config), "closed": run.closed}
        run.expecting = ()

        if step == "verify" and not resolved(graph.get_state(config).values):
            if segment.returned < MAX_RETURNS:
                segment.returned += 1
                segment.queue[:0] = ["investigate", "decide", "verify"]

    # One entry per gated call, deduplicated by name, keeping the arguments
    # of the first: the page needs the words a status update would carry, not
    # a second copy of the same tool.
    seen: dict = {}
    for _, name, args in pending(graph, config):
        seen.setdefault(name, {"name": name, "args": args})
    segment.waiting = [seen[name] for name in sorted(seen)]


def work(agent, config, *, ask=None, on_gate=None) -> Progress:
    """Work the incident, one step at a time.

    `on_gate` is called with the tool names waiting for a decision and returns
    one of approve, reject or edit. Without it, gates are approved.
    """
    import time

    from demo.lab.run import decide, pending

    ask = ask or ASK
    started = time.time()
    returned = 0
    queue = list(ORDER)

    while queue:
        step = queue.pop(0)
        step_config = {**config, "recursion_limit": TURNS_PER_STEP}
        run = incident.current()
        run.closed, run.expecting = False, ENDS[step]
        try:
            agent.invoke({"messages": [{"role": "user", "content": ask[step]}]},
                         step_config)
        except Exception as problem:
            # Out of turns for this step. Whatever it reported stands and the
            # next step is asked for as usual, but a step that keeps running out
            # is a real fault and has to be visible.
            print(f"  {step}: {problem}", flush=True)

        run.closed, run.expecting = False, ()

        while True:
            waiting = pending(agent, config)
            if not waiting:
                break
            names = sorted({name for _, name, _ in waiting})
            word = on_gate(names) if on_gate else "approve"
            try:
                agent.invoke(decide(agent, config, word), step_config)
            except Exception:
                break

        if step == "verify" and not resolved(agent.get_state(config).values):
            if returned < MAX_RETURNS:
                returned += 1
                queue = ["investigate", "decide", "verify", *queue]

    return Progress(steps=incident.steps(), returned=returned,
                    seconds=time.time() - started)

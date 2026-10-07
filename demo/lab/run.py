"""`lab run` - work the incident once, for one table.

The seven steps are the agent's to sequence, not ours. What this module does is
hand it the brief, render each step as it lands, and stop when a gate is reached.
"""

from __future__ import annotations

import time
from pathlib import Path

from langchain_aws import ChatBedrockConverse
from langgraph.types import Command

from agent import build_agent, read_folder
from demo import tools as incident_tools
from demo.lab.dossier import Dossier

MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

from demo.lab.process import ALERT

BRIEF = f"""{ALERT}

Nobody else is awake. Work the incident in seven steps: assess, notify,
investigate, decide, verify, close. Then remember. Say which step you are on.

Two things are not optional.

**Investigate before you decide.** You have no tools for reading evidence -
your investigators do. Send every one that applies in a single message, all at
once; they work at the same time, and one after another costs minutes you do
not have. Deciding before they report is guessing.

**Verify means after.** It reads the numbers following an action, so it tells
you nothing until you have taken one.

You may skip a step your methods say does not apply, and you may go back: if
verify says it is not resolved, investigate again rather than closing."""


def model(max_tokens: int = 1500):
    return ChatBedrockConverse(model=MODEL, region_name="us-east-1", max_tokens=max_tokens)


def thread(table: str, attempt: int) -> dict:
    """A fresh thread per attempt, which is what makes an edited method take.

    deepagents 0.7.14 loads the skill index once per thread and has no supported
    way to reload it; on a reused thread a participant's new method is simply
    not seen. Never set max_concurrency here - it silently collapses the
    investigators back into single file.
    """
    return {
        "configurable": {"thread_id": f"{table}-{attempt}"},
        "recursion_limit": 80,
        # Set at the start, not at the end. LangSmith refuses a second update to
        # a run that has already finished, and the dashboard groups on these.
        "metadata": {"table": f"table-{table}", "attempt": str(attempt)},
        "tags": [f"table-{table}"],
    }


def work(table: str, attempt: int, folder: Path, *, checkpointer=None, resume=None):
    """One pass. Returns the agent's state, paused at a gate or finished."""
    incident_tools.reset()
    agent = build_agent(
        read_folder(folder),
        incident_tools.ALL_TOOLS,
        model=model(),
        checkpointer=checkpointer,
    )
    payload = resume if resume is not None else {"messages": [{"role": "user", "content": BRIEF}]}
    started = time.time()
    state = agent.invoke(payload, thread(table, attempt))
    return state, time.time() - started


def pending(agent, config: dict) -> list[tuple[str, str, dict]]:
    """Every action waiting on a human, as (interrupt id, tool name, arguments).

    Plural: one message can carry several gated calls, so a run can stop with
    more than one interrupt pending. LangGraph requires a decision per interrupt
    id once there is more than one.

    The arguments come with the name because a name alone is not enough to
    approve: "publish a status update" is a decision about what the update
    says, and the words are in here.
    """
    snapshot = agent.get_state(config)
    waiting = []
    for interrupt in snapshot.interrupts:
        for request in interrupt.value.get("action_requests") or []:
            args = request.get("args") or request.get("arguments") or {}
            waiting.append((interrupt.id, request.get("name", "?"),
                            args if isinstance(args, dict) else {}))
    return waiting


def decide(agent, config: dict, word: str, message: str = "") -> Command:
    """One word, applied to everything currently waiting.

    Two shapes have to be right at once, and both are easy to get wrong. One
    interrupt can carry several action requests, and each of those needs its own
    decision in order - a single decision against two requests is rejected. And
    once more than one interrupt is pending, the resume has to be keyed by
    interrupt id rather than passed bare.
    """
    decision: dict = {"type": word}
    if message:
        decision["message"] = message

    per_interrupt: dict[str, int] = {}
    for interrupt_id, _, _ in pending(agent, config):
        per_interrupt[interrupt_id] = per_interrupt.get(interrupt_id, 0) + 1
    if not per_interrupt:
        raise RuntimeError("nothing is waiting on this thread")

    return Command(
        resume={
            interrupt_id: {"decisions": [dict(decision) for _ in range(count)]}
            for interrupt_id, count in per_interrupt.items()
        }
    )


import re

LEAST_SURE = re.compile(r"LEAST SURE\s*:\s*(.+)", re.IGNORECASE)

#: What a tool call adds under the step it happened in. None means the
#: participant does not need to see it - the trace has everything.
SAYS = {
    "publish_status": "drafted for the status page",
    "roll_back": "roll back to yesterday's version",
    "disable_feature": "turn checkout off",
    "double_capacity": "double the capacity",
    "throttle": "slow it down and queue",
    "wake_owner": "wake the service owner",
}


def closing_line(state) -> str:
    for message in reversed(state["messages"]):
        text = getattr(message, "content", "")
        if isinstance(text, list):
            text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
        if isinstance(text, str):
            found = LEAST_SURE.search(text)
            if found:
                return found.group(1).strip()
    return ""


def delegated(state) -> list[str]:
    """Who was sent, in order, without repeats."""
    sent: list[str] = []
    for message in state["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            if call["name"] != "task":
                continue
            who = (call.get("args") or {}).get("subagent_type", "an investigator")
            if who not in sent:
                sent.append(who)
    return sent


def actions_in(state) -> list[str]:
    lines: list[str] = []
    for message in state["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            line = SAYS.get(call["name"])
            if line and line not in lines:
                lines.append(line)
    return lines


def render(state, seconds: float, table: str, attempt: int, voted: str = "") -> None:
    """Turn a finished run into the dossier a participant reads."""
    from demo.evals.agreement import compare
    from demo.simulator.world import run as simulate

    steps = incident_tools.steps()
    closing = closing_line(state)
    sent = delegated(state)
    acted = actions_in(state)

    d = Dossier(table=table, attempt=attempt)
    d.open()
    for name, said, method in steps:
        extra: list[str] = []
        if name == "investigate" and sent:
            extra.append(f"{len(sent)} at once — {', '.join(sent)}")
        if name == "decide" and acted:
            extra.extend(a for a in acted if a != "drafted for the status page")
        d.step(name, said or "-", *extra, method=method)

    plan = incident_tools.plan()
    score = compare(voted, plan) if voted else None
    outcome = simulate(plan or [(154, "nothing")])
    d.outcome(
        outcome.minutes_down,
        outcome.customers_hit,
        outcome.orphaned_payments,
        outcome.people_woken,
        agreement=f"{score.agreement:.0%}" if score else "-",
    )
    if closing:
        head, _, path = closing.partition("->")
        d.least_sure(head.strip(), path.strip() or "your-config/")

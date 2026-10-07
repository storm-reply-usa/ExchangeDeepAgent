"""What the page is told while the agent works.

One vocabulary, in one place. The runtime emits these, the transport carries
them unchanged, and the page switches on `kind` - so a new kind is added here
and nowhere else.

Events are emitted from wherever the work happens, including tool calls running
on a worker thread, and are delivered to whichever sink the current run
installed. The sink lives in a context variable for the same reason the run
does: two tables working at once must not see each other's events.
"""

from __future__ import annotations

import contextvars
import time

#: Every kind the page knows how to draw.
#:
#: plan      the todo list, as it is written and as it is ticked
#: step      one of the seven, with what it found and the method behind it
#: dispatch  the investigators sent out together, by name
#: tool      a tool call starting, and who called it
#: file      a file in the agent folder being read
#: finding   one investigator's sentence back
#: clock     the incident minute, which is not wall-clock time
#: cost      what the incident has cost so far
#: acted     an action ordered, and the minute it lands
#: token     model output, as it is produced
#: gate      an action waiting on a person
#: gap       what this folder has nobody to look at
#: memory    what was written into the agent's own memory
#: page      a page the agent published, and the step it belongs to
#: done      the outcome, once there is one
KINDS = (
    "plan", "step", "dispatch", "tool", "file", "finding",
    "clock", "cost", "acted", "token", "gate", "gap", "memory", "page", "done",
)

_sink: contextvars.ContextVar = contextvars.ContextVar("labcamp_events", default=None)


def collect(fn) -> None:
    """Send every event from here on to `fn`. Pass None to stop."""
    _sink.set(fn)


def sink():
    """The sink this context is emitting to, if any."""
    return _sink.get()


def _writer():
    """LangGraph's custom-stream writer, when this call is inside a graph.

    A tool running inside a subagent gets one too, and what it writes arrives
    tagged with that subagent's namespace - which is what keeps each
    investigator's events in its own lane.
    """
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return None


def emit(kind: str, **what) -> None:
    """Record one event. Cheap and safe to call when nobody is listening.

    Delivery must never take the run down with it: the page losing an event is
    a blemish, the incident stopping is a failure.
    """
    event = {"kind": kind, "at": round(time.time(), 3), **what}
    for deliver in (_writer(), _sink.get()):
        if deliver is None:
            continue
        try:
            deliver(event)
        except Exception:
            pass

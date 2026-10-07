"""The on-call agent, running on AWS.

A table presses Run in the workbench and this is what answers. The agent is
assembled from that table's folder in S3, or from the one shipped here if they
have not changed anything yet, so the thing a participant edits is the thing
that runs - just not on their laptop.

The entrypoint is an async generator, which AgentCore Runtime serves as
Server-Sent Events. Each invocation works the incident until a person is needed
or until it is closed, streaming what the agent does as it does it. A gate ends
the stream; the decision arrives as the next invocation on the same session.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain_aws import ChatBedrockConverse
from langgraph.checkpoint.memory import InMemorySaver

import demo.tools as incident
from agent import build_agent, read_folder
from demo.evals.agreement import compare
from demo.simulator.world import run as simulate
from demo.evals import scorecard

app = BedrockAgentCoreApp()
log = app.logger

HERE = Path(__file__).resolve().parent
SHIPPED = HERE / "your-config"
MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
BUCKET = os.environ.get("LABCAMP_BUCKET", "")

#: Which of the seven steps is worth opening in LangSmith: the one where four
#: investigators run at once, which is the thing that does not read as a mock.
WORTH_SHARING = "investigate"


#: Fetched once, on the first run rather than at import: the runtime has to
#: finish starting in thirty seconds, and a call over the network is not
#: something to spend that budget on.
_traced = False


def tracing() -> None:
    """Turn LangSmith on, with the key fetched rather than carried.

    The key is a secret and this file is in the repository, so the runtime is
    told where the secret lives and reads it once at start-up. Without it the
    agent still works; nothing about the incident is recorded.

    A service key belongs to an organization rather than to a workspace, so
    `LANGSMITH_WORKSPACE_ID` has to be set alongside it. Without that the
    client sends no tenant header and LangSmith answers 403 to everything -
    silently, because tracing never raises.
    """
    global _traced

    where = os.environ.get("LANGSMITH_SECRET")
    if _traced or not where or os.environ.get("LANGSMITH_API_KEY"):
        return
    _traced = True
    try:
        import boto3

        held = boto3.client("secretsmanager").get_secret_value(SecretId=where)
        os.environ["LANGSMITH_API_KEY"] = held["SecretString"].strip()
    except Exception as problem:
        log.warning(f"not tracing: {problem}")


#: Where a table's folder is cached after it is pulled out of S3.
CACHE = Path("/tmp")

#: A folder without this is not a folder, whatever else is in it. Written back
#: memory alone would otherwise read as a table's whole agent.
KEEPSTONE = "doctrine.md"


def folder_for(table: str) -> Path:
    """This table's agent, pulled from S3 if they have one there.

    A prefix holding some files but not a whole folder is not this table's
    agent - it is the leavings of one - and running it would hand them an agent
    with no doctrine, no gates and nobody to send.
    """
    if not BUCKET or not table:
        return SHIPPED

    import boto3

    local = CACHE / f"agent-{table}"
    s3 = boto3.client("s3")
    prefix = f"tables/{table}/your-config/"
    found = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", [])
    if not any(obj["Key"] == prefix + KEEPSTONE for obj in found):
        if found:
            log.warning(f"table {table}: {len(found)} files but no {KEEPSTONE};"
                        f" running the shipped folder instead of theirs")
        else:
            log.info(f"table {table}: no folder of their own yet")
        return SHIPPED
    for obj in found:
        rel = obj["Key"][len(prefix):]
        if not rel:
            continue
        target = local / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, obj["Key"], str(target))
    log.info(f"table {table}: {len(found)} files from S3")
    return local


def save_memory(table: str, folder: Path) -> None:
    """Put back what the agent wrote into its own memory.

    The agent appends to the folder it was assembled from, which on AWS is a
    copy in /tmp. Without this the line it decided was worth carrying forward
    is gone before the next run reads it.
    """
    path = folder / incident.MEMORY_FILE
    if not BUCKET or not table or not path.exists() or folder == SHIPPED:
        # Nothing is written back for a table still running the shipped folder:
        # a lone memory file under their prefix would read as their whole agent
        # on the next run.
        return

    import boto3

    boto3.client("s3").put_object(
        Bucket=BUCKET,
        Key=f"tables/{table}/your-config/{incident.MEMORY_FILE.as_posix()}",
        Body=path.read_bytes(),
    )


def model():
    return ChatBedrockConverse(
        model=MODEL,
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        max_tokens=1500,
    )


#: Runs by session, so one that stopped at a gate is resumed by a later call.
_paused: dict[str, dict] = {}


def _start(payload, session: str) -> dict:
    """Everything a fresh run needs, assembled from this table's folder."""
    table = str(payload.get("table", "1"))
    folder = folder_for(table)
    from demo.lab import process

    # The run first: the agent is built around it, and holds it from then on.
    # It is told its table because the readers go and ask AWS on a namespace
    # of that table's own - and fall back to the simulator when there is
    # nothing published there, which is what makes a failed seeding harmless.
    run = incident.reset(folder=folder, table=table)
    read = read_folder(folder)
    return {
        "graph": build_agent(read, incident.ALL_TOOLS,
                             model=model(), checkpointer=InMemorySaver(), run=run),
        "config": {
            "configurable": {"thread_id": session},
            "metadata": {"table": f"table-{table}",
                         "attempt": str(payload.get("attempt", 1))},
            "tags": [f"table-{table}"],
        },
        "segment": process.Segment(ask=process.asks(read.stage)),
        "started": time.time(),
        "vote": payload.get("vote", ""),
        "table": table,
        "folder": folder,
        "run": run,
        # How far this table has built. The rubric only asks the questions the
        # build has earned, so a folder with no gates is not marked down for
        # failing to stop before acting.
        "stage": read.stage,
    }


@app.entrypoint
async def invoke(payload, context):
    """Work the incident, streaming, until a person is needed or it is closed."""
    from demo.lab import process

    session = getattr(context, "session_id", None) or str(payload.get("table", "1"))
    state = _paused.pop(session, None) or _start(payload, session)
    tracing()
    incident.adopt(state["run"])
    segment = state["segment"]
    yield _hello(state)
    for problem in _problems(state["folder"]):
        yield problem

    async for event in process.work_stream(state["graph"], state["config"], segment,
                                           decision=payload.get("decision")):
        yield event

    if segment.waiting:
        _paused[session] = state
        yield {"kind": "gate", "session": session, "waiting": segment.waiting}
        return

    save_memory(state["table"], state["folder"])
    yield _done(state, session)


def _problems(folder: Path):
    """What is wrong with this table's folder, said before the run, not after.

    deepagents drops a malformed skill without a word, and a participant who
    hears nothing concludes the session is broken rather than their
    frontmatter.
    """
    from agent import read_folder
    from agent.build import nobody_reads
    from agent.validate import check

    missing = nobody_reads(read_folder(folder), incident.ALL_TOOLS)
    if missing:
        yield {"kind": "gap", "unsent": missing}

    found = check(folder)
    if found:
        yield {"kind": "problems", "problems": [
            {"file": p.path, "said": p.what, "fix": p.fix, "fatal": p.fatal}
            for p in found
        ]}


def _hello(state: dict) -> dict:
    """What is actually running, said once per invocation.

    A deployed runtime that resolves a different library than the one every
    test ran against fails in ways that read like a bug in the agent.
    """
    import sys
    from importlib.metadata import version

    import langchain.agents.factory as factory
    import langchain.agents.middleware as middleware

    graph = state["graph"]
    run = state["run"]
    return {
        "kind": "hello",
        "python": ".".join(str(n) for n in sys.version_info[:3]),
        "libraries": {name: version(name) for name in
                      ("langchain", "langchain-core", "langgraph", "deepagents",
                       "langchain-aws")},
        "nodes": sorted(str(n) for n in graph.get_graph().nodes),
        "run": id(run),
        "current": id(incident.current()),
        "bucket": bool(BUCKET),
        "folder": str(state["folder"]),
        "one_langchain": middleware.AgentMiddleware is factory.AgentMiddleware,
        "middleware_from": middleware.AgentMiddleware.__module__,
    }


def _done(state: dict, session: str) -> dict:
    """The outcome, once the incident is closed."""
    plan = incident.plan()
    steps = incident.steps()
    try:
        score = compare(state["vote"], plan)
    except ValueError:
        # A vote arrives from the link, where a person types it. The incident
        # is over by now, and a typo must not take the whole run with it.
        score = compare("", plan)
    checks = scorecard.review(steps, plan, score.as_feedback(),
                              gates=state["segment"].gates,
                              built=state.get("stage", scorecard.WHOLE),
                              called=incident.called())
    return {
        "kind": "done",
        "session": session,
        "seconds": round(time.time() - state["started"], 1),
        "steps": [{"name": n, "said": s, "method": m} for n, s, m in steps],
        "plan": [key for _, key in plan],
        "outcome": score.as_feedback(),
        # Where the incident stands now, which is what says whether the agent
        # got anywhere. A running total of damage only ever grows.
        "recovery": simulate(plan or [(incident.ONSET, "nothing")])
                    .so_far(state["run"].now),
        "agreed_on": score.agreed_on,
        "review": [{"name": c.name, "passed": c.passed, "why": c.why, "part": c.part}
                   for c in checks],
        "trace": _publish(state, score, checks),
        # The board every table's runs land on, when one is configured. The
        # trace is this run; the board is every run, including the room's.
        "dashboard": os.environ.get("LANGSMITH_DASHBOARD_URL", ""),
    }


def _publish(state, score, checks) -> str:
    """Record the run's numbers, and open it for the table to read.

    A shared run opens without an account, which is the only way twenty people
    look at their own trace on a plan with one seat.
    """
    traces = state["segment"].traces
    run_id = traces.get(WORTH_SHARING) or next(iter(traces.values()), "")
    if not run_id or not os.environ.get("LANGSMITH_API_KEY"):
        return ""
    try:
        from langsmith import Client

        client = Client()
        posted = score.as_feedback() | scorecard.as_feedback(checks)
        for key, value in posted.items():
            client.create_feedback(run_id, key=key, score=value)
        return client.share_run(run_id)
    except Exception as problem:
        log.warning(f"scores not posted: {problem}")
        return ""


if __name__ == "__main__":
    app.run()

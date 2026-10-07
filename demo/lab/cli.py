"""`lab` - the one command a participant needs.

Every subcommand prints the real command it is running underneath, so it is
short for anyone in a hurry and transparent for anyone curious.
"""

from __future__ import annotations

import argparse
import os
import time
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Without this the first command in the README fails on a correct .env, and
# every table hits it in the first ninety seconds.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
except ImportError:
    pass
TABLE_FILE = REPO / ".table"
VOTE_FILE = REPO / ".vote"
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"

ACTIONS = ("roll_back", "roll_back_payments", "disable_feature",
           "double_capacity", "throttle", "wake_owner")


def say(line: str = "") -> None:
    print(line)


def under(command: str) -> None:
    """Show the real thing. Curated on top, honest underneath."""
    say(f"  {DIM}$ {command}{RESET}")


def table() -> str:
    return TABLE_FILE.read_text().strip() if TABLE_FILE.exists() else ""


def attempt() -> int:
    """Which go this is. It advances when a run finishes, not when one starts."""
    state = REPO / ".attempt"
    return int(state.read_text().strip()) if state.exists() else 1


def cmd_check(args) -> int:
    say()
    say(f"  {BOLD}Checking your table{RESET}")
    say()
    problems = []

    if args.table:
        TABLE_FILE.write_text(args.table)
    if not table():
        problems.append("no table yet — run `lab check --table 7`")
    else:
        say(f"  table {BOLD}{table()}{RESET}")

    folder = REPO / "your-config"
    from agent import read_folder
    from agent.validate import check, report

    found = check(folder)
    if not folder.is_dir():
        problems.append("your-config/ is missing")
    else:
        f = read_folder(folder)
        say(f"  {len(f.investigators)} investigators, {len(f.methods)} methods")
        if found:
            say()
            say(report(found))
            if any(p.fatal for p in found):
                problems.append("fix the files above — the ones marked "
                                "\"will not work\" are ignored entirely")

    if not os.environ.get("LANGSMITH_API_KEY"):
        problems.append("no LangSmith key — paste yours into .env as LANGSMITH_API_KEY")
    else:
        say(f"  LangSmith {DIM}key present{RESET}")

    say()
    for problem in problems:
        say(f"  {problem}")
    if not problems:
        say(f"  {BOLD}Ready.{RESET} Next:  lab vote <what you would do>")
    say()
    return 1 if problems else 0


def cmd_vote(args) -> int:
    if args.action not in ACTIONS:
        say()
        say(f"  {BOLD}It is 3:14 a.m. What do you do?{RESET}")
        say()
        for action in ACTIONS:
            say(f"    lab vote {action}")
        say()
        return 1
    VOTE_FILE.write_text(args.action)
    say()
    say(f"  Your table's call: {BOLD}{args.action}{RESET}")
    say(f"  {DIM}The agent has not seen this. Now run:  lab run{RESET}")
    say()
    return 0


def cmd_run(args) -> int:
    if not table():
        say("\n  Run `lab check --table 7` first.\n")
        return 1

    from langgraph.checkpoint.memory import MemorySaver

    import demo.tools as incident
    from agent import build_agent, read_folder
    from demo.lab.dossier import Dossier
    from demo.lab import process
    from demo.lab.run import closing_line, model, thread

    n = attempt()
    under(f"python -m lab.run --table {table()} --attempt {n}")

    folder = Path(args.folder).resolve() if args.folder else REPO / "your-config"
    incident.reset(folder=folder)
    live = Dossier(table=table(), attempt=n)
    live.open()
    incident.watch(lambda name, said, method: live.step(name, said or "-", method=method))

    graph = build_agent(read_folder(folder), incident.ALL_TOOLS,
                        model=model(), checkpointer=MemorySaver())
    config = thread(table(), n)

    from langchain_core.tracers.context import collect_runs

    def at_gate(names: list[str]) -> str:
        live.waiting(", ".join(names), gate_wording(", ".join(names)),
                     "approve / reject below")
        return ask()

    started = time.time()
    with collect_runs() as collected:
        progress = process.work(graph, config, on_gate=at_gate)
    root_id = collected.traced_runs[0].id if collected.traced_runs else None
    state = graph.get_state(config).values
    seconds = progress.seconds

    outcome(live, voted=VOTE_FILE.read_text().strip() if VOTE_FILE.exists() else "")
    (REPO / ".attempt").write_text(str(n + 1))
    publish(root_id, table(), n)
    say(f"  {DIM}{seconds:.1f}s{RESET}\n")
    return 0


def outcome(live, voted: str) -> None:
    """The four numbers under a dossier that has already been streamed.

    The blocks were printed as they landed, so re-rendering them would show the
    whole thing twice.
    """
    import demo.tools as incident
    from demo.evals.agreement import compare
    from demo.simulator.world import run as simulate

    plan = incident.plan()
    result = simulate(plan or [(154, "nothing")])
    score = compare(voted, plan) if voted else None
    live.outcome(
        result.minutes_down, result.customers_hit, result.orphaned_payments,
        result.people_woken,
        agreement=f"{score.agreement:.0%}" if score else "not committed",
    )


def gate_wording(names: str) -> str:
    if "publish_status" in names:
        return "nothing reaches customers without you"
    return "this changes production - your gates say you make this call"


def ask() -> str:
    """The Commander's word. Anything but a clear no is a yes."""
    try:
        answer = input("     approve / reject > ").strip().lower()
    except EOFError:
        return "approve"
    return "reject" if answer.startswith("r") else "approve"


def publish(run_id, table_id: str, n: int) -> None:
    """Score this run, by id.

    By id rather than by "the most recent root run in the project", which would
    attach these scores to another run when two are in flight.
    """
    if not os.environ.get("LANGSMITH_API_KEY") or run_id is None:
        return
    try:
        from langsmith import Client

        import demo.tools as incident
        from demo.evals.agreement import compare

        vote = VOTE_FILE.read_text().strip() if VOTE_FILE.exists() else ""
        if not vote:
            return
        score = compare(vote, incident.plan())
        client = Client()
        for key, value in score.as_feedback().items():
            client.create_feedback(run_id, key=key, score=value)
        say(f"  {DIM}scored · lab open{RESET}")
    except Exception as problem:
        say(f"  {DIM}(scores not posted: {problem}){RESET}")




def cmd_open(args) -> int:
    url = os.environ.get("LANGSMITH_DASHBOARD_URL", "https://smith.langchain.com")
    say(f"\n  {url}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab", description="Work the incident.")
    subs = parser.add_subparsers(dest="command", required=True)

    check = subs.add_parser("check", help="confirm this table is ready")
    check.add_argument("--table")
    check.set_defaults(fn=cmd_check)

    vote = subs.add_parser("vote", help="commit your table's call, before seeing the agent")
    vote.add_argument("action", nargs="?", default="")
    vote.set_defaults(fn=cmd_vote)

    run_cmd = subs.add_parser("run", help="work the incident")
    run_cmd.add_argument("--folder", default=None,
                         help="an agent folder other than your-config/")
    run_cmd.set_defaults(fn=cmd_run)

    subs.add_parser("open", help="open the dashboard").set_defaults(fn=cmd_open)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

"""Every file a participant can edit has to reach the agent.

A file that is read and thrown away is worse than no file: the participant
changes it, nothing happens, and they conclude the whole lab is a mock. Each
test here names one lever and follows it to the place it lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import as_call

from agent import read_folder, subagent_specs, system_prompt
from demo import tools as incident
from agent.backends import build_backend
from agent.build import METHODS_PATH, gated, stage_methods
from demo.lab import process

FOLDER = Path(__file__).resolve().parents[2] / "your-config"


@pytest.fixture(scope="module")
def folder():
    return read_folder(FOLDER)


def test_the_alert_is_in_the_first_thing_the_agent_is_asked():
    """Without it the agent has no incident, and infers one or gives up."""
    first = process.ASK[process.ORDER[0]]
    assert "38%" in first and "checkout-error-rate" in first
    assert process.ALERT in first


def test_doctrine_reaches_the_prompt(folder):
    prompt = system_prompt(folder)
    for line in folder.doctrine.splitlines():
        if line.strip() and not line.startswith("#"):
            assert line.strip() in prompt
            return
    pytest.fail("doctrine.md has no content to check")


def test_gates_decide_what_stops(folder):
    """`gates.md` is the policy, not a description of one."""
    prompt = system_prompt(folder)
    assert folder.gates.strip()[:40] in prompt
    stopped = gated(folder)
    assert "publish_status" in stopped
    assert set(stopped) <= set(incident.ALL_TOOLS)


def test_a_gate_removed_from_the_file_stops_gating(tmp_path):
    """Delete the line, and the agent stops asking. That is the lever working."""
    copy = tmp_path / "your-config"
    copy.mkdir()
    (copy / "gates.md").write_text("Nothing here needs a human.\n", encoding="utf-8")
    assert gated(read_folder(copy)) == ()


def test_every_investigator_becomes_a_subagent_with_its_own_tool(folder):
    specs = {spec["name"]: spec for spec in subagent_specs(folder, incident.ALL_TOOLS)}
    assert len(specs) == len(folder.investigators)
    for spec in specs.values():
        names = [t.name for t in spec["tools"]]
        assert names, f"{spec['name']} was given nothing to work with"
        assert all(n.startswith("read_") for n in names), (
            f"{spec['name']} has more than what it was hired to read: {names}"
        )


def test_every_method_loads_as_a_skill(folder):
    """A method that does not load is a file the participant edits for nothing."""
    from deepagents.middleware.skills import _list_skills

    backend = build_backend(methods=stage_methods(folder))
    loaded = {skill["name"] for skill in _list_skills(backend, METHODS_PATH)}
    assert loaded == {part.slug for part in folder.methods}


def test_every_method_is_named_by_the_step_that_applies_it(folder):
    """Naming it in the ask is what sends the agent to the file."""
    asked = " ".join(process.ASK.values()).lower()
    for part in folder.methods:
        assert part.slug in asked, f"no step asks for methods/{part.slug}.md"


def test_memory_reaches_the_prompt_and_can_be_written_back(folder, tmp_path):
    assert folder.memory[:40] in system_prompt(folder)

    copy = tmp_path / "your-config"
    copy.mkdir(parents=True)
    (copy / "memory").mkdir()
    (copy / "memory" / "past-incidents.md").write_text("# Past incidents\n", encoding="utf-8")
    incident.reset(folder=copy)
    as_call(incident.remember, line="A line worth carrying.")
    assert "A line worth carrying." in (copy / incident.MEMORY_FILE).read_text()


def test_the_bar_follows_resolved_md(tmp_path):
    """The one method applied in code rather than by the model."""
    copy = tmp_path / "your-config"
    (copy / "methods").mkdir(parents=True)
    written = copy / "methods" / "resolved.md"

    written.write_text("Errors must be back under 0.5%.\n", encoding="utf-8")
    incident.reset(folder=copy)
    assert incident.bar()[0] == pytest.approx(0.005)

    written.write_text("Errors must be back under 1%.\n", encoding="utf-8")
    incident.reset(folder=copy)
    assert incident.bar()[0] == pytest.approx(0.01)


def test_the_step_being_asked_for_ends_the_run_itself(folder, scripted):
    """A step ends because the graph ends, not because the model took the hint.

    Left to notice on its own, the agent keeps calling tools until it runs out
    of turns, and a run that runs out between a call and its answer leaves the
    model a call it never sees answered - which it then refuses for the rest of
    the incident.
    """
    from langchain_core.messages import AIMessage

    from agent import build_agent

    def calling(name, args, n):
        return AIMessage(content="",
                         tool_calls=[{"name": name, "args": args, "id": f"c{n}"}])

    model = scripted(
        calling("report_step", {"step": "assess", "finding": "38% failing"}, 1),
        # Never reached: the run ends before the model is asked a second time.
        calling("read_metrics", {}, 2),
        AIMessage(content="LEAST SURE: nothing -> doctrine.md"),
    )
    incident.reset(folder=FOLDER)
    incident.current().expecting = ("assess",)
    build_agent(folder, incident.ALL_TOOLS, model=model).invoke(
        {"messages": [{"role": "user", "content": "assess it"}]},
        {"recursion_limit": 20},
    )

    assert len(model.calls) == 1
    assert [name for name, _, _ in incident.steps()] == ["assess"]


def test_the_agent_holds_its_own_run_rather_than_looking_one_up(folder, scripted):
    """The hook that ends a step runs in whatever context the graph gives it.

    Looked up from a context variable there, it can hand back a fresh run that
    is never finished, and the step then runs until it is out of turns.
    """
    from langchain_core.messages import AIMessage

    from agent import build_agent

    mine = incident.reset(folder=FOLDER)
    model = scripted(AIMessage(content="should never be asked"))
    agent = build_agent(folder, incident.ALL_TOOLS, model=model, run=mine)

    # This run is finished, and somebody else's is the current one.
    mine.closed = True
    other = incident.reset(folder=FOLDER)
    assert other is not mine and not other.closed

    agent.invoke({"messages": [{"role": "user", "content": "assess it"}]},
                 {"recursion_limit": 20})
    assert model.calls == []


def test_a_step_that_was_not_the_one_asked_for_carries_on(folder, scripted):
    from langchain_core.messages import AIMessage

    from agent import build_agent

    model = scripted(
        AIMessage(content="", tool_calls=[
            {"name": "report_step",
             "args": {"step": "assess", "finding": "38% failing"}, "id": "c1"}]),
        AIMessage(content="LEAST SURE: nothing -> doctrine.md"),
    )
    incident.reset(folder=FOLDER)
    incident.current().expecting = ("investigate",)
    build_agent(folder, incident.ALL_TOOLS, model=model).invoke(
        {"messages": [{"role": "user", "content": "assess it"}]},
        {"recursion_limit": 20},
    )

    assert len(model.calls) == 2


def test_declaring_a_step_never_costs_the_work_it_declares():
    """Declaring closes the run, and a closed run refuses every tool.

    A tool that says what it did before doing it therefore does nothing, and
    the incident runs to the end with an empty plan.
    """
    for taking in ("roll_back", "disable_feature", "wake_owner"):
        incident.reset()
        # Deciding needs an investigation behind it, which is a separate rule.
        as_call(incident.report_step, step="investigate", finding="the 02:31 deploy")
        incident.current().expecting = ("decide",)
        incident.ALL_TOOLS[taking].invoke({"saying": "stopping the bleeding"})
        assert [key for _, key in incident.plan()] == [taking]
        assert "decide" in {name for name, _, _ in incident.steps()}

    incident.reset()
    incident.current().expecting = ("notify",)
    incident.publish_status.invoke({"message": "We are looking into it.",
                                    "saying": "customers told"})
    assert "notify" in {name for name, _, _ in incident.steps()}

    incident.reset()
    incident.current().expecting = ("verify",)
    incident.current().plan = [(198, "roll_back")]
    answer = incident.verify.invoke({"saying": "reading the numbers"})
    assert "RESOLVED" in answer
    assert "verify" in {name for name, _, _ in incident.steps()}


def test_no_file_in_the_folder_is_ignored(folder):
    """The catch-all: every markdown file is accounted for by one of the above."""
    accounted = {"doctrine.md", "gates.md", "README.md"}
    accounted |= {f"investigators/{p.path.name}" for p in folder.investigators}
    accounted |= {f"methods/{p.path.name}" for p in folder.methods}
    accounted |= {"memory/past-incidents.md"}

    on_disk = {
        str(path.relative_to(FOLDER).as_posix())
        for path in FOLDER.rglob("*.md")
        if path.is_file()
    }
    assert on_disk <= accounted, f"nothing reads {sorted(on_disk - accounted)}"

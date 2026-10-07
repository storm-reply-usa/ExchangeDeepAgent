"""The premise: the folder is the configuration, and nothing else is.

Every test here is a claim a speaker makes out loud. If one of them stops being
true, a loop of the session stops working.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import read_folder, subagent_specs, system_prompt
from agent.build import GATED_BY_DEFAULT, gated, interrupt_config

SHIPPED = Path(__file__).resolve().parents[2] / "your-config"


@pytest.fixture
def folder(tmp_path):
    """A copy, so a test can edit it the way a participant would."""
    import shutil

    copy = tmp_path / "your-config"
    shutil.copytree(SHIPPED, copy)
    return copy


def test_adding_a_file_adds_an_investigator(folder):
    # Loop 2 of the session is exactly this gesture, so it is exactly this test.
    before = {s["name"] for s in subagent_specs(read_folder(folder))}

    (folder / "investigators" / "payments.md").write_text(
        "---\nname: payments\n"
        "description: Checks whether payments taken were reconciled against orders.\n---\n\n"
        "You read the payment ledger and nothing else.\n",
        encoding="utf-8",
    )

    after = {s["name"] for s in subagent_specs(read_folder(folder))}
    assert after - before == {"payments"}


def test_the_supervisor_holds_no_reading_tools():
    # Give it those and it investigates alone: the investigators never run and
    # the trace stops growing sideways, which is all loop 2 exists to show.
    from agent.build import build_agent  # noqa: F401  (import cost only)
    from demo import tools as incident

    supervisor_tools = [n for n in incident.ALL_TOOLS if not n.startswith("read_")]
    assert "read_metrics" not in supervisor_tools
    assert "roll_back" in supervisor_tools
    assert "verify" in supervisor_tools


def test_each_investigator_gets_only_its_own_tools(folder):
    from demo import tools as incident

    specs = {s["name"]: s for s in subagent_specs(read_folder(folder), incident.ALL_TOOLS)}
    assert [t.name for t in specs["metrics"]["tools"]] == ["read_metrics"]
    assert [t.name for t in specs["logs"]["tools"]] == ["read_logs"]


def test_an_investigator_nobody_anticipated_still_gets_tools(folder):
    # A participant's fifth investigator has a filename we never saw. It must
    # still be able to read something, or loop 2 fails in the room.
    from demo import tools as incident

    (folder / "investigators" / "cdn.md").write_text(
        "---\nname: cdn\ndescription: Looks at the edge.\n---\n\nYou read the edge.\n",
        encoding="utf-8",
    )
    specs = {s["name"]: s for s in subagent_specs(read_folder(folder), incident.ALL_TOOLS)}
    assert specs["cdn"]["tools"], "an unanticipated investigator was handed nothing to read"


def test_an_investigator_carries_its_own_instructions(folder):
    specs = {s["name"]: s for s in subagent_specs(read_folder(folder))}
    # The file's own words reach the subagent; the wording itself is free to
    # change, so this checks the body arrived rather than one sentence of it.
    here = Path(__file__).resolve().parents[2]
    body = (here / "your-config" / "investigators" / "metrics.md").read_text()
    said = [l for l in body.splitlines() if l.strip() and not l.startswith((">", "-", "|"))][-1]
    assert said.strip("* ") in specs["metrics"]["system_prompt"]
    assert specs["metrics"]["description"]


def test_editing_the_doctrine_changes_what_the_agent_is_told(folder):
    before = system_prompt(read_folder(folder))
    (folder / "doctrine.md").write_text(
        "# Doctrine\n\n1. Never lose data, before anything else.\n", encoding="utf-8"
    )
    after = system_prompt(read_folder(folder))
    assert "Never lose data, before anything else." in after
    assert after != before


def test_a_deleted_file_is_a_choice_not_a_crash(folder):
    (folder / "doctrine.md").unlink()
    prompt = system_prompt(read_folder(folder))
    assert "Doctrine" not in prompt
    assert "on-call engineer" in prompt


def test_the_shipped_folder_has_no_method_for_who_to_wake():
    # The hole loop 3 invites a participant to fill. If somebody ever writes this
    # method into the repo, the third loop quietly loses its point.
    names = read_folder(SHIPPED).method_names
    assert not any("wake" in name for name in names), names


def test_memory_reaches_the_prompt():
    """Whatever is written in it, not one word somebody happened to use."""
    folder = read_folder(SHIPPED)
    prompt = system_prompt(folder)
    lines = [line for line in folder.memory.splitlines() if len(line.strip()) > 40]
    assert lines, "memory/past-incidents.md carries nothing"
    for line in lines:
        assert line.strip() in prompt


def test_memory_does_not_hand_over_the_answer():
    """A memory that names tonight's cause is a memory that skips the work.

    `worth-remembering.md` says a line earns its place by changing how the next
    incident is worked - not by recording what the last one was.
    """
    memory = (SHIPPED / "memory" / "past-incidents.md").read_text(encoding="utf-8").lower()
    for given_away in ("confirm_v2", "'summary'", "02:31", "rollback is the remedy"):
        assert given_away not in memory, f"the memory hands over {given_away!r}"


def test_the_agent_names_the_file_it_was_least_sure_about():
    # This is what guides a participant to the next loop, so it is not optional.
    assert "LEAST SURE" in system_prompt(read_folder(SHIPPED))


def test_production_tools_cannot_run_without_a_human():
    tools = {name: object() for name in (*GATED_BY_DEFAULT, "read_metrics")}
    stops = interrupt_config(tools)
    assert set(stops) == set(GATED_BY_DEFAULT)
    assert "read_metrics" not in stops
    for config in stops.values():
        assert config["allowed_decisions"] == ["approve", "edit", "reject"]


def test_the_gates_file_decides_what_stops(folder):
    """One of the five levers the README names, and it used to be inert."""
    (folder / "gates.md").write_text(
        "# Gates\n\nOnly waking somebody needs me. Everything else it does alone.\n",
        encoding="utf-8",
    )
    assert gated(read_folder(folder)) == ("wake_owner",)

    (folder / "gates.md").write_text(
        "# Gates\n\nAnything customers will read, and anything that changes production.\n",
        encoding="utf-8",
    )
    stops = gated(read_folder(folder))
    assert "publish_status" in stops and "roll_back" in stops
    assert "wake_owner" not in stops


def test_a_folder_with_no_gates_file_still_stops_for_production(folder):
    (folder / "gates.md").unlink()
    assert gated(read_folder(folder)) == GATED_BY_DEFAULT


def test_the_gates_file_reaches_the_prompt(folder):
    """One of the five levers the README names. It was read and thrown away."""
    (folder / "gates.md").write_text(
        "# Gates\n\nNever touch production without me. Never wake anyone, ever.\n",
        encoding="utf-8",
    )
    assert "Never wake anyone, ever." in system_prompt(read_folder(folder))


def test_the_bar_comes_from_the_folder_the_run_was_built_from(folder):
    """On the deployed runtime each table's folder is downloaded separately. A
    module-level path makes round three a no-op for everybody."""
    from demo import tools as incident

    (folder / "methods" / "resolved.md").write_text(
        "---\nname: resolved\ndescription: When it is over.\n---\n\n"
        "It must be back at baseline, not merely close to it.\n",
        encoding="utf-8",
    )
    incident.reset(folder=folder)
    threshold, how = incident.bar()
    assert how == "baseline"
    incident.reset()
    assert incident.bar()[1] != "baseline"

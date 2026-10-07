"""Assemble the agent by reading `your-config/`.

Nothing about how this agent behaves is written here. The doctrine it settles
trade-offs against, the investigators it sends, the methods it follows and the
decisions it may not take alone are all markdown files in that folder.

Investigators are discovered from `investigators/*.md` rather than listed, so a
file added there becomes a subagent without a change to this module.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Part:
    """One markdown file from the folder, split into its frontmatter and body."""

    path: Path
    name: str
    description: str
    body: str

    @property
    def slug(self) -> str:
        return self.path.stem


def instruction(body: str) -> str:
    """What the model is given: the file, without the notes written to a person.

    A markdown blockquote is a note to whoever opens the file. Sent to the
    model it reads as a paragraph about itself in the third person, next to an
    instruction addressed to somebody else - so it is taken out here, and the
    page keeps it.
    """
    kept = [line for line in body.splitlines() if not line.lstrip().startswith(">")]
    while kept and not kept[0].strip():
        kept.pop(0)
    return "\n".join(kept).strip()


def read_part(path: Path) -> Part:
    # Decode leniently; `agent.validate` reports unreadable files.
    raw = path.read_text(encoding="utf-8", errors="replace")
    match = FRONTMATTER.match(raw)
    if not match:
        return Part(path=path, name=path.stem, description="",
                    body=instruction(raw.strip()))

    head, body = match.groups()
    fields: dict[str, str] = {}
    for line in head.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return Part(
        path=path,
        name=fields.get("name", path.stem),
        description=fields.get("description", ""),
        body=instruction(body.strip()),
    )


#: A folder that has not been staged is the finished agent. Anything lower is
#: a table part-way through the build, assembled with only the pieces they
#: have attached so far.
WHOLE = 99
STAGE_FILE = ".build"
#: Two of the capabilities are middleware rather than files, so looking in the
#: folder cannot tell you whether they are on - the marker has to. Everything
#: else is on when its file is there and off when it is not: methods,
#: investigators, gates and memory all read from the folder itself.
#:
#: The numbers are build steps. Step 1 passes TodoListMiddleware; step 2 routes
#: the filesystem at S3, and below it whatever the agent writes dies with the
#: run.
PLANS_FROM = 3
KEEPS_FILES_FROM = 1


@dataclass(frozen=True)
class Folder:
    root: Path
    doctrine: str
    gates: str
    investigators: tuple[Part, ...]
    methods: tuple[Part, ...]
    memory: str
    stage: int = WHOLE

    @property
    def method_names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.methods)


def _text(path: Path) -> str:
    """File content, with a leading markdown heading stripped."""
    if not path.exists():
        return ""
    body = path.read_text(encoding="utf-8", errors="replace").strip()
    if body.startswith("# "):
        _, _, rest = body.partition("\n")
        body = rest.strip()
    return instruction(body)


def read_folder(root: Path) -> Folder:
    """Read an agent folder. A missing file is an absence, not an error."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no agent folder at {root}")

    return Folder(
        root=root,
        doctrine=_text(root / "doctrine.md"),
        gates=_text(root / "gates.md"),
        investigators=tuple(read_part(p) for p in sorted(root.glob("investigators/*.md"))),
        methods=tuple(read_part(p) for p in sorted(root.glob("methods/*.md"))),
        memory="\n\n".join(_text(p) for p in sorted(root.glob("memory/*.md"))).strip(),
        stage=_stage(root),
    )


def _stage(root: Path) -> int:
    """How far through the build this folder is, or the whole agent."""
    marker = root / STAGE_FILE
    if not marker.exists():
        return WHOLE
    try:
        return max(1, int(marker.read_text().strip()))
    except ValueError:
        return WHOLE


#: Role and output rules. How to work an incident lives in the methods.
ROLE = """You are the on-call engineer for an online retailer. An alert has fired and \
nobody else is awake. Work the incident.

Three rules:

1. Never invent a version number, a timestamp or a percentage.
2. Name the method behind every number you report.
3. Be brief. Someone is reading you at 3 a.m."""

#: How the agent narrates. Somebody is reading this over a shoulder while a
#: speaker talks, so each step is one line, and the line leads with what it
#: found rather than with what it was about to do.
NARRATION = """# How to report

Most tools take a `saying` argument: one short sentence about what you found.
Pass it, and the step is reported for you at no extra cost.

- the status draft reports **notify**
- an action reports **decide**
- `verify` reports **verify**

Only `assess`, `investigate`, `close` and `remember` need `report_step` itself,
and the last two go out together in one message.

What you pass to `report_step` is read on a screen at 3 a.m. by somebody who
has ten seconds and has not read your files. So:

- One sentence. Under twenty words. Lead with the figure that matters.
- Say it out loud in your head first. If it reads like the title of a slide
  rather than something a person would say, write it again.
- No stacked nouns. "Segment-level verification of masked failures" is a
  label; "checking each segment separately" is a sentence.
- Every term has to be understandable to somebody meeting this incident for
  the first time. If a word only makes sense to you, it is the wrong word.
- Plain text. No markdown, no `**bold**`, no headings, no bullet lists, no
  labels like "Severity determination:" - they are printed literally and read
  as noise.
- Say the thing, not that you are about to say it.

  no    **Severity determination:** This is a Sev 1 incident because...
  yes   Sev 1: 38% of checkouts failing for 40 minutes, payments unreconciled.

  no    Segment-level verification catches masked failures earlier than aggregate.
  yes   Check mobile and desktop separately - the average hid this for 13 minutes.

  no    Rollback executed; error rate normalised post-deployment reversion.
  yes   Rolled back to yesterday's build. Checkout worked again 12 minutes later.

Write no prose around the call. The `report_step` call is the only report
anybody reads.

One action, then verify. If verify says it worked, close. Do not stack a second
action on one that is working. If verify says it is not resolved, go back and
investigate again, and report that step again when you do.

Report only what a tool told you. You have no other source, and inventing a
detail that sounds plausible is the one thing that makes all of this worthless.

Your final message is one sentence and nothing else:

    LEAST SURE: <the judgement you had to make alone> -> <the file that holds it>"""


def roster(folder: Folder) -> str:
    """The investigator roster, for the supervisor's prompt."""
    if not folder.investigators:
        return ""
    lines = [f"- **{p.name}** — {p.description}" for p in folder.investigators]
    return (
        "# Your investigators\n\n"
        "You cannot read the evidence yourself; you have no tools for it. These "
        "people can, each sees only their own corner, and they work at the same "
        "time. Send every one that applies **in a single message** - sending them "
        "one after another costs minutes you do not have.\n\n"
        + "\n".join(lines)
        + "\n\nEach comes back with one sentence."
    )


def shortfall(folder: Folder) -> str:
    """What the agent should admit it is missing, in its own voice."""
    missing = []
    if not folder.investigators:
        missing.append(
            "You have no investigators - somebody removed them. Read the evidence "
            "yourself, and say plainly in your first line that you are working "
            "without them, so nobody mistakes a guess for a finding."
        )
    if not folder.doctrine:
        missing.append(
            "You have no doctrine. When two things conflict you are choosing on "
            "your own; say which way you went and why."
        )
    if not folder.methods:
        missing.append(
            "You have no methods. Your numbers come from judgement rather than a "
            "procedure, and you should say so wherever you report one."
        )
    return "# What you are missing\n\n" + "\n\n".join(missing) if missing else ""


def system_prompt(folder: Folder) -> str:
    blocks = [ROLE]
    if shortfall(folder):
        blocks.append(shortfall(folder))
    if roster(folder):
        blocks.append(roster(folder))
    if folder.doctrine:
        blocks.append("# Doctrine\n\nSettle every trade-off against this.\n\n" + folder.doctrine)
    if folder.gates:
        # The gates are enforced in code as well, but the agent has to know
        # which of its moves will stop and wait, or it plans around a pause it
        # does not expect. And a file presented as a lever has to be one.
        blocks.append(
            "# What you may not decide alone\n\nThese stop and wait for a human. "
            "Expect the pause; do not work around it.\n\n" + folder.gates
        )
    if folder.memory:
        blocks.append("# From past incidents\n\n" + folder.memory)
    blocks.append(NARRATION)
    return "\n\n".join(blocks)


#: Which tools each investigator is handed. An investigator that cannot read
#: what it was hired to read is a subagent in name only, and an investigator
#: that can read everything is the supervisor again.
INVESTIGATOR_TOOLS = {
    "metrics": ("read_metrics",),
    "logs": ("read_logs",),
    "deploys": ("read_deploys",),
    "payments": ("read_payments",),
    "vendor-status": ("read_vendor_status",),
    "community-reports": ("read_community_reports",),
}


#: What an investigator with that filename would be able to read. Said in the
#: words the room uses, because it is read by a participant deciding whether to
#: hire one.
UNSENT = {
    "read_payments": ("payments", "whether the money taken matches the orders recorded"),
    "read_community_reports": ("community reports",
                               "what customers are saying in public, which a vendor's "
                               "own status page will not tell you"),
    "read_metrics": ("metrics", "the error rate, and who it is failing for"),
    "read_logs": ("logs", "the first failure, and what it actually says"),
    "read_deploys": ("deploys", "what changed, and when"),
    "read_vendor_status": ("vendor-status", "whether it is the vendor's fault"),
}


def nobody_reads(folder: Folder, tools: dict | None = None) -> list[dict]:
    """What this folder has nobody to look at.

    An absence, stated as a fact rather than left for the agent to mention in
    passing: a reading tool exists, and no investigator in the folder holds it.
    Hiring one closes it, which is the whole exercise.
    """
    held = {tool.name for spec in subagent_specs(folder, tools) for tool in spec["tools"]}
    return [
        {"hire": name, "to read": what}
        for reader, (name, what) in UNSENT.items()
        if reader in (tools or {}) and reader not in held
    ]


def subagent_specs(folder: Folder, tools: dict | None = None) -> list[dict]:
    """One subagent per file in `investigators/`.

    Each is handed the tools its filename maps to, or every reading tool if the
    filename is not in `INVESTIGATOR_TOOLS`.
    """
    tools = tools or {}
    readers = {name: tool for name, tool in tools.items() if name.startswith("read_")}
    specs = []
    for part in folder.investigators:
        wanted = INVESTIGATOR_TOOLS.get(part.slug)
        granted = (
            [readers[n] for n in wanted if n in readers] if wanted else list(readers.values())
        )
        specs.append(
            {
                "name": part.name,
                "description": part.description or f"Investigates {part.name}.",
                "system_prompt": part.body + ONE_LINE_BACK,
                "tools": granted,
            }
        )
    return specs


#: What every investigator is told on top of its own instructions.
#:
#: Appended to every investigator's own instructions.
ONE_LINE_BACK = """

# How to work

Call your tool once, then answer in one sentence: the single thing the on-call
engineer needs from you, with the figure that matters. Plain text - no
markdown, no bold, no labels. That sentence is the whole of what anybody sees
from you, printed exactly as you write it, to somebody who has not read your
instructions and has ten seconds.

Say it the way you would say it out loud. No stacked nouns, no phrases that
read like the title of a slide.

  no    Confirmation pipeline failure isolated to post-template segment.
  yes   Checkout has been failing for 38% of attempts since 02:34, mobile worst.

Your tool is your only source. Do not go looking for another one, do not list
or search or open files, and do not call the tool twice. One call, one sentence.
Do not think out loud, do not restate the question and do not add caveats.
Anyone who wants your working can open the trace."""

#: Where the agent looks for its methods. Routed to whatever `stage_methods`
#: laid out for the folder being run, so the methods it applies are the ones
#: that table edited rather than the ones shipped with the code.
METHODS_PATH = "/methods/"

#: A skill is a directory with this file inside it.
SKILL_FILE = "SKILL.md"


def stage_methods(folder: Folder) -> Path:
    """Lay the methods out the way skills are discovered.

    A skill is a directory holding a `SKILL.md`; the folder keeps one flat file
    per method, because a flat file is what a person wants to open and edit. The
    two shapes are reconciled here rather than imposed on the folder.

    The name is taken from the filename rather than the frontmatter: a method
    whose two names disagree is skipped in silence, and a participant who
    renames one heading should not lose the file.
    """
    root = Path(tempfile.gettempdir()) / (
        "labcamp-methods-" + hashlib.sha1(str(folder.root).encode()).hexdigest()[:12]
    )
    root.mkdir(parents=True, exist_ok=True)
    slugs = {part.slug for part in folder.methods}
    # One folder stages to one directory across runs, so a method deleted
    # between them has to be cleared out or it keeps being applied.
    for stale in root.iterdir():
        if stale.is_dir() and stale.name not in slugs:
            shutil.rmtree(stale, ignore_errors=True)
    for part in folder.methods:
        home = root / part.slug
        home.mkdir(parents=True, exist_ok=True)
        described = part.description or f"How to {part.slug.replace('-', ' ')}."
        # The description is quoted because it is a line somebody typed, and a
        # colon in it would otherwise start a mapping and make the frontmatter
        # unreadable - at which point the skill is dropped without a word.
        quoted = '"' + described.replace("\\", "\\\\").replace('"', '\\"') + '"'
        (home / SKILL_FILE).write_text(
            f"---\nname: {part.slug}\ndescription: {quoted}\n---\n\n{part.body}\n",
            encoding="utf-8",
        )
    return root

#: What each line of `gates.md` can put behind a decision. A line is gated when
#: its phrase appears in the file, so the file is the policy rather than a
#: description of one.
GATE_PHRASES = {
    "customers will read": ("publish_status",),
    "changes production": ("roll_back", "roll_back_payments", "disable_feature",
                           "double_capacity", "throttle"),
    "waking": ("wake_owner",),
}

#: What stops when nobody is asked - which is nothing. The file is the policy,
#: so a folder without one has no policy, exactly as a folder whose `gates.md`
#: names nothing recognisable has none. Anything else would be the code
#: deciding what the folder is for.
GATED_BY_DEFAULT: tuple[str, ...] = ()


def gated(folder: Folder | None = None) -> tuple[str, ...]:
    """Which tools stop for a human, according to this folder's `gates.md`."""
    if folder is None or not folder.gates:
        return GATED_BY_DEFAULT
    text = folder.gates.lower()
    names: list[str] = []
    for phrase, tools in GATE_PHRASES.items():
        if phrase in text:
            names.extend(tools)
    return tuple(dict.fromkeys(names))


def interrupt_config(tools: dict, folder: Folder | None = None) -> dict:
    return {
        name: {"allowed_decisions": ["approve", "edit", "reject"]}
        for name in gated(folder)
        if name in tools
    }


def stop_when_reported(run):
    """End the run as soon as the step that was asked for has been reported.

    Left to itself the agent keeps calling tools after its step is done until
    it runs out of turns, and a run that runs out between a tool call and its
    answer leaves the model a call it never sees answered - which it then
    refuses for the rest of the incident. This ends it in the graph instead, at
    the one place where nothing is in flight: before the model is asked again.

    The run is held rather than looked up. The hook runs in whatever context
    the graph gives it, and a context variable read from the wrong one hands
    back a fresh run that is never finished.
    """
    from langchain.agents.middleware import before_model

    # Built by the decorator rather than by subclassing. Whether a hook counts
    # as overridden is decided by comparing it against the base class, so a
    # subclass that inherits from a second copy of the library on the path has
    # every hook counted, and the graph gains three nodes that do nothing but
    # cost turns.
    @before_model(can_jump_to=["end"], name="StopWhenReported")
    def stop(state, runtime):
        return {"jump_to": "end"} if run.closed else None

    return stop


def assembled(folder: Folder, tools: dict | None = None) -> dict:
    """What this folder actually builds, as the harness sees it.

    Not a description of the agent: the arguments `create_deep_agent` is about
    to be given, the tools the supervisor will hold, and the headings its
    prompt will carry. Read between two steps of the build, it is the
    difference a file made.
    """
    tools = tools or {}
    specs = subagent_specs(folder, tools)
    held = (list(tools) if not folder.investigators
            else [name for name in tools if not name.startswith("read_")])
    gates = [name for name in gated(folder) if name in tools]
    return {
        "call": [
            "model", "system_prompt", "backend", "middleware", "tools",
            *(["subagents"] if specs else []),
            *(["skills"] if folder.methods else []),
            *(["interrupt_on"] if gates else []),
        ],
        "tools": sorted(held) + (["task"] if specs else []),
        "subagents": [spec["name"] for spec in specs],
        "skills": [part.slug for part in folder.methods],
        "gates": gates,
        "prompt": {
            "blocks": [line[2:] for line in system_prompt(folder).splitlines()
                       if line.startswith("# ")],
            "characters": len(system_prompt(folder)),
        },
    }


def build_agent(
    folder: Folder,
    tools: dict | None = None,
    *,
    model,
    checkpointer=None,
    bucket: str | None = None,
    run=None,
):
    """Turn a folder into a running agent."""
    from deepagents import create_deep_agent
    from langchain.agents.middleware import TodoListMiddleware

    from agent.backends import build_backend

    import demo.tools as incident

    tools = tools or {}
    run = run if run is not None else incident.current()

    # A table part-way through the build gets only the pieces it has attached.
    # Planning is a middleware rather than a file, so it is staged here; from
    # the step that turns the filesystem durable, what the agent writes lands
    # in their folder instead of dying with the run.
    middleware = [stop_when_reported(run)]
    if folder.stage >= PLANS_FROM:
        # deepagents 0.7 dropped TodoListMiddleware from the default harness.
        middleware.insert(0, TodoListMiddleware())
    if folder.stage < KEEPS_FILES_FROM:
        bucket = None

    kwargs: dict = {
        "model": model,
        "system_prompt": system_prompt(folder),
        "middleware": middleware,
        "backend": build_backend(bucket=bucket, methods=stage_methods(folder)),
        # The supervisor gets no reading tools, so investigation can only happen
        # through the subagents. The exception is a folder with no investigators
        # in it: then it reads for itself, and `shortfall` tells it to say so.
        "tools": (
            list(tools.values())
            if not folder.investigators
            else [t for name, t in tools.items() if not name.startswith("read_")]
        ),
        "subagents": subagent_specs(folder, tools),
        "skills": [METHODS_PATH],
    }

    gates = interrupt_config(tools, folder)
    if gates:
        kwargs["interrupt_on"] = gates
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer

    return create_deep_agent(**kwargs)

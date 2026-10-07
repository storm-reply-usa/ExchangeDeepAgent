"""Check a participant's folder before it reaches the agent.

Twenty people will write whatever they like into these files. Most of what they
write is fine; the failures that matter are the silent ones, where deepagents
drops a malformed skill without a word and the participant concludes the
session is broken rather than that their frontmatter is.

Nothing raises. A folder with problems still builds an agent; the problems are
returned for a caller to report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agent.build import Folder, read_part

#: deepagents rejects a skill whose name breaks this, and says nothing.
SKILL_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESCRIPTION = 1024

#: Past this, one file starts crowding out the incident itself.
FAT_FILE_LINES = 60


@dataclass(frozen=True)
class Problem:
    path: str
    what: str
    fix: str
    fatal: bool = False


def _check_part(path: Path, kind: str) -> list[Problem]:
    rel = str(path).split("your-config/")[-1]
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [Problem(rel, "this file is not text", "save it as plain UTF-8 text", fatal=True)]
    if not raw.strip():
        return [Problem(rel, "this file is empty", f"write the {kind} in it, or delete the file")]

    part = read_part(path)
    problems: list[Problem] = []

    if not raw.lstrip().startswith("---"):
        problems.append(
            Problem(
                rel,
                "no frontmatter, so this is ignored entirely",
                "start the file with a --- block holding name: and description:",
                fatal=True,
            )
        )
        return problems

    if not part.description:
        problems.append(
            Problem(
                rel,
                "no description, so this is ignored entirely",
                "add description: to the frontmatter - it is what tells the agent when to use this",
                fatal=True,
            )
        )
    elif len(part.description) > MAX_DESCRIPTION:
        problems.append(Problem(rel, "the description is too long and gets cut off",
                                f"keep it under {MAX_DESCRIPTION} characters"))

    if len(part.name) > MAX_NAME:
        problems.append(Problem(rel, "the name is too long", f"keep it under {MAX_NAME} characters",
                                fatal=True))
    elif not SKILL_NAME.match(part.name):
        problems.append(
            Problem(
                rel,
                f"the name {part.name!r} is not allowed, so this is ignored entirely",
                "use lowercase letters, numbers and single hyphens: check-the-money",
                fatal=True,
            )
        )

    if not part.body.strip():
        problems.append(Problem(rel, "there is nothing below the frontmatter",
                                f"write what this {kind} actually does"))

    if len(raw.splitlines()) > FAT_FILE_LINES:
        problems.append(Problem(rel, f"this is {len(raw.splitlines())} lines, which crowds out the incident",
                                f"keep it under {FAT_FILE_LINES} lines"))
    return problems


def check(root: Path) -> list[Problem]:
    """Everything wrong with a folder, worst first."""
    root = Path(root)
    problems: list[Problem] = []

    if not root.is_dir():
        return [Problem(str(root), "there is no agent folder here",
                        "run this from the repository root", fatal=True)]

    investigators = sorted(root.glob("investigators/*.md"))
    methods = sorted(root.glob("methods/*.md"))

    if not investigators:
        problems.append(
            Problem(
                "investigators/",
                "there is nobody to investigate, so the agent can only guess",
                "put at least one .md file back in investigators/",
                fatal=True,
            )
        )

    for path in investigators:
        problems += _check_part(path, "investigator")
    for path in methods:
        problems += _check_part(path, "method")

    readable = [p for p in (*investigators, *methods)
                if not any(x.fatal and x.what.startswith("this file is not text")
                           for x in problems if x.path.endswith(p.name))]
    names = [read_part(p).name for p in readable]
    for name in sorted({n for n in names if names.count(n) > 1}):
        problems.append(Problem("your-config/", f"two files both call themselves {name!r}",
                                "give each one its own name", fatal=True))

    doctrine = root / "doctrine.md"
    if doctrine.exists() and len(doctrine.read_text(encoding="utf-8").splitlines()) > FAT_FILE_LINES:
        problems.append(Problem("doctrine.md", "the doctrine is long enough to crowd out the incident",
                                f"keep it under {FAT_FILE_LINES} lines"))

    return sorted(problems, key=lambda p: not p.fatal)


def report(problems: list[Problem]) -> str:
    """The problems, as a participant should read them."""
    if not problems:
        return ""
    lines = []
    for p in problems:
        mark = "will not work" if p.fatal else "worth fixing"
        lines.append(f"  {p.path}  —  {p.what}")
        lines.append(f"  {'':<{len(p.path)}}     {mark}: {p.fix}")
    return "\n".join(lines)


def degraded(folder: Folder) -> list[str]:
    """What the agent is missing, phrased as the agent would say it."""
    missing = []
    if not folder.investigators:
        missing.append("I have nobody to investigate with, so anything I say about the cause is a guess.")
    if not folder.doctrine:
        missing.append("I have no doctrine, so when two things conflict I am choosing on my own.")
    if not folder.methods:
        missing.append("I have no methods, so my numbers come from judgement rather than a procedure.")
    return missing

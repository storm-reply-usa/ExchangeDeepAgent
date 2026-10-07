"""One run, turned into the events the page draws.

LangGraph reports what it is doing in several parallel streams. This module
reads all of them at once and flattens them into the single vocabulary in
`agent.events`, so the page has one kind of thing to handle.

Each investigator runs inside its own LangGraph namespace, and that namespace is
what keeps its tool calls, its tokens and its finding in its own lane on screen.
The orchestrator's lane is `root`.
"""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk

#: `tasks` carries tool calls with their arguments already parsed, `messages`
#: carries model output as it is produced, and `custom` carries what the
#: incident's own tools say about themselves.
#:
#: Tool arguments are read from `tasks` rather than from the token stream: a
#: provider streams them as partial JSON, so a half-arrived call parses as
#: nothing useful.
MODES = ["tasks", "messages", "custom"]

#: Without this the stream reports the orchestrator only, and every
#: investigator works invisibly.
SUBGRAPHS = True

#: The flat `{type, ns, data}` shape. Version 1 yields ragged tuples instead.
VERSION = "v2"

#: deepagents' own file tools. The path one of these is given is the file that
#: lights up in the rail.
FILE_TOOLS = {"read_file", "edit_file", "write_file"}
#: Reading a file and writing one are the same event to the rail, and two
#: different things to somebody asking what a step was based on.
WROTE = {"edit_file", "write_file"}

#: Calls that report the work rather than do any. Each already arrives as the
#: event it stands for, and showing the call as well says everything twice.
QUIET = {"report_step", "remember"}

#: A path that names a file in the folder, however it was written.
ROUTES = ("your-config/",)

#: Methods are laid out as skills for the agent - one directory each, holding a
#: `SKILL.md` - and shown to the participant as the flat file they edited.
METHODS_ROUTE = "/methods/"
SKILL_FILE = "/SKILL.md"


def lane_of(namespace) -> str:
    """Which lane an event belongs to.

    Only the first element: an investigator that calls a tool of its own nests
    deeper, and those events belong in the investigator's lane rather than in a
    lane of their own.
    """
    return namespace[0] if namespace else "root"


def in_folder(path: str) -> str:
    """A path as the rail shows it, or empty if it is not one of the folder's."""
    if path.startswith(METHODS_ROUTE) and path.endswith(SKILL_FILE):
        return "methods/" + path[len(METHODS_ROUTE):-len(SKILL_FILE)] + ".md"
    for route in ROUTES:
        if route in path:
            return path.split(route, 1)[1]
    # Anything the agent writes at the root of its own filesystem is the
    # participant's folder as far as they are concerned - the post-mortem it
    # files at the close of the incident arrives this way.
    bare = path.lstrip("/")
    if bare and "/" not in bare and bare.endswith(".md") and bare != SKILL_FILE.strip("/"):
        return bare
    return ""


def _from_tasks(data: dict, lane: str, said: dict):
    """Tool calls starting and finishing.

    Start and finish arrive as the same kind of event and are told apart by
    their keys: only a finish carries a result.
    """
    if "result" in data or "error" in data:
        if data.get("name") != "tools":
            return
        # An investigator's lane is named after the task that carries it, so a
        # task finishing at the parent is that lane closing. Whatever it was
        # saying when it closed is its answer.
        closing = said.pop(f"tools:{data.get('id')}", None)
        if closing:
            yield {"kind": "finding", "lane": f"tools:{data.get('id')}",
                   "finding": "".join(closing).strip()}
        yield {"kind": "tool_done", "lane": lane, "task": data.get("id"),
               "failed": bool(data.get("error"))}
        return
    if data.get("name") != "tools":
        return

    # Anything said before a tool call was thinking aloud, not an answer.
    said.pop(lane, None)

    by = (data.get("metadata") or {}).get("lc_agent_name")
    for call in data.get("input") or []:
        name, args = call.get("name"), call.get("args") or {}
        if name == "task":
            # The investigator's own lane is named after the task that carries
            # it, so the page can file its work under it before it starts.
            yield {"kind": "dispatch", "lane": lane,
                   "opens": f"tools:{data['id']}",
                   "investigator": args.get("subagent_type", "?"),
                   "brief": args.get("description", "")}
            continue
        if name == "write_todos":
            yield {"kind": "plan", "lane": lane, "todos": args.get("todos") or []}
            continue
        if name in QUIET:
            continue
        yield {"kind": "tool", "lane": lane, "name": name, "args": args, "by": by}
        if name in FILE_TOOLS:
            inside = in_folder(str(args.get("file_path", "")))
            if inside:
                yield {"kind": "file", "lane": lane, "path": inside, "by": by,
                       "how": "wrote" if name in WROTE else "read"}


def _from_messages(data, lane: str):
    """Model output, as it is produced."""
    message, meta = data
    if isinstance(message, AIMessageChunk) and message.text:
        yield {"kind": "token", "lane": lane, "text": message.text,
               "by": meta.get("lc_agent_name")}


async def events(graph, payload, config):
    """Work one segment, yielding events as the agent produces them."""
    #: What each lane is currently saying, kept so an investigator's answer can
    #: be taken from the lane rather than from a tool call it may not make.
    said: dict[str, list[str]] = {}

    async for part in graph.astream(payload, config, stream_mode=MODES,
                                    subgraphs=SUBGRAPHS, version=VERSION):
        kind, lane, data = part["type"], lane_of(part["ns"]), part["data"]
        if kind == "tasks":
            for event in _from_tasks(data, lane, said):
                yield event
        elif kind == "messages":
            for event in _from_messages(data, lane):
                if event["kind"] == "token" and lane != "root":
                    said.setdefault(lane, []).append(event["text"])
                yield event
        elif kind == "custom":
            # Already in the vocabulary: the incident's tools emit it directly.
            yield {**data, "lane": lane}

# Your configuration

This folder is the agent. What it cares about, which procedures it follows, who it
sends out, where it has to stop and what it remembers are thirteen markdown files,
and nothing else. Change one, press **Run**, and a different agent works the same
incident.

| File | What it decides | Change it when |
|---|---|---|
| `doctrine.md` | what matters most when two good options conflict | it optimised for the wrong thing |
| `gates.md` | which calls stop and wait for a person | it decided something you would want to decide |
| `investigators/` | who is sent to read what, each in a context of its own | nobody checked something you would have checked |
| `methods/` | how it assesses, whom it sends, how it weighs, when it is over, what it writes | a judgement was wrong, or a number was |
| `memory/` | what it already knows when it starts | it does not know something it should |

Every step in the dossier names the file it followed. If a step looks wrong, that
name is where to go.

`investigators/` and `methods/` need a `name` and a `description` in their
frontmatter; the name is lowercase with hyphens. Without them the file is skipped
in silence. `doctrine.md`, `gates.md` and `memory/` are plain markdown. Lines
starting with `>` are notes to you and are never shown to the agent.

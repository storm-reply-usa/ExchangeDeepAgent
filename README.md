# On-call agent

3:14 a.m. 38% of checkouts are failing. Payments clear, order confirmations do
not, so money is being taken against orders nobody recorded.

The on-call engineer is a deep agent, built with `create_deep_agent()` from
[deepagents](https://docs.langchain.com/oss/python/deepagents/overview) and
running on AWS. Nothing about how it behaves is in code: what it cares about,
who it sends, how it decides and where it stops are thirteen markdown files in
`your-config/`. Change one, run it again, and a different agent works the same
incident.

## See it work

**[Open the workbench](https://qxlwizkkj3jj63oqkotjtcmesm0yoknp.lambda-url.us-east-1.on.aws/?table=1)**
— the agent in this repository, running the incident below. Nothing to install
and nothing to sign in to. Build it one capability at a time and watch each
one land in AWS, or go straight to the demo and watch it work the night.

The first time you land on either screen a short walk points at each region
and says what it is for; **Where to look** brings it back.

## Start

Build it first, or skip to the agent as it stands. Either way, press **Run** and
it works the incident in seven steps - assess, notify, investigate, decide,
verify, close, remember - and stops to ask you before anything reaches a
customer or touches production.

## The folder is the agent

```
your-config/
├── doctrine.md          what matters most, in order
├── gates.md             what it may not decide alone
├── investigators/       who is sent to read what, each in a context of its own
├── methods/             how it assesses, whom it sends, when it is over, what it writes
└── memory/              what it already knows when it starts
```

`agent/build.py` turns that folder into one call: the doctrine, the gates and
the memory become blocks of the system prompt, each method becomes a skill, each
investigator becomes a subagent holding one tool. A file the code has never
heard of still works: a new investigator is handed every reading tool.

Every step in the dossier names the file it followed.

## Three changes

At the foot of a run the page offers three, one click each.

| | Open | Watch |
|---|---|---|
| A call you would make differently | `doctrine.md` - swap two lines | a different action, and what it costs |
| Nobody asked it to | `methods/resolved.md` - raise the bar | it refuses 0.90%, goes back, and finds the fraud check the 02:58 release tightened |
| Nobody asked it to | `investigators/payments.md` - a new file | a fifth lane, with a number nobody was bringing |

Then change whatever you like. `gates.md` is read literally: take a line out and
the agent makes those calls alone. `memory/past-incidents.md` is read before the
first step: put something true about tonight in it and watch the first step
change. `investigators/` takes any filename.

## How it is built

[TOUR.md](TOUR.md) walks the six steps of the build, one promise of the
deepagents documentation each: the line that switches each capability on, and
the code in this repository that computes that line from the folder.

## When a file is ignored

Files in `investigators/` and `methods/` need `name` and `description` in their
frontmatter, and a name in lowercase with single hyphens. Get one wrong and
deepagents skips the file **without saying so**. The page names a file that is
not being read at the top of the run, with what to fix. `doctrine.md`,
`gates.md` and `memory/` are plain markdown.

## Running it yourself

```bash
cp .env.example .env          # AWS credentials with Bedrock access in us-east-1
uv sync
uv run pytest demo/tests -q
uv run python -m demo.lab.cli                 # work the incident at a terminal
```

That builds the agent from `your-config/` on disk and runs it against the
simulated night, with no AWS beyond Bedrock. To put it on AgentCore instead,
see [CONTRIBUTING.md](CONTRIBUTING.md).

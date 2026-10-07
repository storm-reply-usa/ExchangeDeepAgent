# Working on this repository

An on-call incident agent, built with `create_deep_agent()` and running on AWS.
Its behaviour is not in the code: what it cares about, who it sends, how it
decides and where it stops are thirteen markdown files in `your-config/`.

```
agent/          turns that folder into a deepagents agent
your-config/    the thirteen files that are the agent
demo/           the worked example: the incident, its tools and its rubric
                  tools.py    what the agent can do about the night
                  simulator/  the night itself, deterministic, and how it is
                              published into CloudWatch and DynamoDB
                  lab/        process.py drives the seven stations of a run;
                              stream.py turns a run into events
                  evals/      scorecard.py grades a run from its trajectory
                  tests/
deploy/         how it reaches AWS
                  runtime/    a build directory: deploy_agent.sh rebuilds it
                              from source every time. Only main.py,
                              pyproject.toml and policies/ are kept
                  agentcore/  the AgentCore CLI's project
                  scripts/    deploy_agent.sh, preflight.py
TOUR.md         the six steps, generated from the build
```

## The example is an example

`demo/` is one problem worked end to end, not scaffolding. The agent reads what
it knows from the folder, sends specialists that cannot see each other's
evidence, reads a real status page through a remote browser, stops before it
touches production, and remembers one line for next time.

To point it at your own problem, replace the tools in `demo/tools.py` and the
files in `your-config/`. Nothing in `agent/` knows what an incident is.

## Running it

```bash
uv sync
cp .env.example .env            # then your LangSmith key
uv run pytest demo/tests -q
```

`AWS_REGION` is `us-east-1` and there are no exceptions. With `LABCAMP_BUCKET`
unset the agent reads the folder from disk and the night from the simulator in
process; with it set, both come from S3 and from CloudWatch.

## Deploying the agent

```bash
./deploy/scripts/deploy_agent.sh
```

It rebuilds `deploy/runtime/` from source first, so what is deployed cannot be
stale, compiles every file before it uploads, and then hands over to
`agentcore deploy`.

## What the tests are for

They are not coverage. Each one holds a claim the product makes: that the
folder really configures the agent, that a hostile folder still runs, that what
the tour quotes is still in the file it quotes from, and that a reader falls
back to the simulator rather than inventing a number. A failing test here means
something the product says is no longer true.

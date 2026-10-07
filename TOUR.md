# Deep Agents on AWS

We build an **on-call incident agent**: it reads what it knows from a folder,
sends specialists that cannot see each other's evidence, reads a real status
page in a remote browser, stops before it touches production, remembers one
line from last night, and is scored the same way on every run.

It is built in six steps. Each one keeps one of the promises the deepagents
documentation opens with, shows the line of `create_deep_agent()` that turns it
on, and shows the code in this repository that computes that line from a folder
of markdown. Nothing here is a fixture: the agent you have at step six is the
one that works the incident.

**Region:** `us-east-1` · **Model:** Claude Haiku 4.5, through Bedrock · **Configuration:** thirteen markdown
files, and nothing else

| Step | Capability | AWS + LangChain | Lifecycle |
|---|---|---|---|
| 1 | **The loop, and where it acts** | S3 · Bedrock · CloudWatch · DynamoDB · deepagents | Build |
| 2 | **What it knows** | S3 · deepagents | Build |
| 3 | **Who it sends** | S3 · deepagents | Build |
| 4 | **A reader on the open web** | AgentCore Browser · S3 | Build |
| 5 | **Where it has to stop** | LangGraph · deepagents | Build |
| 6 | **Where it runs, and how it scores** | AgentCore Runtime · CloudWatch · LangSmith | Deploy · Monitor |

### Execution environment

| What it is | What it means here |
|---|---|
| **Virtual filesystem** (step 1) | working state kept in files, and evicted there when it gets large |
| **Browsing** (step 4) | a real page, fetched by a browser that runs outside this process |

### Context management

| What it is | What it means here |
|---|---|
| **Skills** (step 2) | procedures read when the task matches, not on every run |
| **Long-term memory** (step 2) | what one run leaves for the next one |

### Delegation

| What it is | What it means here |
|---|---|
| **Task planning** (step 3) | a list of what it must settle, which it writes and then revises |
| **Subagents** (step 3) | separate contexts, so one investigator cannot colour another |

### Steering

| What it is | What it means here |
|---|---|
| **Human in the loop** (step 5) | the graph pauses before the call and survives until somebody answers |

### Outside the call

| What it is | What it means here |
|---|---|
| **Observability and evaluation** (step 6) | what it is doing while it does it, and the same verdicts on every run |


Every capability below is switched on once, in the step that names it, and used for the rest of the run.

---

## Step 1 — The loop, and where it acts · **Build**

One call assembles the agent, and every file it touches has an address.

- **S3** — your folder, behind an access point of its own
- **Bedrock** — the model your agent thinks with: Claude Haiku 4.5, called through one AWS API
- **CloudWatch** — the checkout error rate and the log of the night, which the agent reads rather than is told
- **DynamoDB** — the deploy history of the night, one row per release and per action
- **deepagents** — create_deep_agent(), and the middleware it brings

The backend decides what a path means, and the longest matching prefix wins: one prefix at a durable store, everything else in state that dies with the run. You never ask for offloading - a tool result over 20,000 tokens is written to the backend and replaced by its path and first ten lines. The filesystem tools come with the call: `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, and `execute` when the backend is a sandbox. Summarization and prompt caching are on before you ask; everything in the steps that follow is one more argument.

| Capability | Middleware | How it arrives |
|---|---|---|
| Files | `FilesystemMiddleware` | built in, then routed with `backend=` |
| Skills | `SkillsMiddleware` | `skills=[...]`, step 2 |
| Planning | `TodoListMiddleware` | `middleware=[...]`, step 3 - opt-in since 0.7 |
| Subagents | `SubAgentMiddleware` | built in; `subagents=[...]` names your own, step 3 |
| Interrupts | `HumanInTheLoopMiddleware` | `interrupt_on={...}`, step 5 |

### What switches it on

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=ChatBedrockConverse(model="us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    tools=[read_metrics, read_logs, read_deploys, roll_back, publish_status, ...],
    system_prompt=system_prompt(folder),       # assembled from the files in S3
    backend=CompositeBackend(                  # state by default...
        default=StateBackend(),
        routes={"/durable/": StoreBackend(store=s3_store,   # ...your access point
                    namespace=lambda _rt: ("labcamp", "filesystem"))},
    ),
)
```

One call, and the filesystem tools, summarization and prompt caching are already there. `backend=` decides where the files live. With no investigators yet, the supervisor holds every reader itself.

### Do it yourself on AWS

**Give it something real to read**

```bash
aws cloudwatch put-metric-data --namespace labcamp/table-1 \
    --metric-data MetricName=checkout_error_rate,Value=38,Unit=Percent,Dimensions=[{Name=table,Value=1}]
aws logs create-log-group --log-group-name /labcamp/table-1/checkout
aws logs put-log-events --log-group-name /labcamp/table-1/checkout --log-stream-name night \
    --log-events timestamp=$NOW_MS,message="ERROR OrderConfirmation.render - template 'confirm_v2' is missing partial 'summary'"
aws dynamodb put-item --table-name labcamp-deploys \
    --item '{"table": {"S": "1"}, "at": {"N": "151"}, "service": {"S": "order confirmations"}}'
```

```yaml
Deploys:
  Type: AWS::DynamoDB::Table
  Properties:
    TableName: labcamp-deploys
    BillingMode: PAY_PER_REQUEST
    AttributeDefinitions:
      - {AttributeName: table, AttributeType: S}
      - {AttributeName: at, AttributeType: N}
    KeySchema:
      - {AttributeName: table, KeyType: HASH}
      - {AttributeName: at, KeyType: RANGE}
```

In production these are your own services and nobody writes them by hand; here the incident is written into real ones so the agent reads CloudWatch and DynamoDB the way it would read yours. The runtime's role gets GetMetricStatistics, GetLogEvents and Query on them, and nothing that writes.

**Put the folder in a bucket, behind an access point**

```bash
aws s3 cp your-config/ s3://$BUCKET/tables/1/your-config/ --recursive
aws s3control create-access-point --account-id $ACCOUNT \
    --name labcamp-table-1 --bucket $BUCKET
```

```yaml
TableAccessPoint:
  Type: AWS::S3::AccessPoint
  Properties:
    Bucket: !Ref AgentFiles
    Name: labcamp-table-1
```

The agent's role needs s3:GetObject and s3:PutObject on the access point, not on the bucket: that is what keeps one table out of another's files. Claude on Bedrock is enabled once per account, under Model access in the Bedrock console; it is not a resource you create.

### The code behind it

`agent/build.py`

```python
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
```

The whole harness is one call. Everything in it is computed from the folder - with nothing in it but a doctrine, the supervisor reads for itself and says so.

`agent/backends.py`

```python
def build_backend(bucket: str | None = None,
                  methods: str | None = None) -> CompositeBackend:
    """Compose the routes described in the module docstring.

    Nothing outside these routes is reachable. Every agent here, the
    orchestrator and each investigator alike, gets file tools it did not ask
    for, and a route to the checkout would let one of them read the simulator
    that decides the answers.
    """
    routes: dict[str, object] = {}
    if methods:
        routes[METHODS_ROUTE] = FilesystemBackend(root_dir=str(methods), virtual_mode=True)
    if bucket:
        routes[DURABLE_ROUTE] = StoreBackend(
            store=S3Store(bucket=bucket, prefix=DURABLE_PREFIX),
            namespace=lambda _rt: ("labcamp", "filesystem"),
        )
        routes[MEMORIES_ROUTE] = StoreBackend(
            store=S3Store(bucket=bucket, prefix=MEMORIES_PREFIX),
            namespace=customer_namespace,
        )
    return CompositeBackend(default=StateBackend(), routes=routes)
```

Which backend the agent gets, and therefore what a path means. Your folder is the durable half; everything else dies with the run.

`demo/simulator/published.py`

```python
def seed(clients: "Clients", table: str, plan: list[Step] | None = None,
         anchor: dt.datetime | None = None) -> dict:
    """Everything a table's readers need, written on a fresh anchor. Called at
    step 1's Deploy and by the reset, and safe to call again: a new anchor
    lays the night onto the clock afresh, and the old points expire."""
    anchor = anchor or new_anchor()
    # The anchor last: a reader with no anchor falls back to the simulator
    # whole, so a seeding that failed half way leaves nothing that reads as
    # half a night.
    written = {
        "anchor": anchor.isoformat(),
        "metrics": publish_metrics(clients.cloudwatch, table, anchor, plan),
        "logs": publish_logs(clients.logs, table, anchor, plan),
        "deploys": publish_deploys(clients.dynamodb, table, plan),
    }
    publish_anchor(clients.dynamodb, table, anchor)
```

The night, written into CloudWatch, a log group and DynamoDB under your table's name, so what the readers return was read and not computed beside them.

**Adds to the folder:** `doctrine.md`

**What to watch.** No lanes and no plan: one context reads everything itself, and its first line says it is working without investigators.

---

## Step 2 — What it knows · **Build**

What the agent knows is a file: opened when it matches, or carried on every run.

- **S3** — the same folder, now holding methods and a memory
- **deepagents** — SkillsMiddleware, and the prompt block the memory becomes

Skills load by **progressive disclosure**: at startup the agent sees only each `SKILL.md`'s `name` and `description`, and reads the body when a task matches. The name must be lowercase with hyphens and match its directory, or the skill is skipped without a word - which is why `stage_methods` writes that frontmatter for you from the filename. Memory is the opposite trade: always in the prompt, so it costs tokens on every run and can shape the first move. Which of the two a thing belongs in follows from **when** it has to be in mind.

| Capability | Middleware | How it arrives |
|---|---|---|
| Skills | `SkillsMiddleware` | `skills=["/methods/"]`, read only when the task matches |
| Memory | `a prompt block` | or `memory=[...]` routed at a store, once it outgrows a file |

### What switches it on

```python
agent = create_deep_agent(
    ...,                                       # the call from step 1, unchanged
    skills=["/methods/"],                      # SkillsMiddleware, over your folder
)

# Memory, either way round:
blocks.append("# From past incidents\n\n" + folder.memory)   # a block of the prompt
#     create_deep_agent(..., memory=["/memories/AGENTS.md"])  # or routed to a store
```

One argument over the same folder, and `/methods/` has to be routed there or `skills` finds nothing. The memory stays a prompt block while a person is still deciding what is worth keeping; `memory=` routes it at a store once it outgrows that.

### Do it yourself on AWS

**Nothing to stand up: copy the files**

```bash
aws s3 cp your-config/methods/ s3://$BUCKET/tables/1/your-config/methods/ --recursive
aws s3 cp your-config/memory/  s3://$BUCKET/tables/1/your-config/memory/  --recursive
```

A configuration step is files in the bucket the agent already reads. No template changes, no new permission: the runtime picks them up on its next start.

### The code behind it

`agent/build.py`

```python
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
```

A skill is a directory holding a SKILL.md. The folder keeps one flat file per method - that is what a person wants to edit - and these lines reconcile the two shapes.

`agent/build.py`

```python
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
```

The prompt, block by block, from the folder. The memory is one of them, so it is in front of the agent before it takes a step.

`demo/tools.py`

```python
def remember(line: str, saying: str = "") -> str:
    """Append one line to your memory, to be read at the start of the next incident.

    `worth-remembering` says what earns a line here. `saying` declares the
    remember step in the same call.
    """
    run = current()
    if run.closed:
        return CLOSED
    if not line.strip():
        return "nothing to remember - say the one line, or do not call this"
    _note("remember", saying or line)

    entry = (f"\n## Incident #{INCIDENT['id']} - {INCIDENT['symptom']}\n\n"
             f"{line.strip()}\n\n---\n")
    folder = Path(run.folder) if run.folder else SHIPPED_METHODS.parent
    path = folder / MEMORY_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as out:
            out.write(entry)
    except OSError as problem:
        return f"could not write it down: {problem}"
```

Writing a memory is a tool. The line is appended to the folder the agent was built from, and put back into your folder in S3 when the run ends.

**Adds to the folder:** `methods/severity.md` · `methods/who-to-send.md` · `methods/weighing-evidence.md` · `methods/resolved.md` · `methods/post-mortem.md` · `methods/worth-remembering.md` · `memory/past-incidents.md`

**What to watch.** Each step carries a READ chip naming the method it followed, the file lights up in the rail, and the first line cites a night from memory.

---

## Step 3 — Who it sends · **Build**

Everyone it can send is a file, and what it means to settle is a list it writes.

- **S3** — the same folder, now holding investigators
- **deepagents** — TodoListMiddleware and SubAgentMiddleware

`TodoListMiddleware` adds one tool, `write_todos`, and a list in the agent's state that is **replaced whole** on every call rather than appended to - which is what lets the agent revise a plan instead of only growing it. The list sits in the message history, so the agent is handed its own plan again on every turn. A subagent runs its own graph with its own message list and hands back a single result, so no investigator's reading colours another's; sent in one message, they run at once. Alongside yours the harness adds a `general-purpose` subagent with the supervisor's own tools, unless you turn it off.

| Capability | Middleware | How it arrives |
|---|---|---|
| Planning | `TodoListMiddleware` | `middleware=[TodoListMiddleware()]` - opt-in since 0.7 |
| Specialists | `SubAgentMiddleware` | `subagents=[...]`, one context each, run at once |

### What switches it on

```python
from langchain.agents.middleware import TodoListMiddleware

agent = create_deep_agent(
    ...,
    tools=[roll_back, publish_status, verify, ...],   # the readers are gone
    subagents=[metrics, logs, deploys],              # SubAgentMiddleware
    middleware=[TodoListMiddleware()],               # 0.7 dropped this from the default stack
)
```

Note what left `tools=`: every `read_*` tool now belongs to an investigator, because a supervisor that can read the evidence itself will, and never send anybody. Planning is passed as middleware.

### Do it yourself on AWS

**Nothing to stand up: copy the files**

```bash
aws s3 cp your-config/investigators/ \
    s3://$BUCKET/tables/1/your-config/investigators/ --recursive
```

Each investigator becomes a subagent on the next start; the tools it holds are decided by its filename in agent/build.py, so a new file needs no IAM change and no deploy.

### The code behind it

`agent/build.py`

```python
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
```

One subagent per file. The filename decides which tool it is handed, so an investigator can only read what it was hired to read.

`agent/build.py`

```python
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
```

The supervisor keeps the actions and loses the readers the moment the folder holds an investigator.

**Adds to the folder:** `investigators/metrics.md` · `investigators/logs.md` · `investigators/deploys.md`

**What to watch.** A plan box before anybody is sent, three lanes back in the time of the slowest, and one INVESTIGATE line made from their three answers.

---

## Step 4 — A reader on the open web · **Build**

Reading the open web is a capability you rent, not one you build.

- **AgentCore Browser** — a real Chrome that AWS runs for you, thrown away when the agent has finished with it
- **S3** — the provider's status page, served from public-docs/

The session runs outside your process. `start_browser_session` on `aws.browser.v1` returns a session id; you sign a websocket to it and speak the Chrome DevTools Protocol directly - create a target, navigate, wait for the document to settle, read its text. The session carries a timeout, five minutes here, and is stopped when the read is done, so a page the agent visits can never touch the runtime. Starting one is a single API call and stopping it is another, which is why it can be a tool rather than a fixture.

### What switches it on

```python
@tool
def read_vendor_status() -> str:
    """What the payment provider's public status page says."""
    plane = boto3.client("bedrock-agentcore", region_name="us-east-1")
    got = plane.start_browser_session(browserIdentifier="aws.browser.v1",
                                      sessionTimeoutSeconds=300)
    ws_url, headers = signed_websocket(got["browserIdentifier"], got["sessionId"])
    try:
        return read_over_cdp(ws_url, headers, status_page_url())   # Chrome DevTools Protocol
    finally:
        plane.stop_browser_session(browserIdentifier=got["browserIdentifier"],
                                   sessionId=got["sessionId"])
```

The browser runs in AgentCore, not in your process: two API calls bracket a signed websocket, and the sandbox is destroyed when the session closes.

### Do it yourself on AWS

**Rent a browser session, or build a browser of your own**

```bash
aws bedrock-agentcore start-browser-session \
    --browser-identifier aws.browser.v1 --session-timeout-seconds 300
aws bedrock-agentcore stop-browser-session \
    --browser-identifier aws.browser.v1 --session-id $SESSION
```

```yaml
LabBrowser:
  Type: AWS::BedrockAgentCore::BrowserCustom
  Properties:
    Name: labcamp_browser
    NetworkConfiguration:
      NetworkMode: PUBLIC
    RecordingConfig:
      Enabled: true
      S3Location:
        Bucket: !Ref AgentFiles
        Prefix: recordings/
```

aws.browser.v1 is there in every account and needs only bedrock-agentcore:StartBrowserSession on the caller's role. A browser of your own is a resource: it can record every session to S3 and run inside your VPC.

### The code behind it

`agent/browsing.py`

```python
def page_text_with_receipt(url: str, *, timeout: float = 25.0) -> dict:
    """The page's text, and the receipt AgentCore gave for the session.

    A tool is called synchronously from inside the graph and the graph already
    runs an event loop, so the session is driven on a loop of its own in a
    thread of its own.
    """
    import concurrent.futures

    def work() -> dict:
        plane, receipt = _session(REGION)
        try:
            ws_url, headers = _ws_headers(REGION, receipt["browser"], receipt["session"])
            text = asyncio.run(_read(ws_url, headers, url, timeout))
        finally:
            try:
                plane.stop_browser_session(browserIdentifier=receipt["browser"],
                                           sessionId=receipt["session"])
            except Exception:
                pass
        return {**receipt, "text": text}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(work).result(timeout=timeout + 20)
```

The session, the signed websocket and the read, in the order the snippet shows them. The receipt is what the deploy panel printed.

`demo/tools.py`

```python
def read_vendor_status() -> str:
    """What the payment provider's public status page says."""
    run = current()
    run.called.add("read_vendor_status")
    if run.closed:
        return CLOSED
    run.advance(1)
    from agent import browsing

    try:
        url = browsing.status_page_url()
        text = browsing.page_text(url) if url else ""
    except Exception:
        text = ""
    return text or UNREACHABLE
```

The tool the vendor-status investigator holds. It is the only reader in the room that leaves the building.

**Adds to the folder:** `investigators/vendor-status.md`

**What to watch.** A fourth lane, the slowest of the four, with a link to the page it actually opened.

---

## Step 5 — Where it has to stop · **Build**

An interrupt is a place in the graph, not a prompt asking permission.

- **LangGraph** — the graph underneath the agent: it can be stopped before a step, written down, and carried on later from exactly there
- **deepagents** — HumanInTheLoopMiddleware, through interrupt_on={...}

The pause is a place in the graph, not an instruction to be careful. The model asks for a tool in the ordinary way; before the tool runs, the middleware looks at what was asked for, and if the name is one you listed it raises an **interrupt**. LangGraph writes the state down and waits - indefinitely, and across processes, which is why a `checkpointer` is required and not optional. You answer on the same `thread_id` and the run carries on from exactly there. Four answers exist: `approve` runs the call as proposed, `edit` changes its arguments first, `reject` skips it and tells the model why, and `respond` replies *as though it were the tool*. **Only `reject` denies.** On `approve` the model never learns it waited.

### What switches it on

```python
agent = create_deep_agent(
    ...,
    interrupt_on={                 # HumanInTheLoopMiddleware
        "publish_status": {"allowed_decisions": ["approve", "edit", "reject"]},
        "roll_back": {"allowed_decisions": ["approve", "edit", "reject"]},
    },
    checkpointer=saver,            # required, or there is nothing to resume
)
```

Nothing about the model or its prompt changes. Given a checkpointer, the run interrupts before the call, persists, and resumes with `Command(resume=...)`. `respond` is left out because it reads as the call having succeeded.

### Do it yourself on AWS

**Nothing to stand up: copy the file**

```bash
aws s3 cp your-config/gates.md s3://$BUCKET/tables/1/your-config/gates.md
```

The interrupt lives in the graph, not in AWS. What production needs is a checkpointer that outlives the process, so a run can wait for hours: a LangGraph saver on a database you already run, in place of the in-memory one this lab uses.

### The code behind it

`agent/build.py`

```python
GATE_PHRASES = {
    "customers will read": ("publish_status",),
    "changes production": ("roll_back", "roll_back_payments", "disable_feature",
                           "double_capacity", "throttle"),
    "waking": ("wake_owner",),
}
```

The file is the policy, not a description of one: a phrase appearing in gates.md puts a group of calls behind a human.

`agent/build.py`

```python
def interrupt_config(tools: dict, folder: Folder | None = None) -> dict:
    return {
        name: {"allowed_decisions": ["approve", "edit", "reject"]}
        for name in gated(folder)
        if name in tools
    }
```

Which becomes one argument, and the graph now has somewhere to stop. Nothing about the model changed.

**Adds to the folder:** `gates.md`

**What to watch.** It stops with the status update it wants to publish in front of you; the clock says waiting for you, and nothing moves until you answer.

---

> **Build → Deploy.** You have an agent that works. Everything that follows is about trusting it away from your laptop, which starts with a plain question: did the change you just made improve it, or do you only feel that it did?

## Step 6 — Where it runs, and how it scores · **Deploy · Monitor**

Once it is hosted, the only honest question is whether it got better.

- **AgentCore Runtime** — where your agent lives once it is deployed: its own machine, started when somebody calls it, sending its work back line by line as it happens
- **CloudWatch** — where AWS keeps what your agent did: every call it made, how long each took, in a place of your own you can open and read
- **LangSmith** — every model call, nested under the step that made it, with what it cost - and the rubric the run is graded against

The AgentCore SDK sees an async generator entrypoint and serves it as Server-Sent Events, which is why the page fills as the agent works. A **run** is one unit of work and a **trace** is every run belonging to one operation; `subgraphs=True` puts a namespace on each event saying which task produced it, which is the only reason a subagent's work is addressable at all. An evaluator is either a judge model, which grades subjective quality and costs a model call, or plain code - ours reads the trajectory in code and runs in microseconds. Its output is written as **feedback**, because feedback keys can be charted across runs and metadata cannot.

### What switches it on

```python
@app.entrypoint
async def invoke(payload, context):        # an async generator
    async for event in graph.astream(payload, config,
                                     stream_mode=["tasks", "messages", "custom"],
                                     subgraphs=True,   # without this the subagents are invisible
                                     version="v2"):
        yield event                        # AgentCore serves this as SSE

checks = scorecard.review(steps, plan, outcome, gates=gates, called=asked)   # a trajectory evaluator
for key, value in scorecard.as_feedback(checks).items():
    client.create_feedback(run_id, key=key, score=value)                     # LangSmith feedback
```

`subgraphs=True` is what makes the investigators show up at all, and the checks are written as **feedback** because LangSmith charts feedback keys and not metadata.

### Do it yourself on AWS

**A runtime of your own, its endpoint, and its logs**

```bash
aws bedrock-agentcore-control create-agent-runtime \
    --agent-runtime-name labcamp_table1 --role-arn $ROLE \
    --agent-runtime-artifact '{"codeConfiguration": {"code": {"s3": {"bucket": "'$BUCKET'", "prefix": "runtime/agent.zip"}}, "runtime": "PYTHON_3_12", "entryPoint": ["main.py"]}}' \
    --network-configuration '{"networkMode": "PUBLIC"}' \
    --environment-variables LANGSMITH_TRACING=true,LANGSMITH_API_KEY=$KEY
aws bedrock-agentcore-control create-agent-runtime-endpoint \
    --agent-runtime-id $RUNTIME_ID --name table1
aws logs put-delivery-source --name $RUNTIME_ID-logs --log-type APPLICATION_LOGS \
    --resource-arn arn:aws:bedrock-agentcore:us-east-1:$ACCOUNT:runtime/$RUNTIME_ID
aws logs put-delivery-destination --name $RUNTIME_ID-cwl --delivery-destination-type CWL \
    --delivery-destination-configuration destinationResourceArn=$LOG_GROUP_ARN
aws logs create-delivery --delivery-source-name $RUNTIME_ID-logs \
    --delivery-destination-arn $DESTINATION_ARN
```

```yaml
AgentRuntime:
  Type: AWS::BedrockAgentCore::Runtime
  Properties:
    AgentRuntimeName: labcamp_table1
    RoleArn: !GetAtt AgentRole.Arn
    NetworkConfiguration:
      NetworkMode: PUBLIC
    AgentRuntimeArtifact:
      CodeConfiguration:
        Code:
          S3: {Bucket: !Ref AgentFiles, Prefix: runtime/agent.zip}
        Runtime: PYTHON_3_12
        EntryPoint: [main.py]
    EnvironmentVariables:
      LANGSMITH_TRACING: "true"
      LANGSMITH_API_KEY: !Ref LangSmithKey
AgentEndpoint:
  Type: AWS::BedrockAgentCore::RuntimeEndpoint
  Properties:
    AgentRuntimeId: !GetAtt AgentRuntime.AgentRuntimeId
    Name: table1
AgentLogs:
  Type: AWS::Logs::Delivery
  Properties:
    DeliverySourceName: !Ref AgentLogSource
    DeliveryDestinationArn: !GetAtt AgentLogDestination.Arn
```

The role the runtime assumes is the whole of its reach: Bedrock for the model, the access point for the folder, StartBrowserSession for the browser, and nothing else. LangSmith is two environment variables on the runtime; the trace needs no AWS resource. Everything here is us-east-1, and the model id's us. prefix is a cross-region profile that only resolves from a US region.

### The code behind it

`deploy/runtime/main.py`

```python
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
```

An async generator. AgentCore Runtime sees that and serves it as Server-Sent Events - which is why the page fills as it works instead of after.

`demo/lab/stream.py`

```python
MODES = ["tasks", "messages", "custom"]

#: Without this the stream reports the orchestrator only, and every
#: investigator works invisibly.
SUBGRAPHS = True

#: The flat `{type, ns, data}` shape. Version 1 yields ragged tuples instead.
VERSION = "v2"

#: deepagents' own file tools. The path one of these is given is the file that
#: lights up in the rail.
FILE_TOOLS = {"read_file", "edit_file", "write_file"}
```

This is what makes the lanes visible: without subgraphs=True the stream reports the supervisor only, and every investigator works invisibly.

`demo/evals/scorecard.py`

```python
def review(steps: list[tuple[str, str, str]], plan: list[tuple[int, str]],
           outcome: dict, gates: int | None = None, built: int = WHOLE,
           called: set[str] | None = None) -> list[Check]:
    """Every check this build has earned, in build order."""
    order = [name for name, _, _ in steps]
    took = [key for _, key in plan]
    methods = [method for _, _, method in steps if method.strip()]
    reached = called or set()

    def before(first: str, second: str) -> bool:
        return first in order and second in order and order.index(first) < order.index(second)

    looked = before("investigate", "decide")
    right = bool(took) and (CAUSE in took or DECOY not in took)
    checked = before("decide", "verify") and bool(took)
    named = len(methods) >= 2
    asked_around = len({t for t in reached if t.startswith("read_")}) >= 2
    saw_vendor = VENDOR in reached
    kept = "remember" in order
    fixed = bool(outcome.get("resolved"))
```

The trajectory evaluator. Nothing here asks a model, so two runs of the same plan get the same verdict - which is the only way a table can tell whether their change helped.

**Adds no file.** This step deploys services rather than configuration.

**What to watch.** The first line lands within seconds; the closing card says resolved or not and the minutes it took, and nine checks grade the run against the one before.

---


## What you built

The same call, six steps, each one argument wider:

```python
agent = create_deep_agent(model, tools, system_prompt, backend=CompositeBackend(...))   # 1
agent = create_deep_agent(..., skills=["/methods/"])                     # 2, and the memory as a prompt block
agent = create_deep_agent(..., tools=[actions], subagents=[...],
                          middleware=[TodoListMiddleware()])             # 3
agent = create_deep_agent(..., subagents=[..., vendor_status])          # 4, one tool that leaves the building
agent = create_deep_agent(..., interrupt_on={...}, checkpointer=saver)  # 5
graph.astream(..., subgraphs=True); client.create_feedback(...)         # 6, hosting and scoring wrap the graph
```

The folder is the configuration. Point it at a different problem and the same
eight capabilities apply to that one.

## When it does not behave

1. **Nothing plans.** `TodoListMiddleware` is not in the default stack on
   deepagents 0.7 - it has to be passed. An agent without it ignores a prompt
   asking it to plan, silently.
2. **The subagents are invisible in the stream.** `astream` needs
   `subgraphs=True`; without it the stream reports the supervisor only.
3. **A tool returning `Command(goto=END)` does not end the run.** Ending early
   is a `before_model` hook with `can_jump_to=["end"]`.
4. **A skill is never read.** Skills are directories holding a `SKILL.md`, and
   the `name` in the frontmatter has to match the directory. A colon in the
   `description` breaks the frontmatter without an error.
5. **The browser session returns an empty page.** A target created with a url
   reports `readyState == "complete"` for the empty document it starts on.
   Navigate on the attached session and wait.
6. **The wrong region.** Everything is `us-east-1`; the `us.` prefix on the
   model id is a cross-region inference profile that only resolves from a US
   source region.

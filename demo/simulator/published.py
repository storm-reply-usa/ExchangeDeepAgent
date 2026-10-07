"""The simulated night, written into real AWS services and read back from them.

`world.py` stays the only thing that decides what happens: the curve, the
outcomes and the residue are its. What this adds is WHERE the agent finds
them. The three readers stop computing an answer in the same process that
knows it and go and ask CloudWatch and DynamoDB, on a namespace of this
table's own, which is what makes the console link in the dossier open the
metric the agent actually read.

Three things it has to get right.

The clock. A run works minutes 144 to 300 of a night that is not tonight, and
CloudWatch makes a data point retrievable within seconds only when its
timestamp is close to now: a point three hours old can take two hours to
appear, a point a day old two days. So the night is laid onto the real clock
at seeding, ending ninety minutes ahead of it - CloudWatch accepts two hours
ahead - which keeps every minute a run will read inside the fast window for
the length of a session. That anchor is written into the deploy history so
the runtime, in another process later, maps minutes to the same instants.

The rewrite. An agent that rolls back changes what the numbers are from the
minute the rollback lands, and it has to be able to read that change back out
of the same service. Neither service overwrites: CloudWatch adds a second
value at the same timestamp to the first and averages them, and a log event
is append-only. So each plan's night is written under a dimension of its own
on the metric and a stream of its own in the log group, both named the same
way, and a read asks for the plan in force.

The gap. A metric written to CloudWatch is readable a few seconds later, not
at once, and a log line is read straight off its stream rather than through
the filter index, which lags. A reader that finds nothing falls back to the
simulator, and the sentence it composes is the same either way - which is
what the test in `test_the_agent_reads_real_data.py` holds it to.

Every function takes the clients it uses, so a test can hand it the fakes at
the foot of this module and the runtime can hand it boto3.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from dataclasses import dataclass, field

from demo.simulator import incident
from demo.simulator.world import ACTIONS, Step, run

#: One namespace, one log group and one partition per table, so a rollback at
#: one table cannot move the numbers at another.
NAMESPACE = "labcamp/table-{table}"
LOG_GROUP = "/labcamp/table-{table}/checkout"
DEPLOYS_TABLE = os.environ.get("LABCAMP_DEPLOYS_TABLE", "labcamp-deploys")

RATE = "checkout_error_rate"
ORDERS = "orders_per_min"

#: The window the night is published over, in minutes past midnight.
FROM, TO = incident.ONSET - 10, 300

#: How far ahead of the seeding moment the night ends. Two hours is the most
#: CloudWatch accepts; ninety minutes leaves room for a session that runs
#: long, and puts the alert (minute 194) sixteen minutes before the seeding.
ANCHOR_AHEAD = dt.timedelta(minutes=90)

#: The deploy history item that carries the anchor. Sort key -1 keeps it
#: before every release and action, and out of what a read lists.
ANCHOR_AT = -1

#: The log lines, as the checkout service would write them.
TEMPLATE_LINE = ("ERROR OrderConfirmation.render - template 'confirm_v2' is missing "
                 "partial 'summary'")
AUTH_OK_LINE = "INFO PaymentAuthorization - ok"
FRAUD_LINE = "WARN PaymentAuthorization - declined by fraud rule 'strict', card valid"


def region() -> str:
    return os.environ.get("AWS_REGION", "us-east-1")


def new_anchor() -> dt.datetime:
    """The instant the night ends, for a table seeded now."""
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    return now + ANCHOR_AHEAD


def at(minute: int, anchor: dt.datetime) -> dt.datetime:
    """A minute of the night as a real instant, on a night that ends at `anchor`."""
    return anchor - dt.timedelta(minutes=TO - minute)


def anchor_of(ddb, table: str) -> dt.datetime | None:
    """The anchor this table was seeded with, or None when it never was."""
    got = ddb.get_item(TableName=DEPLOYS_TABLE,
                       Key=_marshal({"table": str(table), "at": ANCHOR_AT}))
    item = got.get("Item")
    if not item:
        return None
    epoch = _unmarshal(item).get("epoch")
    return dt.datetime.fromtimestamp(int(epoch), dt.timezone.utc) if epoch else None


def _plan(plan: list[Step] | None) -> list[Step]:
    return sorted(plan) if plan else [(incident.ONSET, "nothing")]


def curve_of(plan: list[Step] | None) -> dict[int, float]:
    return dict(run(_plan(plan)).curve)


def stream_of(plan: list[Step] | None, anchor: dt.datetime) -> str:
    """The log stream a plan's night is written to.

    Named for the anchor as well as the plan, so a table seeded again writes
    fresh streams beside the old ones rather than on top of them, and a read
    never finds two nights in one stream.
    """
    stamp = str(int(anchor.timestamp()))
    steps = [(m, k) for m, k in _plan(plan) if k != "nothing"]
    if not steps:
        return f"night-{stamp}"
    return f"night-{stamp}-" + hashlib.sha1(repr(steps).encode()).hexdigest()[:10]


def cured_at(plan: list[Step] | None) -> int | None:
    """The minute the template fault stops, or None while nothing fixes it."""
    return next((m + ACTIONS[k].takes for m, k in _plan(plan) if ACTIONS[k].fixes), None)


def residue_cleared_at(plan: list[Step] | None) -> int | None:
    """The minute the fraud-check fault stops, or None while it is in place."""
    return next((m + ACTIONS[k].takes for m, k in _plan(plan) if ACTIONS[k].clears_residue),
                None)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def dimensions(table: str, plan: list[Step] | None, anchor: dt.datetime) -> list[dict]:
    """The table, and the night: one plan's curve under one dimension value."""
    return [{"Name": "table", "Value": str(table)},
            {"Name": "night", "Value": stream_of(plan, anchor)}]


def publish_metrics(cw, table: str, anchor: dt.datetime, plan: list[Step] | None = None) -> int:
    """Write the night's error rate and order rate under this plan, one point
    per minute, under a dimension of the plan's own."""
    curve = curve_of(plan)
    dims = dimensions(table, plan, anchor)
    points = []
    for minute in range(FROM, TO + 1):
        rate = curve.get(minute, incident.BASELINE_ERROR_RATE)
        points.append({"MetricName": RATE, "Dimensions": dims,
                       "Timestamp": at(minute, anchor), "Value": round(rate * 100, 4),
                       "Unit": "Percent"})
        points.append({"MetricName": ORDERS, "Dimensions": dims,
                       "Timestamp": at(minute, anchor),
                       "Value": float(int(incident.TRAFFIC_PER_MIN * (1.0 - rate))),
                       "Unit": "Count"})
    # Twenty a call is the API's limit, and it is not negotiable.
    for start in range(0, len(points), 20):
        cw.put_metric_data(Namespace=NAMESPACE.format(table=table),
                           MetricData=points[start:start + 20])
    return len(points)


def log_lines(plan: list[Step] | None) -> list[tuple[int, str]]:
    """The checkout log of the night under this plan, as (minute, line).

    One template failure a minute from onset until the cure lands, a payment
    authorization line every minute throughout, and one fraud-check decline a
    minute from the moment the payments release ships until it is rolled back.
    """
    cured = cured_at(plan)
    cleared = residue_cleared_at(plan)
    lines: list[tuple[int, str]] = []
    for minute in range(FROM, TO + 1):
        lines.append((minute, AUTH_OK_LINE))
        if minute >= incident.ONSET and (cured is None or minute < cured):
            lines.append((minute, TEMPLATE_LINE))
        if minute >= incident.DECOY.at and (cleared is None or minute < cleared):
            lines.append((minute, FRAUD_LINE))
    return lines


def publish_logs(logs, table: str, anchor: dt.datetime, plan: list[Step] | None = None) -> str:
    """Write the night's log under this plan, to a stream of the plan's own."""
    group = LOG_GROUP.format(table=table)
    stream = stream_of(plan, anchor)
    try:
        logs.create_log_group(logGroupName=group)
    except Exception as problem:
        if not _already_exists(problem):
            raise
    try:
        logs.create_log_stream(logGroupName=group, logStreamName=stream)
    except Exception as problem:
        if not _already_exists(problem):
            raise
        # Already written for this plan, on this table.
        return stream
    events = [{"timestamp": int(at(minute, anchor).timestamp() * 1000) + i, "message": line}
              for i, (minute, line) in enumerate(log_lines(plan))]
    events.sort(key=lambda e: e["timestamp"])
    # Ten thousand a call is the API's limit; the night is well under it,
    # but the batch is cut anyway so a longer night would not fail here.
    for start in range(0, len(events), 5000):
        logs.put_log_events(logGroupName=group, logStreamName=stream,
                            logEvents=events[start:start + 5000])
    return stream


def _already_exists(problem: Exception) -> bool:
    """Whether a create call refused because the thing is already there, which
    is the one refusal that is not a failure. Anything else is raised, so a
    seeding that cannot write is a seeding that fails out loud."""
    return "AlreadyExists" in type(problem).__name__ or "AlreadyExists" in str(problem)


def deploy_items(table: str, plan: list[Step] | None = None) -> list[dict]:
    """The deploy history under this plan: the two releases, then every action."""
    items = [
        {"table": str(table), "at": incident.CAUSE.at, "kind": "release",
         "service": incident.CAUSE.service, "summary": incident.CAUSE.summary,
         "reversible_in": incident.CAUSE.reversible_in},
        {"table": str(table), "at": incident.DECOY.at, "kind": "release",
         "service": incident.DECOY.service, "summary": incident.DECOY.summary,
         "reversible_in": incident.DECOY.reversible_in},
    ]
    for minute, key in _plan(plan):
        if key == "nothing":
            continue
        items.append({"table": str(table), "at": minute, "kind": "action",
                      "action": key, "label": ACTIONS[key].label,
                      "lands": minute + ACTIONS[key].takes})
    return items


def publish_deploys(ddb, table: str, plan: list[Step] | None = None) -> int:
    for item in deploy_items(table, plan):
        ddb.put_item(TableName=DEPLOYS_TABLE, Item=_marshal(item))
    return len(deploy_items(table, plan))


def publish_anchor(ddb, table: str, anchor: dt.datetime) -> None:
    ddb.put_item(TableName=DEPLOYS_TABLE, Item=_marshal(
        {"table": str(table), "at": ANCHOR_AT, "kind": "anchor",
         "epoch": int(anchor.timestamp())}))


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
    return written


def after_action(clients: "Clients", table: str, plan: list[Step],
                 anchor: dt.datetime) -> None:
    """The world after an action, written where the agent will read it.

    The whole night's metric under the new plan's dimension, the whole
    night's log on the new plan's stream, and the action in the deploy
    history. About a second.
    """
    publish_metrics(clients.cloudwatch, table, anchor, plan)
    publish_logs(clients.logs, table, anchor, plan)
    publish_deploys(clients.dynamodb, table, plan)


def clear(clients: "Clients", table: str) -> None:
    """Take this table's namespace apart. Metrics expire on their own."""
    try:
        clients.logs.delete_log_group(logGroupName=LOG_GROUP.format(table=table))
    except Exception:
        pass
    for item in deploy_items(table, None) + [{"table": str(table), "at": ANCHOR_AT}]:
        try:
            clients.dynamodb.delete_item(TableName=DEPLOYS_TABLE,
                                         Key=_marshal({"table": item["table"], "at": item["at"]}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Reading. Each returns None when the service has nothing yet, and the reader
# falls back to the simulator.
# ---------------------------------------------------------------------------

def read_rate(cw, table: str, minute: int, anchor: dt.datetime,
              plan: list[Step] | None = None) -> float | None:
    """What the metric says the rate was at that minute under this plan."""
    when = at(min(minute, TO), anchor)
    got = cw.get_metric_statistics(
        Namespace=NAMESPACE.format(table=table), MetricName=RATE,
        Dimensions=dimensions(table, plan, anchor),
        StartTime=when - dt.timedelta(seconds=30),
        EndTime=when + dt.timedelta(seconds=30),
        Period=60, Statistics=["Average"])
    points = sorted(got.get("Datapoints", []), key=lambda p: p["Timestamp"])
    return points[-1]["Average"] / 100 if points else None


@dataclass
class LogFacts:
    """What the checkout log says, up to a minute of the night."""

    first_template: int | None = None
    last_template: int | None = None
    first_fraud: int | None = None
    last_fraud: int | None = None
    auth_ok: bool = False
    minutes: set = field(default_factory=set)


def read_log(logs, table: str, plan: list[Step] | None, upto: int,
             anchor: dt.datetime) -> LogFacts | None:
    """The facts in the log of the plan in force, up to `upto`.

    Read straight off the stream with GetLogEvents, which returns what was
    written seconds ago; the filter API indexes backdated lines minutes
    later and answered empty in the meantime.
    """
    group = LOG_GROUP.format(table=table)
    stream = stream_of(plan, anchor)
    facts = LogFacts()
    cutoff = int(at(min(upto, TO), anchor).timestamp() * 1000) + 999
    token = None
    while True:
        kwargs = {"logGroupName": group, "logStreamName": stream, "startFromHead": True}
        if token:
            kwargs["nextToken"] = token
        try:
            got = logs.get_log_events(**kwargs)
        except Exception:
            return None
        for event in got.get("events", []):
            if event["timestamp"] > cutoff:
                continue
            minute = _minute_of(event["timestamp"], anchor)
            facts.minutes.add(minute)
            message = event.get("message", "")
            if message == TEMPLATE_LINE:
                facts.first_template = minute if facts.first_template is None \
                    else min(facts.first_template, minute)
                facts.last_template = minute if facts.last_template is None \
                    else max(facts.last_template, minute)
            elif message == FRAUD_LINE:
                facts.first_fraud = minute if facts.first_fraud is None \
                    else min(facts.first_fraud, minute)
                facts.last_fraud = minute if facts.last_fraud is None \
                    else max(facts.last_fraud, minute)
            elif message == AUTH_OK_LINE:
                facts.auth_ok = True
        following = got.get("nextForwardToken")
        if not following or following == token:
            break
        token = following
    return facts if facts.minutes else None


def read_deploys(ddb, table: str) -> list[dict] | None:
    got = ddb.query(TableName=DEPLOYS_TABLE,
                    KeyConditionExpression="#t = :t",
                    ExpressionAttributeNames={"#t": "table"},
                    ExpressionAttributeValues={":t": {"S": str(table)}})
    items = [_unmarshal(i) for i in got.get("Items", []) if _unmarshal(i)["at"] != ANCHOR_AT]
    return sorted(items, key=lambda i: i["at"]) if items else None


def _minute_of(timestamp_ms: int, anchor: dt.datetime) -> int:
    """The minute of the night a real instant stands for, on this anchor."""
    when = dt.datetime.fromtimestamp(timestamp_ms / 1000, dt.timezone.utc)
    # Rounded, not floored: a line carries a few milliseconds past its minute
    # so that lines of one minute keep their order.
    return TO - round((anchor - when).total_seconds() / 60)


def _marshal(item: dict) -> dict:
    out = {}
    for key, value in item.items():
        out[key] = {"N": str(value)} if isinstance(value, (int, float)) and not isinstance(value, bool) \
            else {"S": str(value)}
    return out


def _unmarshal(item: dict) -> dict:
    out = {}
    for key, value in item.items():
        if "N" in value:
            number = float(value["N"])
            out[key] = int(number) if number.is_integer() else number
        else:
            out[key] = value.get("S", "")
    return out


# ---------------------------------------------------------------------------
# The clients, real or fake
# ---------------------------------------------------------------------------

@dataclass
class Clients:
    cloudwatch: object
    logs: object
    dynamodb: object


def boto() -> Clients:
    import boto3

    return Clients(boto3.client("cloudwatch", region_name=region()),
                   boto3.client("logs", region_name=region()),
                   boto3.client("dynamodb", region_name=region()))


class FakeCloudWatch:
    """Enough of the CloudWatch API for a test: points by metric and minute."""

    def __init__(self):
        self.points: dict[tuple[str, str, str, int], float] = {}
        self.calls = 0

    def put_metric_data(self, Namespace, MetricData):
        self.calls += 1
        assert len(MetricData) <= 20, "the API takes twenty items a call"
        for p in MetricData:
            dims = ",".join(f"{d['Name']}={d['Value']}" for d in p["Dimensions"])
            key = (Namespace, p["MetricName"], dims, int(p["Timestamp"].timestamp() // 60))
            self.points[key] = p["Value"]
        return {}

    def get_metric_statistics(self, Namespace, MetricName, Dimensions, StartTime, EndTime,
                              Period, Statistics):
        dims = ",".join(f"{d['Name']}={d['Value']}" for d in Dimensions)
        lo, hi = int(StartTime.timestamp() // 60), int(EndTime.timestamp() // 60)
        found = []
        for (ns, name, ds, minute), value in self.points.items():
            if ns == Namespace and name == MetricName and ds == dims and lo <= minute <= hi:
                found.append({"Timestamp": dt.datetime.fromtimestamp(minute * 60, dt.timezone.utc),
                              "Average": value})
        return {"Datapoints": found}


class FakeLogs:
    def __init__(self):
        self.groups: dict[str, dict[str, list[dict]]] = {}

    def create_log_group(self, logGroupName):
        if logGroupName in self.groups:
            raise RuntimeError("ResourceAlreadyExistsException")
        self.groups[logGroupName] = {}

    def create_log_stream(self, logGroupName, logStreamName):
        streams = self.groups.setdefault(logGroupName, {})
        if logStreamName in streams:
            raise RuntimeError("ResourceAlreadyExistsException")
        streams[logStreamName] = []

    def put_log_events(self, logGroupName, logStreamName, logEvents):
        self.groups[logGroupName][logStreamName] += logEvents
        return {}

    def get_log_events(self, logGroupName, logStreamName, startFromHead=True, nextToken=None):
        # One page, and the same token back, which is how the real API says
        # there is nothing further.
        events = sorted(self.groups.get(logGroupName, {}).get(logStreamName, []),
                        key=lambda e: e["timestamp"])
        return {"events": events, "nextForwardToken": "end"}

    def delete_log_group(self, logGroupName):
        self.groups.pop(logGroupName, None)


class FakeDynamo:
    def __init__(self):
        self.items: dict[tuple[str, str], dict] = {}

    def put_item(self, TableName, Item):
        self.items[(Item["table"]["S"], Item["at"]["N"])] = Item
        return {}

    def delete_item(self, TableName, Key):
        self.items.pop((Key["table"]["S"], Key["at"]["N"]), None)
        return {}

    def get_item(self, TableName, Key):
        item = self.items.get((Key["table"]["S"], Key["at"]["N"]))
        return {"Item": item} if item else {}

    def query(self, TableName, KeyConditionExpression, ExpressionAttributeNames,
              ExpressionAttributeValues):
        table = ExpressionAttributeValues[":t"]["S"]
        return {"Items": [i for (t, _), i in self.items.items() if t == table]}


def fakes() -> Clients:
    return Clients(FakeCloudWatch(), FakeLogs(), FakeDynamo())


class Refusing:
    """A client that must never be called: the test for the fallback path."""

    def __getattr__(self, name):
        raise AssertionError(f"AWS was called ({name}) when it should not have been")


def refusing() -> Clients:
    return Clients(Refusing(), Refusing(), Refusing())

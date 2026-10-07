"""The three readers answer from CloudWatch and DynamoDB when a table's night
has been published there, and from the simulator when it has not - and the
sentence is the same to the character either way.

`your-config/` briefs the investigators on the shape of those answers, and the
rubric and the three change cards are built on them. A real read that returned
the same numbers in different words would move everything downstream without
any test noticing, so this one compares the two paths word for word, before
and after every action the night can take.
"""

from __future__ import annotations

import pytest

from demo import tools as incident
from demo.simulator import published
from demo.simulator.incident import DECOY, ONSET
from demo.simulator.world import ACTIONS

TABLE = "7"

#: Every plan the room can produce, and the minutes a reader is likely to ask
#: at: the alert, mid-investigation, after the first cure, and late.
PLANS = [
    [],
    [(200, "roll_back")],
    [(200, "disable_feature"), (202, "roll_back")],
    [(200, "roll_back"), (208, "roll_back_payments")],
    [(200, "wake_owner")],
    [(200, "roll_back_payments")],
]
MINUTES = [194, 199, 205, 209, 215, 240, 299]


def _simulated(plan, minute):
    incident.use(None)
    run = incident.reset(at=minute, table=None)
    run.plan = list(plan)
    return (incident.read_metrics.invoke({}), incident.read_logs.invoke({}),
            incident.read_deploys.invoke({}))


def _real(clients, plan, minute):
    incident.use(clients)
    run = incident.reset(at=minute, table=TABLE)
    run.plan = list(plan)
    return (incident.read_metrics.invoke({}), incident.read_logs.invoke({}),
            incident.read_deploys.invoke({}))


@pytest.fixture(autouse=True)
def _clean():
    yield
    incident.use(None)
    incident.reset()


@pytest.mark.parametrize("plan", PLANS, ids=lambda p: "+".join(k for _, k in p) or "nothing")
@pytest.mark.parametrize("minute", MINUTES)
def test_the_real_read_says_what_the_simulator_says(plan, minute):
    clients = published.fakes()
    published.seed(clients, TABLE)
    if plan:
        published.after_action(clients, TABLE, plan, published.anchor_of(clients.dynamodb, TABLE))
    assert _real(clients, plan, minute) == _simulated(plan, minute)


def test_the_real_read_was_a_read():
    """The fake was asked, and the plan's own stream was the one read."""
    clients = published.fakes()
    published.seed(clients, TABLE)
    plan = [(200, "roll_back")]
    anchor = published.anchor_of(clients.dynamodb, TABLE)
    assert anchor is not None
    published.after_action(clients, TABLE, plan, anchor)
    assert clients.cloudwatch.calls > 0
    streams = clients.logs.groups[published.LOG_GROUP.format(table=TABLE)]
    assert set(streams) == {published.stream_of(None, anchor), published.stream_of(plan, anchor)}
    assert published.read_rate(clients.cloudwatch, TABLE, 240, anchor, plan) == pytest.approx(0.009, abs=1e-4)
    assert published.read_rate(clients.cloudwatch, TABLE, 240, anchor, None) == pytest.approx(0.38, abs=1e-3)
    facts = published.read_log(clients.logs, TABLE, plan, 240, anchor)
    assert facts.first_template == ONSET
    assert facts.last_template == 200 + ACTIONS["roll_back"].takes - 1
    assert facts.first_fraud == DECOY.at


def test_without_a_table_nothing_asks_aws():
    """A laptop, a test, or a table whose seeding failed: the simulator answers
    and AWS is never called."""
    incident.use(published.refusing())
    run = incident.reset(at=200, table=None)
    run.plan = [(200, "roll_back")]
    assert "checkout errors" in incident.read_metrics.invoke({})
    assert "First failure" in incident.read_logs.invoke({})
    assert "Nothing else changed" in incident.read_deploys.invoke({})


def test_an_empty_namespace_falls_back():
    """Seeded two seconds ago and not readable yet, or never seeded: the same
    sentence, from the simulator."""
    clients = published.fakes()          # nothing published
    assert _real(clients, [], 200) == _simulated([], 200)


def test_an_action_republishes_and_the_next_read_sees_it():
    clients = published.fakes()
    published.seed(clients, TABLE)
    incident.use(clients)
    run = incident.reset(at=200, table=TABLE)
    anchor = published.anchor_of(clients.dynamodb, TABLE)
    before = published.read_rate(clients.cloudwatch, TABLE, 240, anchor, None)
    incident.roll_back.invoke({"saying": "rolling back"})
    after = published.read_rate(clients.cloudwatch, TABLE, 240, anchor, list(run.plan))
    assert before == pytest.approx(0.38, abs=1e-3)
    assert after == pytest.approx(0.009, abs=1e-4)
    items = published.read_deploys(clients.dynamodb, TABLE)
    assert [i["kind"] for i in items] == ["release", "release", "action"]


def test_the_metric_write_respects_the_twenty_item_limit():
    clients = published.fakes()
    n = published.publish_metrics(clients.cloudwatch, TABLE, published.new_anchor())
    assert n == 2 * (published.TO - published.FROM + 1)
    assert clients.cloudwatch.calls == -(-n // 20)


def test_clear_takes_the_namespace_apart():
    clients = published.fakes()
    published.seed(clients, TABLE)
    published.clear(clients, TABLE)
    assert published.LOG_GROUP.format(table=TABLE) not in clients.logs.groups
    assert published.read_deploys(clients.dynamodb, TABLE) is None
    assert published.anchor_of(clients.dynamodb, TABLE) is None


def test_the_night_ends_ahead_of_the_seeding_and_inside_what_cloudwatch_accepts():
    """Within two hours ahead, so the write is accepted; far enough ahead that
    the whole of a session's reads stay in the window that is readable in
    seconds rather than hours."""
    import datetime as dt

    anchor = published.new_anchor()
    now = dt.datetime.now(dt.timezone.utc)
    assert dt.timedelta(minutes=80) < anchor - now <= dt.timedelta(minutes=120)
    assert published.at(194, anchor) < now
    assert published.at(published.TO, anchor) == anchor


def test_a_reseed_starts_the_night_over():
    """Seeding twice writes a second night beside the first, on the newer
    anchor, and a read on that anchor sees only the second."""
    clients = published.fakes()
    published.seed(clients, TABLE)
    first = published.anchor_of(clients.dynamodb, TABLE)
    import datetime as dt
    second = first + dt.timedelta(minutes=5)
    published.seed(clients, TABLE, anchor=second)
    group = clients.logs.groups[published.LOG_GROUP.format(table=TABLE)]
    assert set(group) == {published.stream_of(None, first), published.stream_of(None, second)}
    assert published.anchor_of(clients.dynamodb, TABLE) == second
    facts = published.read_log(clients.logs, TABLE, None, 240, second)
    assert facts.first_template == ONSET


def test_a_seeding_that_fails_half_way_leaves_no_anchor():
    """Metrics written, logs refused: no anchor is published, so every reader
    falls back to the simulator rather than reading half a night."""
    clients = published.fakes()
    clients.logs = published.Refusing()
    with pytest.raises(AssertionError):
        published.seed(clients, TABLE)
    assert published.anchor_of(clients.dynamodb, TABLE) is None
    clients.logs = published.FakeLogs()
    assert _real(clients, [], 200) == _simulated([], 200)


def test_each_reader_falls_back_on_its_own():
    """A night with an anchor and metrics but no log: the log reader answers
    from the simulator while the metric reader reads, and the sentences are
    the simulator's either way."""
    clients = published.fakes()
    published.seed(clients, TABLE)
    clients.logs.groups.clear()
    assert _real(clients, [], 200) == _simulated([], 200)

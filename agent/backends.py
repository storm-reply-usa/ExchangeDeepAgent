"""Where the agent's virtual filesystem actually lives.

One `CompositeBackend` routes path prefixes to different backends. Longest prefix
wins, so the routes below never overlap ambiguously:

    /methods/   the running folder's methods, laid out as skills. An attendee
                edits one flat file and the next run reads it.
    /durable/   S3, when a bucket is configured. Survives the thread.
    /memories/  S3 under a per-customer namespace. Survives the thread and is
                isolated per customer.
    everything else   LangGraph state. Ephemeral, gone with the thread.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone

import boto3
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

METHODS_ROUTE = "/methods/"
DURABLE_ROUTE = "/durable/"
MEMORIES_ROUTE = "/memories/"

DURABLE_PREFIX = "labcamp-durable"
MEMORIES_PREFIX = "labcamp-memories"

# What one shift asks the next one to do differently. AGENTS.md tells the agent to read
# this file before it starts and to append to it when a human teaches it something.
LESSONS_FILE = f"{MEMORIES_ROUTE}lessons.md"


def customer_namespace(runtime) -> tuple[str, str, str]:
    """Namespace `/memories/` per customer so one customer never reads another's.

    One bucket per deployment, and when a bucket is given
    a different sentence within the same three minutes. Falling back to the table keeps
    those apart: without it they all write the same object, the writes clobber each other,
    and a Commander opens the file to find somebody else's lesson in it.
    """
    context = getattr(runtime, "context", None) or {}
    customer_id = (
        context.get("customer_id")
        if isinstance(context, dict)
        else getattr(context, "customer_id", None)
    )
    owner = customer_id or os.environ.get("LABCAMP_TABLE") or "anonymous"
    return ("customers", str(owner), "memories")


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


def memory_object_uri(bucket: str, path: str = LESSONS_FILE) -> str:
    """Where a `/memories/` file actually lands in S3.

    The agent writes a path; the backend turns it into a namespaced object.     Commander to open that object and read what the agent learned, so the translation has
    to be stated somewhere they can be pointed at rather than reconstructed on the night.
    """
    if not path.startswith(MEMORIES_ROUTE):
        raise ValueError(f"{path!r} is not under {MEMORIES_ROUTE}")
    namespace = "/".join(customer_namespace(None))
    return f"s3://{bucket}/{MEMORIES_PREFIX}/{namespace}/{path[len(MEMORIES_ROUTE):]}.json"


def _namespace_prefix(namespace: tuple[str, ...]) -> str:
    return "/".join(namespace) + "/" if namespace else ""


class S3Store(BaseStore):
    """A LangGraph `BaseStore` backed by one S3 bucket.

    Enough to show the durable-filesystem pattern: no TTL, no semantic search, no
    protection against concurrent writes. `deepagents-backends` ships a maintained
    `S3Backend` for anything real.
    """

    def __init__(self, bucket: str, prefix: str = "agent-files", region: str | None = None):
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._s3 = boto3.client(
            "s3", region_name=region or os.environ.get("AWS_REGION", "us-east-1")
        )

    def _full_key(self, ns: tuple[str, ...], key: str) -> str:
        return f"{self._prefix}/{_namespace_prefix(ns)}{key.lstrip('/')}.json"

    def _get_one(self, op: GetOp) -> Item | None:
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=self._full_key(op.namespace, op.key))
        except self._s3.exceptions.NoSuchKey:
            return None
        body = json.loads(resp["Body"].read())
        return Item(
            namespace=op.namespace,
            key=op.key,
            value=body["value"],
            created_at=datetime.fromisoformat(body["created_at"]),
            updated_at=datetime.fromisoformat(body["updated_at"]),
        )

    def _put_one(self, op: PutOp) -> None:
        s3_key = self._full_key(op.namespace, op.key)
        if op.value is None:
            self._s3.delete_object(Bucket=self._bucket, Key=s3_key)
            return
        now = datetime.now(timezone.utc).isoformat()
        created_at = now
        try:
            existing = self._s3.get_object(Bucket=self._bucket, Key=s3_key)
            created_at = json.loads(existing["Body"].read())["created_at"]
        except self._s3.exceptions.NoSuchKey:
            pass
        body = {"value": op.value, "created_at": created_at, "updated_at": now}
        self._s3.put_object(Bucket=self._bucket, Key=s3_key, Body=json.dumps(body).encode("utf-8"))

    def _item_from_key(self, key: str) -> SearchItem:
        rel = key[len(self._prefix) + 1 :]
        parts = rel.split("/")
        body = json.loads(self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read())
        return SearchItem(
            namespace=tuple(parts[:-1]),
            key=parts[-1][: -len(".json")],
            value=body["value"],
            created_at=datetime.fromisoformat(body["created_at"]),
            updated_at=datetime.fromisoformat(body["updated_at"]),
        )

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = f"{self._prefix}/{_namespace_prefix(op.namespace_prefix)}"
        paginator = self._s3.get_paginator("list_objects_v2")
        items = [
            self._item_from_key(obj["Key"])
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix)
            for obj in page.get("Contents", []) or []
            if obj["Key"].endswith(".json")
        ]
        start = op.offset or 0
        return items[start : start + (op.limit or 10)]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        paginator = self._s3.get_paginator("list_objects_v2")
        seen: set[tuple[str, ...]] = set()
        for page in paginator.paginate(Bucket=self._bucket, Prefix=f"{self._prefix}/"):
            for obj in page.get("Contents", []) or []:
                parts = obj["Key"][len(self._prefix) + 1 :].split("/")
                if len(parts) > 1:
                    seen.add(tuple(parts[:-1]))
        return sorted(seen)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get_one(op))
            elif isinstance(op, PutOp):
                self._put_one(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:
                raise NotImplementedError(f"S3Store does not handle {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return self.batch(ops)

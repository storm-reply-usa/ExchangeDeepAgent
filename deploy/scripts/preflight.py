"""The check a speaker runs the morning of the event.

Every line either says OK or says exactly what to fix. Run it from the repository root:

    uv run python scripts/preflight.py

Exits non-zero if anything an exercise depends on is broken, so it can also run in CI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env", override=True)
sys.path.insert(0, str(REPO_ROOT))

AGENT_MODEL = os.environ.get("AGENT_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-6"

results: list[tuple[bool, str, str]] = []


def check(label: str):
    """Run one check and record whether it passed, without letting it stop the rest."""

    def decorator(fn):
        try:
            detail = fn() or "ok"
            results.append((True, label, str(detail)))
        except Exception as exc:
            results.append((False, label, f"{type(exc).__name__}: {exc}"))
        return fn

    return decorator


@check("environment variables")
def _env():
    required = ["AWS_REGION", "LANGSMITH_API_KEY", "LABCAMP_BUCKET"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing: " + ", ".join(missing))
    if os.environ["AWS_REGION"] != "us-east-1":
        raise RuntimeError(f"AWS_REGION is {os.environ['AWS_REGION']}, everything is us-east-1")

    # The endpoint is optional, but it has to agree with where the key came from. A key
    # from one instance returns an opaque 401 against the other, and that lands at 0:12
    # with thirty people watching.
    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "")
    key = os.environ["LANGSMITH_API_KEY"]
    if endpoint and "aws.api.smith" not in endpoint and "smith.langchain.com" not in endpoint:
        raise RuntimeError(f"LANGSMITH_ENDPOINT is {endpoint}, which is neither known instance")
    if key.startswith("lsv2_sk_") and not os.environ.get("LANGSMITH_WORKSPACE_ID"):
        raise RuntimeError(
            "this is a Service Key and LANGSMITH_WORKSPACE_ID is not set. The key will "
            "authenticate and every trace write will be silently rejected."
        )
    tenant = "the AWS tenant" if "aws.api.smith" in endpoint else "the default instance"
    return f"all present, us-east-1, LangSmith on {tenant}"


@check("aws credentials")
def _credentials():
    import boto3

    identity = boto3.client("sts", region_name="us-east-1").get_caller_identity()
    return identity["Arn"]


@check("bedrock model access")
def _models():
    import boto3

    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    for model in (AGENT_MODEL, JUDGE_MODEL):
        client.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": "Reply with the single word: ok"}]}],
            inferenceConfig={"maxTokens": 8},
        )
    return "agent and judge models answer"


@check("knowledge base answers")
def _kb():
    from demo.tools import search_runbooks

    answer = search_runbooks.invoke("SessionNotFound mobile checkout SameSite cookie")
    if "samesite" not in answer.lower():
        raise RuntimeError("the knowledge base answered but did not surface the SameSite precedent")
    if "s3://" not in answer:
        raise RuntimeError("the knowledge base answered without an s3:// source URI")
    return "finds the SameSite precedent with its s3:// source"


@check("gateway tools")
def _gateway():
    import asyncio

    from agent.gateway import get_gateway_tools

    names = sorted(t.name for t in asyncio.run(get_gateway_tools()))
    expected = {"get_error_rates", "get_service_logs", "list_recent_deploys", "rollback_deploy"}
    if not expected <= set(names):
        raise RuntimeError(f"expected {sorted(expected)}, got {names}")
    return ", ".join(names)


@check("provider status page")
def _public_doc():
    import httpx

    from demo.tools import presign_provider_status_page

    response = httpx.get(presign_provider_status_page(), timeout=15.0)
    response.raise_for_status()
    return f"{len(response.text)} bytes behind a presigned URL"


@check("s3 durable route is writable")
def _s3():
    from agent.backends import S3Store

    store = S3Store(bucket=os.environ["LABCAMP_BUCKET"], prefix="labcamp-preflight")
    store.put(("preflight",), "check", {"content": "ok"})
    item = store.get(("preflight",), "check")
    store.put(("preflight",), "check", None)
    if item is None:
        raise RuntimeError("wrote to S3 but could not read it back")
    return "write, read and delete all succeed"


@check("skills and house rules are readable")
def _skills():
    from deepagents.middleware.skills import _list_skills_with_errors

    from agent.agent import SKILLS_PATH
    from agent.backends import build_backend

    backend = build_backend(str(REPO_ROOT))
    skills, error = _list_skills_with_errors(backend, SKILLS_PATH)
    if error:
        raise RuntimeError(error)
    return ", ".join(sorted(s["name"] for s in skills)) or "no skills found"


@check("tracing reaches langsmith")
def _tracing():
    """Authentication is not the check. Writing a trace and reading it back is.

    A Service Key scoped to no workspace authenticates happily and then fails every trace
    write with a background 403 that nothing surfaces. The scoreboard would be empty on
    the day with no error anywhere, so this check sends a real trace and looks for it.
    """
    import time
    import uuid

    from langsmith import Client, traceable

    marker = f"preflight-{uuid.uuid4().hex[:8]}"

    @traceable(name=marker)
    def _ping() -> str:
        return "ok"

    _ping()
    client = Client()
    client.flush()

    project = os.environ.get("LANGSMITH_PROJECT", "labcamp-deep-agents-aws")
    for _ in range(10):
        time.sleep(1.5)
        if any(r.name == marker for r in client.list_runs(project_name=project, limit=20)):
            return f"wrote and read back a trace in '{project}'"
    raise RuntimeError(
        f"the trace never arrived in '{project}'. If the key is a Service Key, set "
        "LANGSMITH_WORKSPACE_ID — without it every write is silently rejected."
    )


def main() -> int:
    width = max(len(label) for _, label, _ in results)
    for passed, label, detail in results:
        print(f"{'OK  ' if passed else 'FAIL'}  {label.ljust(width)}  {detail}")
    failures = sum(1 for passed, _, _ in results if not passed)
    print()
    print(f"{len(results) - failures}/{len(results)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

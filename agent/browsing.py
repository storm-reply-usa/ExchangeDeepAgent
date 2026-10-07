"""Reading a real page, through AgentCore Browser.

Everything else an investigator reads comes out of the simulator, which is the
point: the incident has to be the same for every table. The provider's status
page is the exception. It is a real document fetched over the network by a
Chromium that runs in AWS rather than in this process, so ruling a third party
out costs what it costs in life.

The Chrome DevTools Protocol is spoken directly over the session's websocket.
Playwright would do the same job and bring a driver and a browser download with
it, neither of which this runtime needs.
"""

from __future__ import annotations

import asyncio
import json
import os

REGION = os.environ.get("AWS_REGION", "us-east-1")
#: A target created with a url reports `readyState == "complete"` for the empty
#: document it starts on, so the page is navigated on the attached session and
#: waited for instead.
BLANK = "about:blank"
#: The browser AgentCore provides without one being created first.
DEFAULT_BROWSER = "aws.browser.v1"
SESSION_SECONDS = 300
SETTLE = 0.25
PATIENCE = 80


async def _read(ws_url: str, headers: dict, url: str, timeout: float) -> str:
    import websockets

    counter = 0

    async with websockets.connect(ws_url, additional_headers=headers,
                                  max_size=None, open_timeout=timeout) as ws:
        async def call(method, params=None, session=None):
            nonlocal counter
            counter += 1
            message = {"id": counter, "method": method, "params": params or {}}
            if session:
                message["sessionId"] = session
            await ws.send(json.dumps(message))
            while True:
                got = json.loads(await asyncio.wait_for(ws.recv(), timeout))
                if got.get("id") == message["id"]:
                    return got.get("result", {})

        target = (await call("Target.createTarget", {"url": BLANK}))["targetId"]
        session = (await call("Target.attachToTarget",
                              {"targetId": target, "flatten": True}))["sessionId"]
        await call("Page.enable", session=session)
        await call("Runtime.enable", session=session)
        await call("Page.navigate", {"url": url}, session=session)
        for _ in range(PATIENCE):
            got = await call(
                "Runtime.evaluate",
                {"expression": "document.readyState + '|' + document.body.innerText.length",
                 "returnByValue": True},
                session=session)
            state = str(got.get("result", {}).get("value", ""))
            if state.startswith("complete") and not state.endswith("|0"):
                break
            await asyncio.sleep(SETTLE)
        got = await call("Runtime.evaluate",
                         {"expression": "document.body.innerText", "returnByValue": True},
                         session=session)
        return str(got.get("result", {}).get("value", "")).strip()


def _session(region: str):
    """Open a browser session on the data plane, and keep what AWS answered.

    The AgentCore SDK wraps these two calls, but bundling it into the workbench
    function pulls the whole web stack up with it. Both are ordinary boto3, and
    the websocket is signed the way any AgentCore request is.
    """
    import boto3

    plane = boto3.client("bedrock-agentcore", region_name=region)
    got = plane.start_browser_session(browserIdentifier=DEFAULT_BROWSER,
                                      sessionTimeoutSeconds=SESSION_SECONDS)
    meta = got.get("ResponseMetadata", {})
    return plane, {
        "browser": got["browserIdentifier"],
        "session": got["sessionId"],
        "request_id": meta.get("RequestId", ""),
        "status": meta.get("HTTPStatusCode"),
    }


def _ws_headers(region: str, browser: str, session: str):
    """A signed websocket address for that session's automation stream."""
    import base64
    import datetime
    import secrets

    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    host = f"bedrock-agentcore.{region}.amazonaws.com"
    path = f"/browser-streams/{browser}/sessions/{session}/automation"
    frozen = boto3.Session().get_credentials().get_frozen_credentials()

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    request = AWSRequest(method="GET", url=f"https://{host}{path}",
                         headers={"host": host, "x-amz-date": stamp})
    SigV4Auth(frozen, "bedrock-agentcore", region).add_auth(request)

    headers = {
        "Host": host,
        "X-Amz-Date": stamp,
        "Authorization": request.headers["Authorization"],
        "Upgrade": "websocket",
        "Connection": "Upgrade",
        "Sec-WebSocket-Version": "13",
        "Sec-WebSocket-Key": base64.b64encode(secrets.token_bytes(16)).decode(),
        "User-Agent": f"labcamp-oncall/1.0 (Session: {session})",
    }
    if frozen.token:
        headers["X-Amz-Security-Token"] = frozen.token
    return f"wss://{host}{path}", headers


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


def page_text(url: str, *, timeout: float = 25.0) -> str:
    """The rendered text of `url`, read by a browser running in AgentCore."""
    return page_text_with_receipt(url, timeout=timeout)["text"]


def public_url(page: str, *, expires: int = 900) -> str:
    """A link the browser can open to one of the pages served for this lab."""
    import boto3

    bucket = os.environ.get("LABCAMP_BUCKET")
    if not bucket:
        return ""
    return boto3.client("s3", region_name=REGION).generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": f"public-docs/{page}"},
        ExpiresIn=expires)


def status_page_url(*, expires: int = 900) -> str:
    """The provider's own status page."""
    return public_url("provider-status.html", expires=expires)


def reports_url(*, expires: int = 900) -> str:
    """What customers are reporting in public, which the provider does not say."""
    return public_url("community-reports.html", expires=expires)

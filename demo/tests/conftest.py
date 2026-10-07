"""A chat model that answers from a script.

Every test in this suite runs with no AWS session and no model calls, so the whole
repository can be checked in CI and on a plane. The scripted model implements just
enough of the chat-model surface for the deep agent harness to drive it.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of AIMessages, one per model call."""

    script: list[AIMessage] = []
    calls: list[list[BaseMessage]] = []
    bound_tools: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        self.bound_tools = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        index = len(self.calls) - 1
        message = self.script[index] if index < len(self.script) else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def scripted():
    """Build a ScriptedChatModel from a list of AIMessages."""

    def _make(*script: AIMessage) -> ScriptedChatModel:
        return ScriptedChatModel(script=list(script), calls=[], bound_tools=[])

    return _make


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def as_call(tool, **args) -> str:
    """Invoke a tool the way the agent does, and answer with what it said.

    A tool that can end its step takes the id of the call it is answering, so
    langchain requires the whole tool call rather than its arguments alone, and
    hands back a message or a command rather than a string.
    """
    return said(raw_call(tool, **args))


def raw_call(tool, **args):
    """The same, kept whole: a `ToolMessage`, or a `Command` that ends the step."""
    return tool.invoke(
        {"name": tool.name, "args": args, "id": f"call-{tool.name}",
         "type": "tool_call"}
    )


def said(answer) -> str:
    """What a tool answered, whether or not it ended its step with it."""
    for message in getattr(answer, "update", {}).get("messages", []):
        return str(message.content)
    return str(getattr(answer, "content", answer))

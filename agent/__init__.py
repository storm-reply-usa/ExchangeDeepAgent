"""The agent, assembled from `your-config/`.

Nothing here imports the AgentCore runtime, so the agent can be built and
inspected without an AWS session; the SDK clients its tools need are imported
inside the tool call.
"""

from agent.build import Folder, Part, build_agent, read_folder, subagent_specs, system_prompt

__all__ = [
    "Folder",
    "Part",
    "build_agent",
    "read_folder",
    "subagent_specs",
    "system_prompt",
]

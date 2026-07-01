from typing_extensions import override

from sweagent.agent.agents import DefaultAgent
from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.codequery.tools import EditTools
from sweagent.sop.registry import OminiRegistryKey
from sweagent.tools.query import parse_command
from sweagent.types import AgentInfo, Trajectory


class AliceAgent(DefaultAgent, AbstractAgentHook):
    """
    A specialized agent for file editing with tracking and summarization features.

    Manages file editing lifecycle with automatic log clearing, summary generation,
    and diff-based change tracking.

    Attributes:
        name (str): Agent identifier set to "alice"
    """

    name: str = "race_validation"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_hook(self)

    @override
    def on_run_start(self):
        action = parse_command(EditTools.clear_edit_log.__name__)
        action.execute(root_path=self.tools.workspace, env=self._env, memory=self.global_memory)

    @override
    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo):
        # First give an overall summary from Alice
        self.generate_summary(OminiRegistryKey.EDIT_SUMMARY)
        summary = self.global_memory.get(OminiRegistryKey.EDIT_SUMMARY)

        # Second summarize the successful patches
        action = parse_command(EditTools.generate_patch.__name__)
        resp = action.execute(root_path=self.tools.workspace, env=self._env, memory=self.global_memory)
        diff = resp.message
        self.global_memory.set(OminiRegistryKey.DIFF, diff)
        summary += f"\n\nBelow are successfully applied patches:\n<diff>\n{diff}\n</diff>\n"

        # Third the summary for failed patches
        # action = parse_command(EditTools.output_failed_patch.__name__)
        # resp = action.execute(root_path=self.tools.workspace, env=self._env, memory=self.global_memory)
        # diff = resp.message
        # summary += f"\nBelow are failed patches with reasons:\n<diff>\n{diff}\n</diff>\n"

        # Write back the summary
        self.global_memory.set(OminiRegistryKey.EDIT_SUMMARY, summary)

        self.logger.info(f"Patching summary for {self.name}:\n{summary}")
        self.as_tool_response = summary

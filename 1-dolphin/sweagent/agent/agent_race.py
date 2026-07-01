from typing_extensions import override

from sweagent.agent.agents import DefaultAgent
from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.codequery.tools import EditTools
from sweagent.sop.registry import OminiRegistryKey
from sweagent.tools.query import parse_command
from sweagent.types import AgentInfo, AgentRunResult, StepOutput, Trajectory, TrajectoryStep

class RaceAgent(DefaultAgent, AbstractAgentHook):
    """
    A specialized agent for race detection.
    Attributes:
        name (str): Agent identifier set to "race_detection"
    """

    name: str = "race_detection"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_hook(self)

    @override
    def on_step_done(self, *, step: StepOutput, info: AgentInfo):
        if step.action == "plan_done":
            self.generate_summary(OminiRegistryKey.PLAN_SUMMARY)
            summary = self.global_memory.get(OminiRegistryKey.PLAN_SUMMARY)
            # reset history for next plan
            self._clear_history()
            self.prepare_prompt()
            self.add_plan_summary_to_history()

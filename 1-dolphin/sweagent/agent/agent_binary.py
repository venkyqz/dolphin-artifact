from typing_extensions import override

from sweagent.agent.agents import DefaultAgent, MultiStageAgent
from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.codequery.tools import EditTools
from sweagent.sop.registry import Memory, OminiRegistryKey
from sweagent.tools.query import ToolCallModel, parse_command
from sweagent.types import (
    AgentInfo,
    StepOutput,
    Trajectory,
)


class BinaryAgent(DefaultAgent):
    """
    An agent analyzing binaries
    """

    name: str = "binary"

    def __init__(self, *args, registry=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_memory: Memory = registry or Memory.get_share_instance()

    @override
    def on_run_start(self):
        action = parse_command(EditTools.clear_edit_log.__name__)
        action.execute(root_path=self.tools.workspace, env=self._env, memory=self.global_memory)

    @override
    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo):
        pass


class BinarySummaryAgent(DefaultAgent):
    """
    An agent summarizing binary
    """

    name: str = "binary_summary"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_hook(self)

    @override
    def on_run_start(self):
        action = parse_command(EditTools.clear_edit_log.__name__)
        action.execute(root_path=self.tools.workspace, env=self._env, memory=self.global_memory)

    @override
    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo):
        pass


class BinaryPlanAgent(DefaultAgent):
    """
    An agent making bug detection plan in binaries
    """

    name: str = "binary_planner"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_hook(self)

    @override
    def on_run_start(self):
        action = parse_command(EditTools.clear_edit_log.__name__)
        action.execute(root_path=self.tools.workspace, env=self._env, memory=self.global_memory)

    @override
    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo):
        pass


class BinaryAnalyzeAgent(DefaultAgent):
    """
    An agent analyze potential bug given plan in binaries
    """

    name: str = "binary_analyze"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_hook(self)

    @override
    def on_run_start(self):
        action = parse_command(EditTools.clear_edit_log.__name__)
        action.execute(root_path=self.tools.workspace, env=self._env, memory=self.global_memory)

    @override
    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo):
        pass


class BinaryValidateAgent(DefaultAgent):
    """
    An agent validate bug reports in binaries
    """

    name: str = "binary_validate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_hook(self)

    @override
    def on_run_start(self):
        action = parse_command(EditTools.clear_edit_log.__name__)
        action.execute(root_path=self.tools.workspace, env=self._env, memory=self.global_memory)

    @override
    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo):
        # TODO: summarize the PoC
        pass

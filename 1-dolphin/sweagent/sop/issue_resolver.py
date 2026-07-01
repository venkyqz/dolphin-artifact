"""
Implement the SOP (Standard Operating Procedures) for multi-agent interactions
"""

import shutil
from pathlib import Path
from typing import Self

from sweagent.agent.agent_binary import BinaryAgent
from sweagent.agent.agents import AbstractAgent, DefaultAgent, get_agent_from_config
from sweagent.agent.config import SOPAgentConfig
from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.run.hooks.abstract import CombinedRunHooks, RunHook
from sweagent.sop.registry import Memory
from sweagent.tools.tools import ToolHandler
from sweagent.types import AgentRunResult
from sweagent.utils.telemetry import get_format_logger


class IssueResolvingSOP(DefaultAgent):
    """
    A team of agents that work together to resolve issues.

    NOTE:
        Though `IssueResolvingTeam` inherits `DefaultAgent`, it only reuses api from agent and make cli entry process more simple.
        `IssueResolvingTeam` is just a workflow but an agent.


    The team consists of:
    1. ScoutAgent: Analyzes and reproduces the issue
    2. ProblemSolvingAgent (Anders): Solves the issue
    3. ReviewerAgent (Carol): Reviews and improves the solution
    """

    name = "issue_resolving_sop"

    def __init__(
        self,
        agents: list[AbstractAgent],
        tools: ToolHandler,
    ):
        super().__init__(templates=None, history_processors=None, tools=tools, model=None)
        self.logger = get_format_logger("swe-agent-sop", emoji="👥")
        self.agents = agents
        self.problem_statement: ProblemStatement = None
        self.output_dir: Path = None
        self._hooks: list[RunHook] = []
        self.global_memory = Memory()
        self._chooks = CombinedRunHooks()

    @classmethod
    def from_config(cls, config: SOPAgentConfig) -> Self:
        config = config.model_copy(deep=True)

        agents = []
        for ac in config.agents:
            agent = get_agent_from_config(ac)
            agents.append(agent)

        # To ensure that all models stay completely independent, we deepcopy the
        # model config, because it lives on as a property in the model, tools, etc.
        return cls(agents=agents, tools=ToolHandler(config.tools))

    def add_hook(self, hook: AbstractAgentHook) -> None:
        """Add a hook to the team."""
        hook.on_init(agent=self)
        self._chooks.add_hook(hook)

        for a in self.agents:
            a.add_hook(hook)

    def get_agent_by_type(self, agent_type: type) -> type | None:
        """Get an agent of a specific type from the team."""
        for agent in self.agents:
            if isinstance(agent, agent_type):
                return agent
        return None

    def run_agent_by_type(self, agent_type: type, previous_res: AgentRunResult = None) -> AgentRunResult | None:
        """Find and run agent by type"""

        agent = self.get_agent_by_type(agent_type)
        if not agent:
            return None
        assert isinstance(agent, DefaultAgent)

        self.logger.info(f"========= Running {agent_type.__name__} agent =========")
        result = agent.run(
            env=self._env,
            problem_statement=self.problem_statement,
            memory=self.global_memory,
            output_dir=self.output_dir / self.problem_statement.id,
        )
        self.logger.info(f"{agent_type.__name__} agent completed")
        return result

    def reset_agent_by_type(self, agent_type: type) -> None:
        """Re-initialize an agent of a specific type from its replay config.

        This is useful when we need to reset an agent's state between runs,
        especially in loops where the same agent runs multiple times.

        Args:
            agent_type: The type of agent to re-initialize
        """
        # Find the agent to reinitialize
        for i, agent in enumerate(self.agents):
            if isinstance(agent, agent_type):
                self.agents[i].reset()
                self.logger.info(f"Reinitialized {agent_type.__name__} agent")
                return

        self.logger.warning(f"No agent of type {agent_type.__name__} found to reset")

    def reset_git_diff(self) -> None:
        """Before doing revision, we need to undo all the changes."""

        # reset git
        self._env.execute_command(["git", "reset", "--hard"], cwd=self._env.get_cwd())
        # NOTE: we can only reset git managed files since reproduce files exist
        self._env.execute_command(["git", "checkout", "HEAD", "--force"], cwd=self._env.get_cwd())

    def run_one(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        memory: Memory = None,
        output_dir: Path = Path("."),
    ) -> AgentRunResult | None:
        self._env = env
        self.output_dir = output_dir
        self.global_memory = memory
        self.problem_statement = problem_statement

        self.setup_tools()

        self.logger.info("Starting team workflow")

        # We should clean the output dir?
        output_dir = self.output_dir / self.problem_statement.id
        if output_dir.exists():
            shutil.rmtree(output_dir)

        res: AgentRunResult | None = None
        res = self.run_agent_by_type(type(self.agents[0]))
        self.logger.info("Team workflow completed")
        return res

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        memory: Memory = None,
        output_dir: Path = Path("."),
    ) -> AgentRunResult | None:
        if len(self.agents) != 1:
            self.logger.warning(f"Multiple agent defined, but we do not define team sop yet.")
        elif len(self.agents) == 0:
            self.logger.warning(f"No agent defined.")
            return None
        return self.run_one(env, problem_statement, memory, output_dir)

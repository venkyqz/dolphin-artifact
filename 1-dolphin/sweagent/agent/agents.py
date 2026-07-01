from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Self

import yaml
from jinja2 import Template
from pydantic import BaseModel
from tenacity import RetryError
from typing_extensions import override
from unidiff import UnidiffParseError

from sweagent import __version__, get_agent_commit_hash, get_rex_commit_hash, get_rex_version
from sweagent.agent.action_sampler import AbstractActionSampler, ActionSamplerConfig
from sweagent.agent.config import (
    AgentConfig,
    DefaultAgentConfig,
    MultiStageAgentConfig,
    StageAgentConfig,
    StageConfig,
    TemplateConfig,
)
from sweagent.agent.history_processors import HistoryProcessor
from sweagent.agent.hooks.abstract import AbstractAgentHook, CombinedAgentHook
from sweagent.agent.models import (
    AbstractModel,
    GenericAPIModelConfig,
    HumanModel,
    HumanThoughtModel,
    LiteLLMModel,
    ModelConfig,
    get_model,
)
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.agent.types import ToolResponse
from sweagent.codequery.tools import EditTools
from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import (
    ContentPolicyViolationError,
    ContextWindowExceededError,
    CostLimitExceededError,
    FormatError,
    TotalCostLimitExceededError,
    _AgentInterrupt,
    _NoSuchToolException,
)
from sweagent.sop.registry import Memory, OminiRegistryKey
from sweagent.tools.commands import Command
from sweagent.tools.parsing import (
    ActionOnlyParser,
    BaseCommandParser,
    ThoughtActionParser,
    CommandActionOnlyParser,
)
from sweagent.tools.query import ToolCallModel
from sweagent.tools.tools import ToolHandler
from sweagent.types import AgentInfo, AgentRunResult, StepOutput, Trajectory, TrajectoryStep
from sweagent.utils.config import _strip_abspath_from_dict
from sweagent.utils.markdown import MarkdownParser
from sweagent.utils.patch_formatter import PatchFormatter
from sweagent.utils.telemetry import get_logger, logger
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError, SwerexException
from swerex.runtime.remote import RemoteRuntime

RETRY_WITH_OUTPUT_TOKEN = "###SWE-AGENT-RETRY-WITH-OUTPUT###"
RETRY_WITHOUT_OUTPUT_TOKEN = "###SWE-AGENT-RETRY-WITHOUT-OUTPUT###"
EXIT_FORFEIT_TOKEN = "###SWE-AGENT-EXIT-FORFEIT###"


class _BlockedActionError(Exception):
    """Raised when the agent's action is blocked"""


class _RetryWithOutput(Exception):
    """Used for internal control flow"""


class _RetryWithoutOutput(Exception):
    """Used for internal control flow"""


class _ExitForfeit(Exception):
    """Used for internal control flow"""


class _TotalExecutionTimeExceeded(Exception):
    """Used for internal control flow"""


class AbstractAgent:
    name: str = None
    # we can use this to find a named agent class, and call from_config to init it.
    _named_cls: dict[str, type[AbstractAgent]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # if we define name in subclass, we save it into mapping
        name = cls.name
        if name:
            if name in cls._named_cls:
                raise ValueError("different agents with same name")
            if name:
                cls._named_cls[name] = cls

    def __init__(self, *args, **kwargs):
        model: AbstractModel
        replay_config: BaseModel | None
        logger: logging.Logger

    @classmethod
    def from_config(cls, config: AgentConfig) -> Self: ...

    def add_hook(self, hook: AbstractAgentHook) -> None: ...

    def get_trajectory_data(self) -> dict[str, Any]: ...

    def step(self) -> StepOutput: ...

    def run(self, *args, **kwargs) -> AgentRunResult: ...


class DefaultAgent(AbstractAgent):
    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: ToolHandler,
        history_processors: list[HistoryProcessor],
        model: AbstractModel,
        max_requeries: int = 3,
        name: str = "main",
        _catch_errors: bool = True,
        _always_require_zero_exit_code: bool = False,
        action_sampler_config: ActionSamplerConfig | None = None,
        # extend
        global_memory: Memory = None,
    ):
        """The agent handles the behaviour of the model and how it interacts with the environment.

        To run the agent, either call `self.run` or `self.setup` and then `self.step` in a loop.
        """

        """memory used between different stages of the agent"""
        self.local_memory: Memory = Memory()
        """memory used between different agents"""
        self.global_memory: Memory | None = global_memory
        self._catch_errors = _catch_errors
        self._always_require_zero_exit_code = _always_require_zero_exit_code
        self.name = name
        self.output_dir = Path()

        if model:
            if isinstance(model, GenericAPIModelConfig):
                self.model = get_model(model, tools=tools.config)
            elif isinstance(model, AbstractModel):
                self.model = model
            else:
                raise TypeError("should use model config or model")
        else:
            self.model = None

        self.templates = templates
        self.tools: ToolHandler = tools

        if isinstance(self.model, HumanThoughtModel):
            self.tools.config.parse_function = ThoughtActionParser()
        elif isinstance(self.model, HumanModel):
            self.tools.config.parse_function = CommandActionOnlyParser()
        self.history_processors = history_processors
        self.max_requeries = max_requeries
        self.logger = get_logger("swea-agent", emoji="🤠")
        # Set in run method
        self._env: SWEEnv | None = None
        self._problem_statement: ProblemStatement | ProblemStatementConfig | None = None
        # individual traj file
        self.traj_path: Path | None = None
        # the path to traj file that contains all agent results
        self.combined_traj_path: Path | None = None
        self.combined_traj_content: str = ""

        # Add instance-level run counter
        self.instance_runs: int = 0

        #: The following three attributes collect the information about how the agent
        #: solved the problem.
        self.history = []
        self._trajectory = []
        self.info = AgentInfo()

        self._chook = CombinedAgentHook()

        self._replay_config: BaseModel | None = None
        """This can be set to a RunSingleConfig from the Run instance whenever possible.
        It can be used to replay the agent's trajectory in an environment.
        """

        self._action_sampler: AbstractActionSampler | None = None
        if action_sampler_config is not None:
            self._action_sampler = action_sampler_config.get(self.model, self.tools)

        #: Count how many timeout errors have occurred consecutively. Kills agent
        #: after 5 of them.
        self._n_consecutive_timeouts = 0
        # Total time spent in environment execution (commands, tools etc.)
        self._total_execution_time = 0.0
        # Total time spent in agent processing (thinking, planning etc.), including execution time
        self._total_agent_run_time = 0.0

        self._tool_stats: dict[str, int] = {}

        self.markdown_parser = MarkdownParser()
        self.as_tool_response: str = ""
        self.as_tool_arguments: str = ""

    def _update_tool_stats(self, tool_name: str) -> None:
        """Update tool usage statistics.

        Args:
            tool_name: Name of the tool that was called
        """
        if tool_name in self._tool_stats:
            self._tool_stats[tool_name] += 1
        else:
            self._tool_stats[tool_name] = 1

        self.logger.debug(f"Tool '{tool_name}' called. Total calls: {self._tool_stats[tool_name]}")

    @classmethod
    def from_config(cls, config: DefaultAgentConfig) -> Self:
        # To ensure that all models stay completely independent, we deepcopy the
        # model config, because it lives on as a property in the model, tools, etc.
        config = config.model_copy(deep=True)
        model = get_model(config.model, config.tools)
        return cls(
            templates=config.templates,
            tools=ToolHandler(config.tools),
            history_processors=config.history_processors,
            model=model,
            max_requeries=config.max_requeries,
            action_sampler_config=config.action_sampler,
            name=config.name,
        )

    def add_hook(self, hook: AbstractAgentHook) -> None:
        """Add hook to agent"""
        hook.on_init(agent=self)
        self._chook.add_hook(hook)

    # Properties
    # ----------

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def replay_config(self) -> BaseModel | None:
        return self._replay_config

    @replay_config.setter
    def replay_config(self, value: BaseModel):
        # Do import here to avoid circular dependency
        from sweagent.run.run_single import RunSingleConfig

        self._replay_config = RunSingleConfig.model_validate(_strip_abspath_from_dict(value.model_dump()))

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the history of the agent for this attempt since the last reset,
        processed through all history processors.
        """
        filtered_history = [entry for entry in self.history if entry["agent"] == self.name]  # type: ignore

        # Chain the history processors
        messages = filtered_history
        for processor in self.history_processors:
            messages = processor(messages)

        return messages  # type: ignore

    # Methods
    # -------

    def memorize(self, step: StepOutput) -> None:
        """This method is used for setting up the memory,
        invoked after each step"""
        pass

    def _clear_history(self) -> None:
        """Clear the message history, for multi-stage agent"""
        self.history = []

    def _append_history(self, item: dict[str, Any]) -> None:
        """Adds an item to the history."""
        item["agent"] = self.name
        item["message_type"] = item.get("message_type", "")
        self._chook.on_query_message_added(**item)
        self.history.append(item)  # type: ignore

    def setup(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:
        """Setup the agent for a new instance. This includes
        formatting the system message and adding demonstrations to the history.

        This method is called by `self.run`.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._problem_statement = problem_statement
        self._env = env
        assert self._env is not None
        assert self._problem_statement is not None
        iid = self._problem_statement.id
        self.logger.info("Setting up agent for instance %s", iid)

        # Save/reset some attributes
        self.traj_path = self.output_dir / (self._problem_statement.id + "." + self.name + ".traj")
        self.combined_traj_path = self.output_dir / (self._problem_statement.id + ".traj")
        self.logger.info("Trajectory will be saved to %s", self.combined_traj_path)

        self._chook.on_tools_installation_started()
        self._chook.on_setup_attempt()
        self.info = AgentInfo()
        self.info["swe_agent_hash"] = get_agent_commit_hash()
        self.info["swe_agent_version"] = __version__
        self.info["swe_rex_version"] = get_rex_version()
        self.info["swe_rex_hash"] = get_rex_commit_hash()
        self._env.set_env_variables({"PROBLEM_STATEMENT": self._problem_statement.get_problem_statement()})
        # reset to the working dir, put it into memory so that prompts can know it
        self.global_memory.set(OminiRegistryKey.WORKING_DIR, self._env.reset_cwd())
        self._chook.on_setup_done()

    def setup_tools(self) -> None:
        """We decouple tool setup from the original setup to make MultiStageAgent use the setup directly"""

        # currently, if using local runtime, we disable tools install
        if isinstance(self._env.deployment.runtime, RemoteRuntime):
            self.tools.install(self._env)

        # for multi-agent interact tool calls
        if self._env.repo:
            self.tools.open_workspace(f"{self._env.local_root}/{self._env.repo.repo_name}")
        else:
            self.tools.open_workspace(f"{self._env.local_root}")

    def prepare_prompt(self) -> None:
        """
        Prepare the prompt for the agent.
        You may override this method to add custom prompt preparation logic.
        """
        self.add_system_message_to_history()
        self.add_tool_templates_message_to_history()
        self.add_demonstrations_to_history()
        self.add_instance_template_to_history(state=self.tools.get_state(self._env))

    def add_tool_templates_message_to_history(self) -> None:
        """Add tool templates message to history."""
        if self.templates.tool_template:
            tool_templates_msg = Template(self.templates.tool_template).render(**self._get_format_dict())
            self.logger.info(f"Tool templates ({self.name})\n{tool_templates_msg}")
            self._append_history(
                {"role": "system", "content": tool_templates_msg, "agent": self.name, "message_type": "system_prompt"}
            )

    def add_system_message_to_history(self) -> None:
        """Add system message to history"""
        assert self._problem_statement is not None
        system_msg = Template(self.templates.system_template).render(**self._get_format_dict())
        self.logger.info(f"SYSTEM ({self.name})\n{system_msg}")
        self._append_history(
            {"role": "system", "content": system_msg, "agent": self.name, "message_type": "system_prompt"}
        )

    def add_demonstrations_to_history(self) -> None:
        """Add demonstrations to history"""
        for demonstration_path in self.templates.demonstrations:
            self._add_demonstration_to_history(demonstration_path)

    def _add_demonstration_to_history(self, demonstration_path: Path) -> None:
        """Load demonstration from disk and add to history"""
        if self.templates.demonstration_template is None and not self.templates.put_demos_in_history:
            msg = "Cannot use demonstrations without a demonstration template or put_demos_in_history=True"
            raise ValueError(msg)

        # Load history
        self.logger.info(f"DEMONSTRATION: {demonstration_path}")
        _demo_text = Path(demonstration_path).read_text()
        if demonstration_path.suffix == ".yaml":
            demo_history = yaml.safe_load(_demo_text)["history"]
        else:
            demo_history = json.loads(_demo_text)["history"]

        if self.templates.put_demos_in_history:
            # Add demonstrations to history step-by-step
            for entry in demo_history:
                if entry["role"] != "system":
                    entry["is_demo"] = True
                    self._append_history(entry)
        else:
            # Add demonstration as single message to history
            demo_history = [entry for entry in demo_history if entry["role"] != "system"]
            demo_message = "\n".join([entry["content"] for entry in demo_history])
            assert self.templates.demonstration_template is not None
            demonstration = Template(self.templates.demonstration_template).render(demonstration=demo_message)
            self._append_history(
                {
                    "agent": self.name,
                    "content": demonstration,
                    "is_demo": True,
                    "role": "user",
                    "message_type": "demonstration",
                },
            )

    def _get_format_dict(self, **kwargs) -> dict[str, Any]:
        """Get the dictionary of key value pairs used to format the templates

        Args:
            **kwargs: additional keyword arguments to be added to the format dictionary
        """
        assert self._problem_statement is not None
        assert self._env is not None

        """We can use memory inside the template, specifically, the memory has a dict structure 
        that can be set by other code, the template can use the {{key}} to automatically include the 
        memory content into the prompt. The local and global memory are placed together, try not to 
        have conflict keys."""
        input_dict: dict[str, str] = dict()
        if self.local_memory:
            input_dict.update(self.local_memory.kv)
        if self.global_memory:
            input_dict.update(self.global_memory.kv)

        # state and global_memory could both have "diff" key, causing: dict() got multiple values for keyword argument 'diff'
        input_dict.update(kwargs)

        return dict(
            command_docs=self.tools.config.command_docs,
            command_tips=self.tools.config.command_tips,
            command_json_schema=self.tools.config.command_json_schema,
            as_tool_arguments=self.as_tool_arguments,
            **self.tools.config.env_variables,
            **input_dict,
            problem_statement=self._problem_statement.get_problem_statement(),
            repo=self._env.repo.repo_name if self._env.repo is not None else "",
            **self._problem_statement.get_extra_fields(),
        )

    def _add_templated_messages_to_history(
        self, templates: list[str], tool_call_ids: list[str] | None = None, **kwargs: str | int | None
    ) -> None:
        """Populate selected template(s) with information (e.g., issue, arguments, state)
        and add to history.

        Args:
            templates: templates to populate and add to history
            tool_call_ids: tool call ids to be added to the history
            **kwargs: keyword arguments to be passed to the templates (in addition to the
                ones in `self._get_format_dict`)
        """
        messages = []

        format_dict = self._get_format_dict(**kwargs)
        for template in templates:
            try:
                messages.append(Template(template).render(**format_dict))
            except KeyError:
                self.logger.debug("The following keys are available: %s", format_dict.keys())
                raise

        message = "\n".join(messages)

        # We disable syntax highlighting here, because some inputs can lead to a complete cross-thread
        # freeze in the agent. See https://github.com/SWE-agent/SWE-agent/issues/901 .
        self.logger.info(f"🤖 MODEL INPUT\n{message}", extra={"highlighter": None})
        history_item: dict[str, Any] = {
            "role": "user",
            "content": message,
            "agent": self.name,
            "message_type": "observation",
        }
        if tool_call_ids:
            assert len(tool_call_ids) == 1, "This should be ensured by the FunctionCalling parse method"
            history_item["role"] = "tool"
            history_item["tool_call_ids"] = tool_call_ids
        self._append_history(history_item)

    def add_step_to_history(self, step: StepOutput) -> None:
        """Adds a step (command that was run and output) to the model history"""
        self._append_history(
            {
                "role": "assistant",
                "content": step.output,
                "thought": step.thought,
                "action": step.action,
                "agent": self.name,
                "tool_calls": step.tool_calls,
                "message_type": "action",
            },
        )

        elided_chars = 0
        if step.observation.strip() == "":
            # Show no output template if observation content was empty
            templates = [self.templates.next_step_no_output_template]
        elif len(step.observation) > self.templates.max_observation_length:
            templates = [self.templates.next_step_truncated_observation_template]
            elided_chars = len(step.observation) - self.templates.max_observation_length
            step.observation = step.observation[: self.templates.max_observation_length]
        else:
            # Show standard output template if there is observation content
            templates = [self.templates.next_step_template]
        self._add_templated_messages_to_history(
            templates,
            action=step.action,
            observation=step.observation,
            elided_chars=elided_chars,
            max_observation_length=self.templates.max_observation_length,
            tool_call_ids=step.tool_call_ids,
            **step.state,
        )

    def add_instance_template_to_history(self, state: dict[str, str]) -> None:
        """Add observation to history, as well as the instance template or demonstrations if we're
        at the start of a new attempt.
        """
        templates: list[str] = []
        # Determine observation template based on what prior observation was
        assert self.history[-1]["role"] == "system" or self.history[-1].get("is_demo", False)
        # Show instance template if prev. obs. was initial system message
        templates = [self.templates.instance_template]
        if self.templates.strategy_template is not None:
            templates.append(self.templates.strategy_template)

        self._add_templated_messages_to_history(templates, **state)  # type: ignore

    def get_agent_name_in_trajectory_data(self) -> str:
        """Returns the agent name used in the trajectory data, can be overridden by subclass"""
        return self.name

    def get_trajectory_filename(self) -> str:
        """construct the name for saving the trajectory data, consider instance count"""
        return self.output_dir / (
            self._problem_statement.id + "." + self.name + "." + str(self.instance_runs) + ".traj"
        )

    def get_trajectory_data(self) -> dict[str, Any]:
        """Get all data that we save in .traj files."""

        assert self._env is not None
        # The deepcopy here is important because else the
        # data["info"]["model_stats"] update will create havoc!
        attempt_data = copy.deepcopy(
            {
                "agent_name": self.get_agent_name_in_trajectory_data(),
                "trajectory": self.trajectory,
                "history": self.history,
                "info": self.info,
                "local_memory": self.local_memory.dump() or {},
                "global_memory": self.global_memory.dump() or {},
            }
        )
        attempt_data["replay_config"] = (
            json.loads(self.replay_config.model_dump_json()) if self.replay_config is not None else None
        )
        attempt_data["environment"] = self._env.name
        return attempt_data

    def get_combined_trajectory_data(self, traj_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Get combined trajectory data from multiple agents"""

        # refresh cached content if it is empty
        if (
            self.combined_traj_content == ""
            and self.combined_traj_path is not None
            and self.combined_traj_path.exists()
        ):
            self.combined_traj_content = self.combined_traj_path.read_text().strip()

        if self.combined_traj_content == "":
            # no content, add new things
            return [traj_data]
        else:
            # Read existing data
            existing_data = json.loads(self.combined_traj_content)
            # make existing data a list for future combine
            if isinstance(existing_data, list):
                ...
            else:
                existing_data = [existing_data]
            # Find and replace the last trajectory for this agent, because one agent can be run multiple times
            if existing_data[-1].get("agent_name") == self.get_agent_name_in_trajectory_data():
                existing_data[-1] = traj_data
            else:
                existing_data.append(traj_data)
            # Write back the combined data
            return existing_data

    def save_trajectory(
        self,
    ) -> None:
        """Save the trajectory to disk.
        This includes the history, the environment state, and the model stats.
        """
        # Add total execution time to info before saving
        self.info["total_execution_time"] = self._total_execution_time
        self.info["total_agent_run_time"] = self._total_agent_run_time
        self.info["tool_stats"] = self._tool_stats

        data = self.get_trajectory_data()
        self.traj_path = Path(self.get_trajectory_filename())
        assert self.traj_path is not None
        self.traj_path.write_text(json.dumps(data, indent=2))

        # If combined_path is provided, append to it
        combined_data = self.get_combined_trajectory_data(data)
        self.combined_traj_content = json.dumps(combined_data, indent=2)
        self.combined_traj_path.write_text(self.combined_traj_content)

    def get_model_requery_history(
        self, error_template: str, *, output: str, **kwargs: str | int | float | bool | None
    ) -> list[dict[str, str]]:
        """Ask the model to correct after a hitting one of the following errors:

        1. Malformatted output (could not parse action)
        2. Blocked action (command is on the blocklist)
        3. Bash command syntax error

        At the time this function is called, the proposed action and observation are not part of the history
        yet.

        This function adds temporary history based on the error template and queries the model.
        If the model is able to correct itself, the records of the mistakes will not be part of the history
        (but they are saved in the trajectory).

        Args:
            error_template: error template
            output: model output
            **kwargs: keyword arguments to be passed to the error template

        Returns:
            model output after requery
        """
        format_dict = {**kwargs, **self._get_format_dict()}
        error_template = Template(error_template).render(**format_dict)

        self.logger.warning(f"{error_template}")

        return self.messages + [
            {"role": "assistant", "content": output, "agent": self.name},
            {"role": "user", "content": error_template, "agent": self.name},
        ]

    def attempt_autosubmission_after_error(self, step: StepOutput) -> StepOutput:
        """For most exceptions, we attempt to still extract the patch and submit that.
        This means we send the `submit` command to the runtime and parse the output.
        """
        self.logger.warning("Attempting autosubmission after error")
        step = step.model_copy(deep=True)
        step.done = True
        assert self._env is not None
        if not asyncio.run(self._env.deployment.is_alive(timeout=10)):
            # The agent is dead. This is very bad. Maybe we can take a 'diff' that was saved
            # for a previous step? (if running with diff in tools)
            self.logger.error("Runtime is no longer alive")
            try:
                last_trajectory_step = self.trajectory[-1]
            except IndexError:
                self.logger.info("No last trajectory step to extract patch from")
                return step
            if "diff" not in last_trajectory_step["state"]:
                self.logger.info("No diff in last trajectory step state, cannot autosubmit")
                return step
            diff = last_trajectory_step["state"]["diff"]
            self.logger.info("Using diff from last trajectory step to autosubmit")
            step.submission = diff
            if step.submission:
                step.observation = "Environment died unexpectedly. Exited (autosubmitted)"
                step.exit_status = f"submitted ({step.exit_status})"
            else:
                self.logger.info("Diff from last traj step empty.")
            return step
        # Let us manually run the submission command and collect the output
        if self.tools:
            try:
                action = ToolCallModel(command=EditTools.submit.__name__, tool=EditTools.submit.__name__)
                res = self.tools.invoke(action, env=self._env, memory=self.global_memory)
                step = step.model_copy(deep=True)
                step.submission = res.submission
                step.done = True
            except Exception:
                self.logger.error("Failed to call submit api tool")
        else:
            # NOTE: we ignore path `.agent` since they comes from agent and unrelated to issue resolve patch
            submission_command = "git add -A && git diff --cached -- ':!.agent' > /root/model.patch"
            self.logger.info("Executing submission command %s in %s", submission_command, self._env.get_cwd())
            try:
                self._env.execute_command(submission_command, check=True, cwd=self._env.get_cwd())
            except Exception as e:
                self.logger.error("Failed to execute submission command, got %s", e)
            # There's still hope for the submission, because the `/root/model.patch` file might have been
            # generated by the state command
            step = self.handle_submission(step, observation="", force_submission=True)
        if step.submission:
            self.logger.info("Exiting with autosubmission")
            step.observation = "Exited (autosubmitted)"
        return step

    def handle_submission(self, step: StepOutput, *, observation="", force_submission: bool = False) -> StepOutput:
        """Check if there was a submission in the observation and handle it.

        Args:
            step:
            observation: If specified, will use this rather than stepobservation
            force_submission: If True, will always submit even if no submission is found

        Returns:
            step: step with submission and observation updated (if submission was found)
        """
        step = step.model_copy(deep=True)
        assert self.tools is not None
        is_submission = self.tools.check_for_submission_cmd(observation or step.observation)
        if is_submission or force_submission:
            assert self._env is not None
            try:
                submission = self._env.read_file("/root/model.patch", encoding="utf-8", errors="backslashreplace")
            except FileNotFoundError:
                self.logger.warning("Submission file not found, no submission was made")
                return step
            except Exception as e:
                self.logger.exception("Failed to read submission file, got %s", e)
                return step
            if submission.strip() != "":
                step.submission = submission
            else:
                step.submission = None
            step.observation = submission
            if not step.exit_status:
                step.exit_status = "submitted"
            elif step.submission:
                step.exit_status = f"submitted ({step.exit_status})"
            step.done = True
            self.logger.info(f"Found submission: {submission}")
        return step

    def _get_edited_files_with_context(self, patch: str) -> dict[str, str]:
        """Get the edited files with context from the patch"""
        assert self._env is not None
        try:
            if self._env.repo is None:
                pf = None
            else:
                pf = (
                    PatchFormatter(
                        patch,
                        read_method=lambda path: self._env.read_file(
                            PurePosixPath(f"{self._env.remote_root}") / self._env.repo.repo_name / path
                        ),  # type: ignore[attr-defined]
                    )
                    if patch
                    else None
                )
        except UnidiffParseError:
            self.logger.error("Failed to parse patch with unidiff. Some variables will be empty.")
            pf = None
            # We still need to populate the variables
        out = {}
        for context_length in [30, 50, 70]:
            value = "Empty. No edited files found."
            if pf is not None:
                value = pf.get_files_str(original=False, context_length=context_length)
            out[f"edited_files{context_length}"] = value
        return out

    @contextmanager
    def _execution_timer(self, step: StepOutput):
        """Context manager for tracking execution time."""
        execution_t0 = time.perf_counter()
        try:
            yield
        finally:
            step.execution_time = time.perf_counter() - execution_t0
            self._total_execution_time += step.execution_time

    @contextmanager
    def _run_timer(self):
        """Context manager for tracking execution time."""
        execution_t0 = time.perf_counter()
        try:
            yield
        finally:
            run_time = time.perf_counter() - execution_t0
            self._total_agent_run_time = run_time

    def handle_action(self, step: StepOutput) -> StepOutput:
        """Runs an action proposed by the agent in the environment and returns the corresponding output.

        Args:
            action: command to run in bash shell
            output: output from model (only used for error handling)

        Returns:
            action_execution_output: action execution output
        """
        if self.tools.should_block_action(step.action):
            raise _BlockedActionError()

        if step.action.strip() == "exit":
            self.logger.info("Exiting agent")
            step.done = True
            step.observation = "Exited"
            step.exit_status = "exit_command"
            assert self._env is not None
            step.state = self.tools.get_state(env=self._env)  # for history
            return step

        assert self._env is not None
        self._chook.on_action_started(step=step)

        with self._execution_timer(step):
            run_action: str = self.tools.guard_multiline_input(step.action).strip()
            try:
                step.observation = self._env.communicate(
                    input=run_action,
                    timeout=self.tools.config.execution_timeout,
                    check="raise" if self._always_require_zero_exit_code else "ignore",
                )
            except CommandTimeoutError:
                try:
                    if self._n_consecutive_timeouts >= self.tools.config.max_consecutive_execution_timeouts:
                        msg = "Exiting agent due to too many consecutive execution timeouts"
                        self.logger.critical(msg)
                        raise
                    self._env.interrupt_session()
                    self._n_consecutive_timeouts += 1
                except Exception as f:
                    self.logger.exception("Failed to interrupt session after command timeout: %s", f, exc_info=True)
                    raise
                step.observation = Template(self.templates.command_cancelled_timeout_template).render(
                    **self._get_format_dict(),
                    timeout=self.tools.config.execution_timeout,
                    command=run_action,
                )
            else:
                self._n_consecutive_timeouts = 0

        self._chook.on_action_executed(step=step)
        step.state = self.tools.get_state(env=self._env)

        if RETRY_WITH_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITH_OUTPUT_TOKEN, "")
            raise _RetryWithOutput()
        elif RETRY_WITHOUT_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITHOUT_OUTPUT_TOKEN, "")
            raise _RetryWithoutOutput()
        elif EXIT_FORFEIT_TOKEN in step.observation:
            raise _ExitForfeit()

        return self.handle_submission(step)

    def forward(self, history: list[dict[str, str]]) -> StepOutput:
        """Forward the model without handling errors.

        All exceptions raised will contain the `StepOutput` object
        with some of the attributes set.

        Args:
            history: history to query the model with

        Returns:
            step_output: step output
        """
        if self._total_execution_time > self.tools.config.total_execution_timeout:
            raise _TotalExecutionTimeExceeded()

        # we continuously add actions, output etc. to the step object
        # because some of the specific exception handling requires some of these
        # attributes (e.g., if we want to requery the model for a bash syntax error, we
        # need to have the previous model output to format the requery template)
        step = StepOutput()
        try:
            # Forward model and get actions
            self._chook.on_model_query(messages=history, agent=self.name)
            # todo: Add all options to the extra info
            if self._action_sampler is not None:
                assert self._problem_statement is not None
                best = self._action_sampler.get_action(
                    problem_statement=self._problem_statement,
                    trajectory=self.trajectory,
                    history=history,
                )
                output = best.completion
                # todo: Handle history and trajectory
                step.extra_info.update(best.extra_info)
            else:
                output = self.model.query(history)  # type: ignore

            # MUST set output first
            step.output = output["message"]

            # we process both internal (in-process) tool & external (shell) tool
            # should bypass next steps
            step = self.interact(step, output)

            return step
        except Exception as e:
            if step.action == step.thought == "":
                # Probably the parsing failed/no action included. Let's still fill in thought
                # so that trajectory viewers have something to show us for this step.
                step.thought = step.output
            # Attach the step object to the exception
            e.step = step  # type: ignore
            raise

    def interact(self, step: StepOutput, output: dict) -> StepOutput | None:
        parser = self.tools.config.parse_function
        message = output.get("message", "")

        if isinstance(parser, BaseCommandParser):
            # FIXME: we only support one command in once
            thought, action = parser(output, self.tools.commands)

            # note this is important for setting correct role and content for history message inside _add_templated_messages_to_history!
            if output.get("tool_calls") is not None:
                step.tool_call_ids = [call["id"] for call in output["tool_calls"]]
                step.tool_calls = output["tool_calls"]

            self._chook.on_actions_generated(step=step)

            try:
                command: Command = self.tools.search_command(action.tool)

                # Update tool statistics
                self._update_tool_stats(command.name)

                if command.is_agent:
                    agent = self.tools.get_agent(command)

                    with self._execution_timer(step):
                        self.global_memory.set(OminiRegistryKey.TOOL_AGENT_ARGS, action.kwargs)
                        result: AgentRunResult = agent.run_as_tool(env=self._env,
                                                                   problem_statement=self._problem_statement,
                                                                   as_tool_arguments=action.kwargs,
                                                                   memory=self.global_memory,
                                                                   output_dir=self.output_dir)

                        tool_resp = ToolResponse(
                            status="success",
                            message=result.response,
                            command=step.action,
                            completion=False,
                            submission=None,
                        )

                    step.action = action.command if action.command != "" else action.tool
                    step.output = message
                    step.observation = tool_resp.message
                    step.done = False

                    self.logger.info(f"💭 THOUGHT\n{step.thought or step.output}\n\n")
                    self.logger.info(f"🎬 ACTION\n{step.action.strip()}\n\n")
                    self.logger.info(f"Agent Response:\n{step.observation}")

                    return step

                elif command.is_api:
                    # TODO: we must define a api to call external tools
                    with self._execution_timer(step):
                        tool_resp: ToolResponse = self.tools.invoke(action, env=self._env, memory=self.global_memory)

                    step.output = message
                    step.action = action.command if action.command != "" else action.tool
                    step.observation = tool_resp.message
                    step.done = tool_resp.completion

                    self.logger.info(f"💭 THOUGHT\n{step.thought or step.output}\n\n")
                    self.logger.info(f"🎬 ACTION\n{step.action.strip()}\n\n")
                    self.logger.info(f"Tool Response:\n{step.observation}")

                    if action.tool == EditTools.submit.__name__:
                        step.submission = step.observation
                        if not step.exit_status:
                            step.exit_status = "submitted"
                        elif step.submission:
                            step.exit_status = f"submitted ({step.exit_status})"
                        # step.done = True
                        self.logger.info(f"Found submission: {step.submission}")

                    return step
                elif command.is_shell:
                    step.action = action.command if action.command != "" else action.tool
                    step.output = message

                    self._chook.on_actions_generated(step=step)

                    self.logger.info(f"💭 THOUGHT\n{step.thought or step.output}\n\n")
                    self.logger.info(f"🎬 ACTION\n{step.action.strip()}\n\n")
                    # FIXME: for env interact with shell, aka. ACI, we use original logic for current version
                    step = self.handle_action(step)

                    return step
                else:
                    # FIXME: for bash command and so on, they are injected in dynamic way without command type info.
                    self.logger.debug("Missing command type, run in default env")

                    step.action = action.command if action.command != "" else action.tool
                    step.output = message

                    self._chook.on_actions_generated(step=step)

                    self.logger.info(f"💭 THOUGHT\n{step.thought or step.output}\n\n")
                    self.logger.info(f"🎬 ACTION\n{step.action.strip()}\n\n")

                    step = self.handle_action(step)
                    return step
            except _NoSuchToolException:
                self.logger.info("No such tool.")
                return step
        else:
            thought, action = parser(output, self.tools.commands)
            if output.get("tool_calls") is not None:
                step.tool_call_ids = [call["id"] for call in output["tool_calls"]]
                step.tool_calls = output["tool_calls"]
            step.action = action.command if action.command != "" else action.tool
            step.output = message
            step.thought = thought

            self._chook.on_actions_generated(step=step)

            self.logger.info(f"💭 THOUGHT\n{step.thought}\n\n")
            self.logger.info(f"🎬 ACTION\n{step.action.strip()}\n\n")

            # FIXME: for env interact with shell, aka. ACI, we use original logic for current version
            step = self.handle_action(step)
            return step

    def forward_without_action(self, history: list[dict[str, str]]) -> StepOutput:
        """Forward the model without handling errors, and without action requirement

        Args:
            history: history to query the model with

        Returns:
            step_output: step output
        """
        if self._total_execution_time > self.tools.config.total_execution_timeout:
            raise _TotalExecutionTimeExceeded()

        step = StepOutput()
        try:
            # Forward model and get response
            self._chook.on_model_query(messages=history, agent=self.name)
            output = self.model.query(history)
            step.output = output["message"]
            self.logger.info(f"Model Response:\n{step.output}")
            return step
        except Exception:
            raise

    def forward_with_handling(self, history: list[dict[str, str]]) -> StepOutput:
        """Forward the model and handle errors, requerying the model if we can.
        For example, if the model outputs a bash command that has syntax errors,
        we will not execute it but requery the model for a corrected command.

        Note: This will update the trajectory, but not the history.

        Args:
            history: history to forward

        Returns:
            step_output: step output
        """

        def handle_error_with_autosubmission(exit_status: str, message: str) -> StepOutput:
            """Attempts to autosubmit (extract patch from the environment) and stops the loop."""
            self.logger.warning(message)
            return self.attempt_autosubmission_after_error(
                StepOutput(
                    thought=message,
                    exit_status=exit_status,
                    output=message,
                    done=True,
                )
            )

        def handle_error_with_retry(exception: Exception, template: str, n_requeries: int) -> list[dict[str, str]]:
            """Requeries the model if the error is a format/blocklist/bash syntax error."""
            self.logger.warning("Requerying model after %s (%dth requery)", exception, n_requeries)
            step: StepOutput = getattr(exception, "step", StepOutput())
            self.add_step_to_trajectory(step)
            exception_message = getattr(exception, "message", "")
            if not exception_message:
                try:
                    exception_message = exception.args[0]
                except (IndexError, AttributeError):
                    pass
            return self.get_model_requery_history(
                error_template=template,
                **step.to_template_format_dict(),
                **getattr(exception, "extra_info", {}),
                exception_message=exception_message,
            )

        n_format_fails = 0
        while n_format_fails < self.max_requeries:
            try:
                return self.forward(history)

            # Errors that are raised

            except KeyboardInterrupt:
                raise

            # Errors that cause requery

            except FormatError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e, template=self.tools.config.format_error_template, n_requeries=n_format_fails
                )
            except _BlockedActionError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e, template=self.tools.config.filter.blocklist_error_template, n_requeries=n_format_fails
                )
            except ContentPolicyViolationError:
                self.logger.warning("Content policy violation, trying to resample")
                n_format_fails += 1
                # Try if simply resampling helps here
                pass
            except BashIncorrectSyntaxError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.shell_check_error_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithOutput as e:
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.next_step_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithoutOutput:
                pass
                # Requery with the same template as the last step

            # Errors that cause exit

            except _ExitForfeit:
                self.logger.info("Exiting due to forfeit")
                return handle_error_with_autosubmission(
                    "exit_forfeit",
                    "Exiting due to forfeit",
                )

            except _TotalExecutionTimeExceeded:
                self.logger.exception("Exiting due to total execution time exceeded", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_total_execution_time",
                    "Exit due to total execution time exceeded",
                )

            except CommandTimeoutError:
                self.logger.exception("Exiting due to multiple consecutive command timeouts", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_command_timeout",
                    "Exit due to multiple consecutive command timeouts",
                )

            except ContextWindowExceededError:
                return handle_error_with_autosubmission(
                    "exit_context",
                    "Exit due to context window",
                )
            except TotalCostLimitExceededError:
                raise
            except CostLimitExceededError:
                return handle_error_with_autosubmission(
                    "exit_cost",
                    "Exit due to cost limit",
                )
            except RetryError as e:
                self.logger.exception(f"Exiting due to retry error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_api",
                    f"Exit due to retry error: {e}",
                )
            except SwerexException as e:
                self.logger.exception(f"Exiting due to environment error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_environment_error",
                    f"Exit due to environment error: {e}",
                )
            except _AgentInterrupt as e:
                self.logger.exception(f"Exiting due to agent interrupt: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_agent_interrupt",
                    f"Exit due to agent interrupt: {e}",
                )
            except RuntimeError as e:
                self.logger.exception(f"Exiting due to runtime error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to runtime error: {e}",
                )
            except Exception as e:
                self.logger.exception(f"Exiting due to unknown error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to unknown error: {e}",
                )
        self.logger.exception(
            "Exit due to repeated format/blocklist/bash syntax errors",
            exc_info=True,
        )
        return handle_error_with_autosubmission(
            "exit_format",
            "Exit due to repeated format/blocklist/bash syntax errors",
        )

    def add_step_to_trajectory(self, step: StepOutput) -> None:
        trajectory_step = TrajectoryStep(
            {
                "action": step.action,
                "observation": step.observation,
                "response": step.output,
                "thought": step.thought,
                "execution_time": step.execution_time,
                "state": step.state,
                "messages": self.messages,
                "extra_info": step.extra_info,
            },
        )
        self.trajectory.append(trajectory_step)

    def generate_summary(self, memory_key: str):
        """Add prompt for stage summary to history"""
        if self.templates.stage_done_summary_template != "":
            stage_summary_msg = Template(self.templates.stage_done_summary_template).render(**self._get_format_dict())
            self.logger.info(f"Stage summary prompt ({self.name})\n{stage_summary_msg}")
            self._append_history(
                {"role": "user", "content": stage_summary_msg, "agent": self.name, "message_type": "summary"}
            )
            # query model
            stage_summary: str = self.forward_without_action(self.history).output
            self.global_memory.set(memory_key, stage_summary)
            # we may not need to append the summary to history, just inside the memory
            # append history for displaying inside the trajectory
            self._append_history(
                {
                    "role": "assistant",
                    "content": stage_summary,
                    "agent": self.name,
                    "message_type": "summary",
                },
            )

    def step(self) -> StepOutput:
        """Run a step of the agent. This is a wrapper around `self.forward_with_handling`
        with additional bookkeeping:

        1. Update message history with performed action and observation
        2. Update trajectory with the final executed result
        3. Update the info dictionary

        Returns:
            step_output: step output (same as the output of `self.forward_with_handling`)
        """

        assert self._env is not None
        self._chook.on_step_start()

        n_step = len(self.trajectory) + 1
        self.logger.info("=" * 25 + f" STEP {n_step} {self.name} " + "=" * 25)
        step_output = self.forward_with_handling(self.messages)
        self.add_step_to_history(step_output)
        self.memorize(step_output)

        # NOTE: we always keep a valid submission
        self.info["submission"] = step_output.submission or self.info.get("submission", None)
        self.info["exit_status"] = step_output.exit_status  # type: ignore
        self.info.update(self._get_edited_files_with_context(patch=step_output.submission or ""))  # type: ignore
        self.info["model_stats"] = self.model.stats.model_dump()
        self.info["tool_stats"] = self._tool_stats.copy()

        self.add_step_to_trajectory(step_output)

        self._chook.on_step_done(step=step_output, info=self.info)
        return step_output

    def reset(self) -> None:
        """
        A method to reset state of the agent to run again
        Specifically, the following properties have to be reset:
        1. history
        2. _trajectory
        3. info
        """
        #: The following three attributes collect the information about how the agent
        #: solved the problem.
        self.history = []
        self._trajectory = []
        self.info = AgentInfo()
        # reset tool stats
        self._tool_stats = {}

    def run_as_tool(self,
                    env: SWEEnv,
                    problem_statement: ProblemStatement | ProblemStatementConfig,
                    as_tool_arguments: Dict = None,
                    memory: Memory = None,
                    output_dir: Path = Path(".")
                    ) -> AgentRunResult:
        # FIXME: reset api calls
        model: LiteLLMModel = self.model
        model.stats.api_calls = 0
        self.as_tool_arguments = as_tool_arguments or {}
        res = self.run(env, problem_statement, memory, output_dir)
        self.reset()
        return res

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        memory: Memory = None,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Run the agent on a problem instance. This method contains the
        main loop that repeatedly calls `self._step` until the problem is solved.

        Args:
            setup_args: Arguments to pass to the agent's setup method.
            problem_statement: the txt problem statement to solve.
            env: The environment to run the agent on.
            memory: external memory shared between agents.
            output_dir: Directory to save the trajectory to
        """
        # increment instance run count
        self.instance_runs += 1

        self.global_memory = memory
        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)
        self.setup_tools()

        # we move prepare prompt to here to allow a more flexible setup
        self.prepare_prompt()

        # Run action/observation loop
        self._chook.on_run_start()
        step_output = StepOutput()

        with self._run_timer():
            while not step_output.done:
                step_output = self.step()
                self.save_trajectory()

        # do extra save
        self.save_trajectory()

        self._chook.on_run_done(trajectory=self.trajectory, info=self.info)

        self.logger.info("Trajectory saved to %s", self.combined_traj_path)
        # Here we want to return the "global" information (e.g., submission should
        # be the best submission instead of the last one, etc.), so we get it from the traj file
        data = self.get_trajectory_data()
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"], response=self.as_tool_response)


class MultiStageAgent(DefaultAgent):
    """
    An agent that runs multiple loops of conversions,
    use MultiStepTemplateConfig to configure prompts.
    """

    class StageMarker:
        """A data class used to mark the stage information such as
        index, name, etc."""

        index: int = 0
        name: str = ""

        def __init__(self, index: int, name: str) -> None:
            self.index = index
            self.name = name

    def __init__(
        self,
        *args,
        stage_agents: list[DefaultAgent] = None,
        stages: list[StageMarker] = None,  # extend
        model_config: ModelConfig,  # extend
        **kwargs,
    ):
        """The agent handles the behaviour of the model and how it interacts with the environment.

        To run the agent, either call `self.run` or `self.setup` and then `self.step` in a loop.
        """

        # for multi-stage state
        self.stages = stages
        self.stage_agents: list[DefaultAgent] = stage_agents or []
        self.model_config = model_config

        self.current_stage: StageMarker = None
        # subclass can modify this flag to break the stage loop
        self.break_stage: bool = False
        # skip the current stage
        self.goto_next_stage = False

        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config: MultiStageAgentConfig) -> Self:
        # To ensure that all models stay completely independent, we deepcopy the
        # model config, because it lives on as a property in the model, tools, etc.
        config = config.model_copy(deep=True)
        model = get_model(config.model, config.tools)

        templates: list[TemplateConfig] = []
        tools: list[ToolHandler] = []
        stages: list[MultiStageAgent.StageMarker] = []
        stage_agents: list[DefaultAgent] = []

        for index, stage_config in enumerate(config.stage_configs):
            if isinstance(stage_config, StageAgentConfig):
                stage_config: StageConfig = stage_config.agent

            # we will try to get corresponding agent from config name
            # if no such agent defined, we do auto generated with default behavior
            # in this way, we can simplify agent definition unless we want to do a fine-grained control agent behavior
            try:
                if not stage_config.model:
                    stage_config.model = config.model.model_copy(deep=True)
                stage_agent = get_agent_from_config(stage_config)
            except ValueError:
                logger.warning(
                    "no such agent defined with name {}, auto generated via {}.",
                    stage_config.name,
                    DefaultAgent.__name__,
                )
                if stage_config.model:
                    stage_model = get_model(stage_config.model, stage_config.tools)
                else:
                    stage_model = model
                stage_agent = DefaultAgent(
                    name=stage_config.name,
                    templates=stage_config.templates,
                    tools=ToolHandler(stage_config.tools),
                    model=stage_model,
                    history_processors=config.history_processors,
                )
            stage_agents.append(stage_agent)

        return cls(
            stage_agents=stage_agents,
            templates=templates,
            tools=tools,
            stages=stages,
            history_processors=config.history_processors,
            model=None,  # multi-stage agent do not init model in __init__
            model_config=config.model,
            max_requeries=config.max_requeries,
            action_sampler_config=config.action_sampler,
            name=config.name,
        )

    @override
    def get_agent_name_in_trajectory_data(self) -> str:
        # Return consistent agent name for MultiStageAgent to avoid duplicate entries
        return self.name

    def get_trajectory_filename(self) -> str:
        """construct the name for saving the trajectory data, consider instance count"""
        return self.output_dir / (
            self._problem_statement.id
            + "."
            + self.get_agent_name_in_trajectory_data()
            + "."
            + str(self.instance_runs)
            + ".traj"
        )

    def reset(self) -> None:
        """
        A method to reset state of the agent to run again
        Specifically, the following properties have to be reset:
        1. history
        2. _trajectory
        3. info
        """
        super().reset()
        # subclass can modify this flag to break the stage loop
        self.break_stage = False
        # skip the current stage
        self.goto_next_stage = False

    @override
    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        memory: Memory = None,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """The main entry point of multi-stage agent

        Args:
            setup_args: Arguments to pass to the agent's setup method.
            env: The environment to run the agent on.
            traj_dir: Directory to save the trajectory to
        """
        # increment instance run count
        self.instance_runs += 1

        self.global_memory = memory
        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)

        # Run action/observation loop
        self._chook.on_run_start()
        with self._run_timer():
            # Run multiple stages using stage_agents
            for index, stage_agent in enumerate(self.stage_agents):
                # Create stage marker for compatibility
                stage = MultiStageAgent.StageMarker(index, stage_agent.name)
                self.current_stage = stage

                self._chook.on_stage_start(stage=stage)
                with self._run_timer():
                    if self.break_stage:
                        self.logger.info("Breaking stage %s", stage_agent.name)
                        break
                    if self.goto_next_stage:
                        self.logger.info("Skipping stage %s", stage_agent.name)
                        self.goto_next_stage = False
                        continue

                    self.logger.info(f"Starting stage {stage_agent.name}...")

                    # Run the stage agent
                    stage_agent.run(
                        env=env, problem_statement=problem_statement, memory=self.global_memory, output_dir=output_dir
                    )

                    # Merge stage agent's info and trajectory into main agent
                    # self.info.update(stage_agent.info)
                    # Copy trajectory steps from stage agent
                    # for traj_step in stage_agent.trajectory:
                    #     if traj_step not in self.trajectory:
                    #         self.trajectory.append(traj_step)

                self._chook.on_stage_stop(
                    trajectory=stage_agent.trajectory, info=stage_agent.info, stage=self.current_stage
                )

                # do extra save
                # this might override combined_content inside sub agent
                # self.save_trajectory()

                # reset history for next stage
                self._clear_history()

        self._chook.on_run_done(trajectory=self.trajectory, info=self.info)
        self.logger.info("Trajectory saved to %s", self.combined_traj_path)

        # Here we want to return the "global" information (e.g., submission should
        # be the best submission instead of the last one, etc.), so we get it from the traj file
        data = self.get_trajectory_data()
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])


def get_agent_from_config(config: AgentConfig) -> AbstractAgent:
    # if name given, we construct it from matched named agent class
    if config.name in AbstractAgent._named_cls:
        return AbstractAgent._named_cls[config.name].from_config(config)

    if config is None:
        return None
    elif config.type == "default":
        return DefaultAgent.from_config(config)
    else:
        msg = f"Unknown agent type and name: {config.type} {config.name}"
        raise ValueError(msg)

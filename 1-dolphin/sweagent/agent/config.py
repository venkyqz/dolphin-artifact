from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from simple_parsing.helpers.fields import field

from sweagent.agent.action_sampler import ActionSamplerConfig
from sweagent.agent.history_processors import DefaultHistoryProcessor, HistoryProcessor
from sweagent.agent.models import ModelConfig
from sweagent.tools.commands import Command
from sweagent.tools.tools import ToolConfig
from sweagent.utils.config import _convert_paths_to_abspath
from sweagent.utils.jinja_warnings import _warn_probably_wrong_jinja_syntax
from sweagent.utils.log import get_logger


class DefaultTemplateConfig(BaseModel):
    """This configuration is used to define almost all message templates that are
    formatted by the agent and sent to the LM.
    """

    name: str = ""
    file: str = ""

    system_template: str = ""
    instance_template: str = ""
    next_step_template: str = "Observation: {{observation}}"

    next_step_truncated_observation_template: str = (
        "Observation: {{observation}}<response clipped>"
        "<NOTE>Observations should not exceeded {{max_observation_length}} characters. "
        "{{elided_chars}} characters were elided. Please try a different command that produces less output "
        "or use head/tail/grep/redirect the output to a file. Do not use interactive pagers.</NOTE>"
    )
    """Message template for when the agent's observation was truncated.
    Available variables: `observation`, `max_observation_length`, `elided_chars`
    """

    max_observation_length: int = 100_000
    """Truncate observation to this length if it exceeds it."""

    next_step_no_output_template: str = None  # type: ignore
    """Template for the next step when the last output was empty. Defaults to next_step_template."""

    strategy_template: str | None = None
    tool_template: str | None = None
    one_shot_template: str | None = None
    summary_template: str | None = None

    conversation_templates: list[str] | None = Field(default_factory=list)
    capability_templates: dict[str, Any] | None = Field(default_factory=dict)

    demonstration_template: str | None = None

    demonstrations: list[Path] = field(default_factory=list)
    """Paths to demonstrations. If path is not absolute, it is assumed to be
    relative to the SWE_AGENT_CONFIG_ROOT (if set) or the SWE-agent repository root
    """

    put_demos_in_history: bool = False
    """If True, add demonstration to history instead of as a single message"""

    shell_check_error_template: str = (
        "Your bash command contained syntax errors and was NOT executed. "
        "Please fix the syntax errors and try again. This can be the result "
        "of not adhering to the syntax for multi-line commands. Here is the output of `bash -n`:\n"
        "{{bash_stdout}}\n{{bash_stderr}}"
    )
    """Message template for when the agent's bash command contains syntax errors.
    Available variables: `bash_stdout`, `bash_stderr`
    """

    command_cancelled_timeout_template: str = (
        "The command '{{command}}' was cancelled because it took more than {{timeout}} seconds. "
        "Please try a different command that completes more quickly."
    )
    """Message template for when the agent's command was cancelled because it took too long.
    Available variables: `timeout`, `command`
    """

    stage_done_summary_template: str = ""
    """Message template for when the agent's stage was done for summary."""

    type: Literal["default"] = "default"

    def model_post_init(self, __context):
        self.demonstrations = _convert_paths_to_abspath(self.demonstrations)
        if self.next_step_no_output_template is None:
            self.next_step_no_output_template = self.next_step_template

    @model_validator(mode="after")
    def validate_template_jinja_syntax(self) -> Self:
        template_fields = [field for field in self.model_fields.keys() if field.endswith("_template")]
        for field in template_fields:
            value = getattr(self, field)
            _warn_probably_wrong_jinja_syntax(value)
        return self

    @model_validator(mode="after")
    def warn_models_in_history(self) -> Self:
        if self.put_demos_in_history and self.demonstration_template is not None:
            logger = get_logger("swea-config", emoji="🔧")
            logger.warning("demonstration_template is ignored when put_demos_in_history is True")
        return self


TemplateConfig = DefaultTemplateConfig


class DefaultAgentConfig(BaseModel):
    """This configuration object specifies the behavior of an agent."""

    name: str = "main"
    commands: list[Command] = Field(description="Commands provided by the agent (as tool).", default_factory=list)
    templates: TemplateConfig | None = Field(description="Template options.", default=None)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    history_processors: list[HistoryProcessor] = Field(default_factory=lambda: [DefaultHistoryProcessor()])
    model: ModelConfig = Field(description="Model options.")

    max_requeries: int = 3
    """Maximum number of times to requery the model after an error, such as a
    formatting error, a blocked action, or a bash syntax error.
    """
    action_sampler: ActionSamplerConfig | None = None

    # to support inherit from DefaultAgent
    # type:
    type: str = "default"

    # pydantic config
    model_config = ConfigDict(extra="forbid")


class StageConfig(BaseModel):
    """This is used in MultiStageAgentConfig to specify the part that are different across stages
    This can be extended by adding more field, otherwise will be shared between stages, e.g., history_processors, model"""

    commands: list[Command] = Field(description="Commands provided by the agent (as tool).", default_factory=list)
    templates: TemplateConfig = Field(description="Template options.")
    tools: ToolConfig = Field(default_factory=ToolConfig)
    name: str = Field(description="The name of the stage, used to mark stage-specific methods.")
    history_processors: list[HistoryProcessor] = Field(default_factory=lambda: [DefaultHistoryProcessor()])
    model: ModelConfig | None = Field(description="Model options.", default=None)
    max_requeries: int = 3
    action_sampler: ActionSamplerConfig | None = None

    type: str = "default"


class StageAgentConfig(BaseModel):
    agent: StageConfig


class MultiStageAgentConfig(BaseModel):
    """This configuration object specifies the behavior of multiple agents."""

    name: str = "multi-main"
    stage_configs: list[StageAgentConfig] = Field(
        default_factory=list, description="Configs for multiple stages inside one agent."
    )
    # history_processors: List[HistoryProcessor] = Field(default_factory=lambda: [DefaultHistoryProcessor()])
    model: ModelConfig = Field(description="Model options.")

    max_requeries: int = 3
    """Maximum number of times to requery the model after an error, such as a
    formatting error, a blocked action, or a bash syntax error.
    """
    action_sampler: ActionSamplerConfig | None = None

    type: str = "multi"


class SOPAgentConfig(BaseModel):
    name: str
    agents: list[DefaultAgentConfig | MultiStageAgentConfig] = Field(
        default_factory=list, description="Team including multiple agents"
    )
    model: ModelConfig = Field(description="Model options.")
    tools: ToolConfig = Field(default_factory=ToolConfig)

    type: str = "sop"


AgentConfig = Annotated[DefaultAgentConfig | MultiStageAgentConfig | SOPAgentConfig, Field(union_mode="left_to_right")]

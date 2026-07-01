from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Union, Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from sweagent.codequery.tools import BaseToolManager
from sweagent.exceptions import _NoSuchToolException
from sweagent.tools import tool_desc
from sweagent.tools.commands import Command
from sweagent.utils.config import _convert_path_to_abspath, load_agent_yaml_config


class BundleConfig(BaseModel):
    tools: dict[str, dict]
    state_command: str | None = None


class BaseBundle(BaseModel):
    ...


class APIBundle(BaseBundle):
    api: str
    type: Literal["api"] = "api"

    @property
    def position(self) -> str:
        return self.api

    @cached_property
    def commands(self) -> list[Command]:
        manager = BaseToolManager.tool_index.get(self.api, None)
        if not manager:
            raise _NoSuchToolException(self.api)
        fn = getattr(manager, self.api)
        cmd: Command = tool_desc.render_tool_by_decorator(fn)
        cmd.type = self.type
        return [cmd]

    @property
    def state_command(self) -> str | None:
        return None


class ShellBundle(BaseBundle):
    path: Path
    hidden_tools: list[str] = Field(default_factory=list)
    _config: BundleConfig = PrivateAttr(default=None)
    type: Literal["shell"] = "shell"

    @property
    def position(self) -> str:
        return self.path.__str__()

    @model_validator(mode="after")
    def validate_tools(self):
        self.path = _convert_path_to_abspath(self.path)
        if not self.path.exists():
            msg = f"Bundle path '{self.path}' does not exist."
            raise ValueError(msg)

        config_path = self.path / "config.yaml"
        if not config_path.exists():
            msg = f"Bundle config file '{config_path}' does not exist."
            raise ValueError(msg)

        config_data = yaml.safe_load(config_path.read_text())
        self._config = BundleConfig(**config_data)

        invalid_hidden_tools = set(self.hidden_tools) - set(self._config.tools.keys())
        if invalid_hidden_tools:
            msg = f"Hidden tools {invalid_hidden_tools} do not exist in available tools"
            raise ValueError(msg)
        return self

    @property
    def state_command(self) -> str | None:
        return self.config.state_command

    @property
    def config(self) -> BundleConfig:
        return self._config

    @property
    def commands(self) -> list[Command]:
        return [
            Command(
                name=tool,
                **(
                    tool_config.model_dump()
                    if isinstance(tool_config, Command)
                    else tool_config
                ),
                type=self.type,
            )
            for tool, tool_config in self.config.tools.items()
            if tool not in self.hidden_tools
        ]


class AgentBundle(BaseBundle):
    agent_config: Path
    """
    Agent bundle that loads a single agent configuration file.
    """
    type: Literal["agent"] = "agent"

    def __init__(self, /, **kwargs):
        super().__init__(**kwargs)
        from sweagent.agent.config import DefaultAgentConfig
        self._agent_config: DefaultAgentConfig = PrivateAttr(default=None)

    @property
    def position(self) -> str:
        return self.agent_config.__str__()

    @model_validator(mode="after")
    def validate_config(self):
        self.agent_config = _convert_path_to_abspath(self.agent_config)
        if not self.agent_config.exists():
            msg = f"Agent config agent '{self.agent_config}' does not exist."
            raise ValueError(msg)

        if not self.agent_config.is_file():
            msg = f"Agent config agent '{self.agent_config}' must be a file."
            raise ValueError(msg)

        config_data = load_agent_yaml_config(self.agent_config)
        from sweagent.agent.config import DefaultAgentConfig
        self._agent_config = DefaultAgentConfig.model_validate(config_data['agent'])
        return self

    @property
    def config(self):
        return self._agent_config

    @property
    def state_command(self) -> str | None:
        return None

    @property
    def commands(self) -> list[Command]:
        return self._agent_config.commands


Bundle = Union[APIBundle, ShellBundle, AgentBundle]

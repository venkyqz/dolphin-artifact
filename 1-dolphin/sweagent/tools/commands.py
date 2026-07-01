"""
Core module for defining and parsing commands in the SWE Agent system.

This module provides the foundational classes and utilities for defining commands that can be executed by the agent.
It is used extensively by:

- tools.py: For command installation, execution and environment management
- parsing.py: For parsing model outputs into executable commands
- utils.py: For handling multi-line commands and argument quoting

Key Classes:
- Command: Represents an executable command with arguments and documentation
- Argument: Defines an argument that can be passed to a command

The module supports both simple bash commands and complex multi-line commands with typed arguments.
Commands can be defined either in bash scripts with YAML docstrings or as bash functions.
"""

from __future__ import annotations

import json
import re
import string
from functools import cached_property
from io import StringIO
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

from openai.types.shared import FunctionDefinition
from pydantic import BaseModel, Field, field_validator, model_validator

from sweagent.utils.jinja_warnings import _warn_probably_wrong_jinja_syntax

ARGUMENT_NAME_PATTERN = r"[a-zA-Z_][a-zA-Z0-9_-]+"


def _extract_keys(format_string: str) -> set[str]:
    """Given a format string, returns a set of all the keys in the format string.

    Used for validating that command signatures match their argument definitions.

    Args:
        format_string: A Python format string containing named fields

    Returns:
        Set of field names found in the format string
    """
    formatter = string.Formatter()
    keys = set()
    for _, field_name, _, _ in formatter.parse(format_string):
        if field_name is not None:
            keys.add(field_name)
    return keys


class IndentWriter:
    def __init__(self, indent="    "):
        self.buffer = StringIO()
        self.level = 0
        self.indent = indent

    def write_line(self, line=""):
        self.buffer.write(f"{self.indent * self.level}{line}\n")

    def write(self, line=""):
        self.buffer.write(f"{self.indent * self.level}{line}")

    def indent_block(self):
        class Context:
            def __enter__(ctx_inner):
                self.level += 1

            def __exit__(ctx_inner, *args):
                self.level -= 1

        return Context()

    def getvalue(self):
        return self.buffer.getvalue()


class Argument(BaseModel):
    f"""Defines an argument that can be passed to a command.

    Attributes:
        name: The argument name, must match {ARGUMENT_NAME_PATTERN!r}
        type: The argument type (e.g. "string", "integer")
        description: Human readable description of the argument
        required: Whether this argument must be provided
        enum: Optional list of allowed values
        argument_format: Format string for how to render the argument value in the command
    """

    name: str
    type: str
    items: dict[str, str] | None = None
    description: str
    required: bool
    enum: list[str] | None = None
    argument_format: str = "{{value}}"
    """How to invoke the argument in the command. Make sure to use jinja syntax ({{value}}) instead of {value})."""

    default: Optional[Any] = None
    choices: Optional[Dict] = None
    newline: bool = False

    @field_validator("argument_format")
    def validate_argument_format(cls, value: str) -> str:
        _warn_probably_wrong_jinja_syntax(value)
        return value


class Command(BaseModel):
    """Represents an executable command with arguments and documentation.

    A command can be either a simple bash command or a multi-line command terminated by an end marker.

    Attributes:
        name: The command name
        docstring: Human readable description of what the command does
        signature: Optional custom signature override
        end_name: For multi-line commands, the terminating marker
        arguments: List of arguments accepted by the command
        type: the type of the command, can be shell or api, should be set by the ToolConfig.commands

    Properties:
        invoke_format: Format string for constructing the full command invocation
    """

    name: str
    docstring: str | None
    tip: str | None = None
    signature: str | None = None
    # if there is an end_name, then it is a multi-line command
    end_name: str | None = None
    arguments: List[Argument] = Field(default_factory=list)
    type: str = ""

    fn: Optional[Any] = None

    @property
    def is_shell(self):
        return self.type == "shell"

    @property
    def is_api(self):
        return self.type == "api"

    @property
    def is_agent(self):
        return self.type == "agent"

    @cached_property
    def invoke_format(self) -> str:
        """Gets the format string for invoking this command with arguments.

        Returns either the custom signature with argument placeholders replaced,
        or a default format of "command arg1 arg2 ...".
        """
        if self.signature:
            # First validate that all arguments are present in the original signature
            if not all(
                f"<{arg.name}>" in self.signature
                or f"[<{arg.name}>]" in self.signature
                or f"{{{arg.name}}}" in self.signature
                for arg in self.arguments
            ):
                msg = (
                    f"Missing arguments in signature: {self.signature}. Did you format the signature correctly? "
                    "You must include all argument names in the signature with <name>, [<name>], or {name} notation."
                )
                raise ValueError(msg)

            # Then do the replacement
            return re.sub(
                rf"\[?<({ARGUMENT_NAME_PATTERN})>\]?", r"{\1}", self.signature
            )
        else:
            # cmd arg_format_1 arg_format_2 ...
            _invoke_format = f"{self.name} "
            for arg in self.arguments:
                _invoke_format += f"{{{arg.name}}} "
            return _invoke_format

    def get_function_calling_tool(self, provider="openai") -> dict:
        """Converts this command into an OpenAI function calling tool definition.

        Returns:
            Dict containing the OpenAI function schema for this command
        """
        if provider == "anthropic":
            return self.to_anthropic_function()
        else:
            f = self.to_openai_function().model_dump(exclude_none=True)
            tool = {
                "function": f,
                "type": "function",
            }
            return tool

    @model_validator(mode="after")
    def validate_arguments(self) -> Command:
        """Validates command argument configuration.

        Checks:
        - Required arguments come before optional ones
        - Argument names are unique
        - Argument names match the pattern
        - Arguments match the signature

        Returns:
            The validated Command instance

        Raises:
            ValueError: If validation fails
        """
        if not self.arguments:
            return self
        found_optional = False
        for arg in self.arguments:
            if found_optional and arg.required:
                msg = f"Command '{self.name}': Required argument '{arg.name}' cannot come after optional arguments"
                raise ValueError(msg)
            if not arg.required:
                found_optional = True
        duplicates = {
            arg.name for arg in self.arguments if self.arguments.count(arg) > 1
        }
        if duplicates:
            msg = f"Command '{self.name}': Duplicate argument names: {duplicates}"
            raise ValueError(msg)
        for arg in self.arguments:
            if not re.match(ARGUMENT_NAME_PATTERN, arg.name):
                msg = f"Command '{self.name}': Invalid argument name: '{arg.name}'"
                raise ValueError(msg)
        if (invoke_keys := _extract_keys(self.invoke_format)) != {
            arg.name for arg in self.arguments
        }:
            msg = f"Command '{self.name}': Argument names ({invoke_keys}) in signature / invoke_format {self.invoke_format!r} do not match argument names"
            raise ValueError(msg)
        return self

    def signature_text(self, writer=None) -> str:
        if not writer:
            writer = IndentWriter()
        writer.write(self.name)

        # Process all args and format based on required flag
        for arg in self.arguments:
            # Format arg name with <> if required, [] if optional
            arg_format = f"<{arg.name}>" if arg.required else f"[{arg.name}]"

            if arg.newline:
                writer.write_line()
                writer.write_line(arg_format)
            else:
                writer.buffer.write(f" {arg_format}")

        return writer.getvalue()

    def pp_all(self):
        print("CLI Format:")
        print(self.format_as_cli_usage())
        print("\nYAML Format:")
        print(self.format_as_yaml())
        print("\nOpenAI Tool Format:")
        print(self.format_as_openai_tool())
        print("\nMCP Format:")
        print(self.format_as_mcp())

    def format_as_yaml(self) -> str:
        # Use pydantic's json() method to convert to dict, then use yaml library to convert to YAML
        import yaml

        class GeneralArgument(BaseModel):
            name: str
            type: str
            description: str
            required: bool
            default: Optional[Any] = None

        class GeneralTool(BaseModel):
            signature: str
            docstring: str
            arguments: List[GeneralArgument]

        # Create tool parameter list
        arguments = []
        for arg in self.arguments:
            arguments.append(
                GeneralArgument(
                    name=arg.name,
                    type=arg.type.lower() if arg.type else "string",
                    description=arg.description,
                    required=arg.required,
                    default=arg.default,
                )
            )

        # Create tool definition
        tool_def = GeneralTool(
            signature=self.signature_text(),
            docstring=self.docstring.strip() if self.docstring else "",
            arguments=arguments,
        )

        # Create final document
        d = dict(tools={self.name: tool_def.model_dump()})
        return yaml.dump(d, sort_keys=False)

    def format_as_cli_usage(self) -> str:
        writer = IndentWriter()
        writer.write_line(self.name)

        with writer.indent_block():
            # Format argument name based on required flag
            if self.docstring:
                writer.write_line("Description:")
                with writer.indent_block():
                    writer.write_line(self.docstring.strip())

            writer.write_line("Usage:")
            with writer.indent_block():
                self.signature_text(writer=writer)

            writer.write_line()
            if self.arguments:
                writer.write_line("Arguments:")

                def render_choices(arg):
                    if arg.choices:
                        # for choices, we should do double indent block for a better format result.
                        with writer.indent_block():
                            with writer.indent_block():
                                for name, help in arg.choices.items():
                                    writer.write_line(f"{name}: {help}")

                with writer.indent_block():
                    for arg in self.arguments:
                        # Format argument name based on required flag
                        desc = f"<{arg.name}>" if arg.required else f"[{arg.name}]"
                        desc += "  Required." if arg.required else "  Optional."

                        if arg.type:
                            desc += f"  Type: {arg.type}."
                        if arg.default:
                            desc += f"  Default: {arg.default}."

                        desc += f"  {arg.description}" if arg.description else ""
                        writer.write_line(desc)

                        render_choices(arg)

        return writer.getvalue()

    @staticmethod
    def normalize_json_type(t):
        # Base schema with common properties
        # Map common types to JSON Schema types
        mapping = {
            "str": "string",
            "string": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }
        if t.lower() not in mapping:
            raise JSONDecodeError
        return mapping.get(t.lower(), t.lower())

    def to_openai_function(self):
        """
        ref:
        https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools

        example:
        {
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
              "type": "object",
              "properties": {
                "location": {
                  "type": "string",
                  "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                  "type": "string",
                  "enum": ["celsius", "fahrenheit"]
                }
              },
              "required": ["location"]
            }
          }
        }

        """

        # Convert arguments to OpenAI function parameters format
        parameters = {"type": "object", "properties": {}, "required": []}

        for arg in self.arguments:

            schema = {
                "type": self.normalize_json_type(arg.type),
                "description": arg.description,
            }

            if arg.default:
                schema["default"] = arg.default

            if arg.required:
                parameters["required"].append(arg.name)

            if arg.enum:
                schema["enum"] = arg.enum
            elif arg.choices:
                enums = []
                descriptions = []
                for k, v in arg.choices.items():
                    enums.append(k)
                    descriptions.append(f"{k}: {v}")

                schema["enum"] = enums
                schema["description"] = (
                    schema["description"] + "\n\n" + "\n".join(descriptions)
                )

            parameters["properties"][arg.name] = schema

        # Create function definition using OpenAI's model
        function_def = FunctionDefinition(
            name=self.name,
            description=self.docstring.strip() if self.docstring else "",
            parameters=parameters,
        )
        return function_def

    def to_anthropic_function(self) -> dict:
        """
        Returns a tool definition compatible with Anthropic's tool schema.

        Example output:
        {
          "name": "get_stock_price",
          "description": "Get the current stock price for a given ticker symbol.",
          "input_schema": {
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
              }
            },
            "required": ["ticker"]
          }
        }
        """
        input_schema = {"type": "object", "properties": {}, "required": []}
        for arg in self.arguments:
            prop = {
                "type": self.normalize_json_type(arg.type),
                "description": arg.description,
            }
            if arg.enum:
                prop["enum"] = arg.enum
            elif arg.choices:
                enums = []
                descriptions = []
                for k, v in arg.choices.items():
                    enums.append(k)
                    descriptions.append(f"{k}: {v}")
                prop["enum"] = enums
                prop["description"] = (
                    prop["description"] + "\n\n" + "\n".join(descriptions)
                )
            if arg.default is not None:
                prop["default"] = arg.default
            input_schema["properties"][arg.name] = prop
            if arg.required:
                input_schema["required"].append(arg.name)
        return {
            "name": self.name,
            "description": self.docstring.strip() if self.docstring else "",
            "input_schema": input_schema,
        }

    def format_as_openai_tool(self) -> str:
        function_def = self.to_openai_function()
        return json.dumps(function_def.model_dump(exclude_none=True), indent=2)

    def format_as_mcp(self):
        """
        Format the function into the MCP format.
        Returns a Prompt compatible format as JSON string.
        """

        from mcp.server.fastmcp.prompts.base import Prompt, PromptArgument

        fn_placeholder = lambda: None
        prompt = Prompt(
            name=self.name,
            description=self.docstring.strip() if self.docstring else "",
            arguments=[],
            fn=self.fn or fn_placeholder,
        )
        # Add parameters as PromptArgument format
        for arg in self.arguments:
            argument = PromptArgument(
                name=arg.name,
                description=arg.description if arg.description else "",
                required=arg.required,
            )
            prompt.arguments.append(argument)

        return json.dumps(prompt.model_dump(), indent=2)


# Default Bash tool
BASH_COMMAND = Command(
    name="bash",
    # name="execute_bash",
    signature="<command>",
    # signature="echo '<command>'\n<command>\necho \"root@workspace:${{PWD}} #\n[Command finished with exit code ${{?}}]\"",
    docstring="runs the given command directly in bash",
    arguments=[
        Argument(
            name="command",
            type="string",
            description="The bash command to execute.",
            required=True,
        )
    ],
)

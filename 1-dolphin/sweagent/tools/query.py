import inspect
import platform
import shlex
import traceback
from typing import List, Optional
from typing import Type

from pathlib import Path
from pydantic import BaseModel
from pydantic import Field

from sweagent.agent.types import ToolResponse
from sweagent.sop.registry import Memory

from sweagent.codequery.tools import BaseToolManager
from sweagent.environment.swe_env import SWEEnv
from sweagent.tools.tool_desc import Tool
from sweagent.utils.telemetry import logger
from sweagent.exceptions import _NoSuchToolException

# Store the instances of tool classes
tool_executors = {}

def convert_argument_types(func, args: list, kwargs: dict) -> tuple[list, dict]:
    """Convert string arguments to their proper types based on function signature.

    Args:
        func: The function to call
        args: List of positional arguments (potentially as strings)
        kwargs: Dictionary of keyword arguments (potentially as strings)

    Returns:
        Tuple of (converted_args, converted_kwargs)
    """
    try:
        sig = inspect.signature(func)
        param_list = list(sig.parameters.values())
        converted_args = []
        converted_kwargs = {}

        # Convert positional arguments
        for i, value in enumerate(args):
            if i < len(param_list):
                param = param_list[i]
                converted_value = _convert_value(value, param.annotation)
                converted_args.append(converted_value)
            else:
                # More args than parameters, keep as-is
                converted_args.append(value)

        # Convert keyword arguments
        for param_name, value in kwargs.items():
            param = sig.parameters.get(param_name)
            if param:
                converted_kwargs[param_name] = _convert_value(value, param.annotation)
            else:
                converted_kwargs[param_name] = value

        return converted_args, converted_kwargs

    except Exception:
        # If anything goes wrong with type conversion, return original args
        return args, kwargs

def _convert_value(value, param_type):
    """Convert a single value to the specified type."""
    # Skip conversion if already the right type or None
    if value is None:
        return value

    # Handle common type conversions
    if param_type == int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    elif param_type == float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    elif param_type == bool:
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        else:
            return bool(value)
    else:
        # Keep as-is for other types (str, etc.)
        return value

class ToolCallModel(BaseModel):
    """Model for parsed command results"""

    # string format tool call action
    command: str = ""
    # the name of tool to be used
    tool: str = ""
    args: List[str] = Field(default_factory=list)
    kwargs: Optional[dict] = Field(default_factory=dict)

def execute_command(command: str, root_path: str, env: SWEEnv = None, memory: Memory = None):
    """Execute a command string by parsing it and dispatching to appropriate tool.

    Args:
        command (str): Command string to execute
        root_path: The directory to execute the command in
        env: The system execution environment
        memory: Global memory for sharing data

    Returns:
        Output from the dispatched tool

    Raises:
        ValueError: If command parsing or dispatch fails
    """
    try:
        call: ToolCallModel = parse_command(command)

        if not call.tool:
            raise _NoSuchToolException("No command provided")

        return invoke(call, root_path, env, memory)

    except _NoSuchToolException as nst_exception:
        traceback.print_exception(nst_exception)
        raise nst_exception
    except Exception as e:
        traceback.print_exception(e)
        raise ValueError(f"Failed to execute command '{command}': {str(e)}")


def invoke(
    call: ToolCallModel, root_path: str, env: SWEEnv = None, memory: Memory = None
):
    global tool_executors

    tool_name = call.tool
    real_args = list(call.args)

    tool_manager_cls: Type[BaseToolManager] | None = BaseToolManager.tool_index.get(
        tool_name, None
    )
    if not tool_manager_cls:
        raise _NoSuchToolException(tool_name)

    # Lookup the cache to obtain the class instance for the tool named tool_name
    cache_locator = (tool_manager_cls, root_path, env, memory)
    tm_inst = tool_executors.get(cache_locator, None)
    if not tm_inst:
        tm_inst = tool_manager_cls(root_path=root_path, env=env, memory=memory)
        tool_executors[cache_locator] = tm_inst

    if not hasattr(tm_inst, tool_name):
        raise _NoSuchToolException(tool_name)

    # Call the tool and return the result
    tool_fn = getattr(tm_inst, tool_name)

    # Convert argument types based on function signature
    converted_args, converted_kwargs = convert_argument_types(tool_fn, real_args, call.kwargs)
    output = tool_fn(*converted_args, **converted_kwargs)

    completion = Tool.is_completion(tool_fn)
    if Tool.is_submission(tool_fn):
        submission = output
    else:
        submission = None

    return ToolResponse(
        status="success",
        message=f"{output}",
        command=call.command,
        completion=completion,
        submission=submission,
    )


def parse_command(command: str) -> ToolCallModel:
    """Parse a command string into tool, arguments and post-arguments.

    Args:
        command (str): Command string to parse, can be single or multiple lines

    Returns:
        ToolCallModel: Contains:
            - tool: The command/tool name from first argument
            - args: Remaining arguments from first line

    Examples:
        >>> parse_command('cmd arg1 arg2')
        ToolCallModel(command='cmd arg1 arg2', tool='cmd', args=['arg1', 'arg2'], kwargs={})

        >>> parse_command('cmd arg1\\nline2\\nline3')
        ToolCallModel(command='cmd arg1\\nline2\\nline3', tool='cmd', args=['arg1', 'line2', 'line3'], kwargs={})
    """
    # Use platform-specific parsing for Windows paths
    if platform.system() == "Windows":
        # On Windows, use posix=False to avoid treating backslashes as escape characters
        parts: List[str] = shlex.split(command, posix=False)
        # When posix=False, quotes are not automatically removed, so we need to do it manually
        parts = [part.strip('"\'') for part in parts]
    else:
        # On Unix-like systems, use default POSIX parsing
        parts: List[str] = shlex.split(command)

    # Extract tool and remaining args
    tool = parts[0] if parts else ""
    args = parts[1:] if len(parts) > 1 else []

    return ToolCallModel(
        tool=tool,
        args=args,
        command=command,
    )

import inspect
from collections import defaultdict
from functools import wraps
from io import StringIO
from typing import Any, Callable, Dict, List, Optional

from docstring_parser import parse  # Use docstring_parser library to parse docstring
from pydantic import BaseModel

from sweagent.tools.commands import Argument, Command

# Map Python types to JSON schema types
type_string_mapping = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "NoneType": "null",
    # Add common aliases
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
    "null": "null",
}


class Arg:
    def __init__(
        self,
        default=Ellipsis,
        description="",
        newline=False,
        choices: Optional[Dict] = None,
    ):
        self.has_default = default is not Ellipsis
        self.default = default
        self.description = description
        self.newline = newline
        self.choices = choices

    def __repr__(self):
        if self.has_default:
            return repr(self.default)
        return "..."

    def __class_getitem__(cls, item):
        return cls(item)

    def __eq__(self, other):
        # To support runtime comparisons
        if self.has_default:
            return self.default == other
        return False


class Tool:
    """
    Decorator for defining a Tool for LLM interaction.
    Args:
        name (str, optional): Name of the tool. If not provided, the function name will be used.
        help (str, optional): Help text for the tool. If not provided, the docstring will be used.
        tip (str, optional): Additional tip text to be appended to the help message.
    Example:
        @Tool(name="echo", help="Echo a string", tip="Use this for simple text output")
        def echo(x:string = Arg(help="A string value to return")) -> string:
            return x
    Explain:
        We can render an api with tool decorator into a specified format automatically
        with its name, help and parameter description (rendered from Arg and type annotation).

        Furthermore, we can define different format output, aka. render to different format.

    """

    def __init__(self, name=None, help=None, tip=None, is_completion=False, is_submission=False):
        self.name = name
        self.help = help
        self.tip = tip
        self.is_completion = is_completion
        self.is_submission = is_submission

    def __call__(self, func):
        # Store command metadata
        func._command_name = self.name or func.__name__
        func._command_help = self.help
        func._command_tip = self.tip
        func._is_tool = True
        func._is_completion = self.is_completion
        func._is_submission = self.is_submission

        # Preserve original function metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._wrapper(func, *args, **kwargs)

        # Copy command metadata to wrapper
        wrapper._command_name = func._command_name
        wrapper._command_help = func._command_help
        wrapper._command_tip = func._command_tip
        wrapper._is_tool = func._is_tool
        wrapper._is_completion = func._is_completion
        wrapper._is_submission = func._is_submission

        return wrapper

    @staticmethod
    def _wrapper(func, *args, **kwargs):
        sig = inspect.signature(func)
        bound_args = []

        parameters = list(sig.parameters.items())

        if len(parameters) > len(args):
            bound_args.extend(args)
            for idx, (name, param) in enumerate(parameters[len(args) :]):
                # Fall back to default values
                tool_arg = param.default
                if isinstance(tool_arg, Arg):
                    if name in kwargs:
                        value = kwargs.get(name, None)
                        bound_args.append(value)
                    else:
                        if tool_arg.has_default:
                            bound_args.append(tool_arg.default)
                        else:
                            raise TypeError(f"Missing required argument: {name}")
                elif tool_arg is not inspect.Parameter.empty:
                    raise TypeError(f"Missing required argument: {name}")
        else:
            bound_args = args

        if func._command_name in (
            "query_instruction",
            "query_basicblock",
            "query_function",
            "query_binary",
            "update_function",
            "update_binary",
        ):
            from sweagent.codequery.tools.internal.binary_database import BinaryDatabase

            result = func(*bound_args)
            current_location = BinaryDatabase.get_current_location()
            return f"{result}\n\nCurrent location: {current_location}"

        return func(*bound_args)

    @staticmethod
    def is_completion(tool_fn):
        return getattr(tool_fn, "_is_completion", False)

    @staticmethod
    def is_submission(tool_fn):
        return getattr(tool_fn, "_is_submission", False)


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


class GeneralToolSet(BaseModel):
    tools: Dict[str, GeneralTool]


def render_tool_by_comment(func):
    """Force generate usage text for functions without Arg parameters"""
    sig = inspect.signature(func)

    # Get function name and docstring
    func_name = func.__name__

    # Get docstring and parse it
    docstring = inspect.getdoc(func) or ""
    parsed = parse(docstring)

    # Extract main description
    help = parsed.description or ""

    # Extract parameter documentation
    param_docs = {param.arg_name: param.description for param in parsed.params}

    parameters = list(sig.parameters.items())

    # Skip first parameter if it's self
    if list(parameters) and list(parameters)[0][0] == "self":
        parameters = list(parameters)[1:]

    # Parse parameters into ArgumentInfo objects
    required_args = []
    optional_args = []

    for name, param in parameters:
        # Get type annotation from parameter if available
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        type_str = type_string_mapping.get(param_type.__name__)

        # Check if parameter has default value
        has_default = param.default != inspect.Parameter.empty
        default_value = str(param.default) if has_default else None

        arg_doc = param_docs.get(name, "")

        arg_info = Argument(
            name=name,
            description=arg_doc,  # No help text available without Arg annotation
            required=not has_default,
            default=default_value,
            type=type_str,
        )

        if has_default:
            optional_args.append(arg_info)
        else:
            required_args.append(arg_info)

    # Create command usage model
    usage = Command(
        name=func_name,
        docstring=help,
        arguments=required_args,
        fn=func,
    )

    return usage


def render_tool_by_decorator(func) -> Command:
    """Generate usage text for functions with Arg parameters"""
    sig = inspect.signature(func)

    # Get function name, docstring and command metadata
    func_name = getattr(func, "_command_name", func.__name__)
    help = getattr(func, "_command_help", "")

    # Get tip from function metadata and combine with help
    tip = getattr(func, "_command_tip", None)
    if tip:
        help = f"{help}\n\nTip: {tip}" if help else f"Tip: {tip}"

    # Parse parameters into ArgumentInfo objects
    arguments = []

    for name, param in sig.parameters.items():
        if isinstance(param.default, Arg):
            arg = param.default
            # Get type annotation from parameter if available
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str

            # Use arg.typ if no parameter annotation
            type_str = param_type.__name__

            arg_info = Argument(
                name=name,
                description=arg.description,
                choices=arg.choices,
                required=not arg.has_default,
                default=arg.default if arg.has_default else None,
                type=type_str,  # Add type information to ArgumentInfo
                newline=arg.newline,
            )
            arguments.append(arg_info)

    # Create command usage model
    usage = Command(
        name=func_name,
        docstring=help,
        arguments=arguments,
        fn=func,
        tip=tip,
    )

    return usage

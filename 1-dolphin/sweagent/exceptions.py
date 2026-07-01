from typing import Any, Literal

"""This module contains all custom exceptions used by the SWE-agent."""


class FormatError(Exception):
    """Raised when the model response cannot properly be parsed into thought and actions."""


class FunctionCallingFormatError(FormatError):
    """Format error exception used by the function
    calling parser."""

    def __init__(
        self,
        message: str,
        error_code: Literal[
            "missing", "multiple", "incorrect_args", "invalid_json", "invalid_command", "missing_arg", "unexpected_arg"
        ],
        **extra_info: Any,
    ):
        super().__init__(message + f" [error_code={error_code}]")
        self.message = message
        self.extra_info = {"error_code": error_code, **extra_info}

class ThinkingActionFormatError(FormatError):
    """Format error exception used by the thinking action parser."""

    MISSING = "missing"
    MISSING_NAME = "missing_name"
    MISSING_TAG = "missing_tag"
    MULTIPLE = "multiple"
    INCORRECT_ARGS = "incorrect_args"
    INVALID_JSON = "invalid_json"
    INVALID_COMMAND = "invalid_command"
    MISSING_ARG = "missing_arg"
    UNEXPECTED_ARG = "unexpected_arg"
    UNEXPECTED_ERROR = "unexpected_error"

    def __init__(
        self,
        message: str,
        error_code: str,
        **extra_info: Any,
    ):
        super().__init__(message + f" [error_code={error_code}]")
        self.message = message
        self.error_code = error_code
        self.extra_info = {"error_code": error_code, **extra_info}

class ContextWindowExceededError(Exception):
    """Raised when the context window of a LM is exceeded"""


class CostLimitExceededError(Exception):
    """Raised when we exceed a cost limit"""


class InstanceCostLimitExceededError(CostLimitExceededError):
    """Raised when we exceed the cost limit set for one task instance"""


class TotalCostLimitExceededError(CostLimitExceededError):
    """Raised when we exceed the total cost limit"""


class InstanceCallLimitExceededError(CostLimitExceededError):
    """Raised when we exceed the per instance call limit"""


class ContentPolicyViolationError(Exception):
    """Raised when the model response violates a content policy"""


class ModelConfigurationError(Exception):
    """Raised when the model configuration is invalid/no further retries
    should be made.
    """

class _UserInterrupt(Exception):
    """user interrupt and give up"""


class _NoActionException(Exception):
    def __init__(self):
        super().__init__("No command call given in the response")

class _NoSuchToolException(Exception):
    def __init__(self, message=""):
        super().__init__(f"No such tool found: {message}")

class _AgentInterrupt(Exception):
    """agent interrupt and give up"""

class DuplicateToolNameException(Exception):
    def __init__(self, name=""):
        super().__init__(f"Register tool with duplicate name: {name}")
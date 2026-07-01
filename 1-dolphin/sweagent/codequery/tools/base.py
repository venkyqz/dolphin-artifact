import os
from pathlib import Path
from typing import Dict, Any

from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import DuplicateToolNameException
from sweagent.sop.registry import Memory


class BaseToolManager:
    """
    Base class that handles tool registration for classes that define tools.
    """
    tool_index: Dict[str, Any] = {}

    # Global constant for pagination limit
    PAGINATION_LIMIT = 100

    def __init__(self, root_path, env: SWEEnv = None, memory: Memory = None):
        self.root_path = os.path.abspath(root_path)
        self.env = env
        self.memory = memory

    def __init_subclass__(cls, **kwargs):
        """
        Automatically register all methods as tools when subclassing BaseToolManager.
        Only register tools defined directly in this class, not inherited ones.
        """
        super().__init_subclass__(**kwargs)

        # Only register methods defined in this specific class
        for attr_name, fn in cls.__dict__.items():
            if not attr_name.startswith("_") and callable(fn):
                # Skip inherited methods by checking if the method is defined in the current class
                if fn.__qualname__.split('.')[0] != cls.__name__:
                    continue

                if cls.is_tool(fn):
                    if attr_name in cls.tool_index:
                        raise DuplicateToolNameException(attr_name)
                    cls.tool_index[attr_name] = cls

    @classmethod
    def is_tool(cls, func):
        """
        A helper method to check if current function is wrapped by tool decorator.
        """
        if hasattr(func, "_is_tool"):
            return True
        return False
from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class OminiRegistryKey:
    # general
    TOOL_AGENT_ARGS = "kwargs"
    PLAN_SUMMARY = "plan_summary"

    # problem
    PROBLEM_ID = "problem_id"
    WORKING_DIR = "working_dir"

    # alice
    EDIT_SUMMARY = "edit_summary"
    DIFF = "diff"

class Memory:
    """
    An external memory for persisting agent messages
    Work like a Registry for sharing key-value data between different parts of a workflow.
    """

    _default_instance = None
    _default_instance_lock = Lock()

    def __init__(self):
        """A key-value storage for memory content"""
        self.kv: dict[str, Any] = dict()
        self._lock = Lock()

    @staticmethod
    def get_share_instance():
        """Get the singleton instance of WorkflowRegistry.
        This instance is shared in all agents.

        Returns:
            Memory: The singleton instance
        """
        if Memory._default_instance is None:
            with Memory._default_instance_lock:
                if Memory._default_instance is None:
                    Memory._default_instance = Memory()
        return Memory._default_instance

    def set(self, key: str, value: Any) -> None:
        """Set a value for a key in the registry.

        Args:
            key: The key to store the value under
            value: The value to store
        """
        with self._lock:
            self.kv[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the registry by key.

        Args:
            key: The key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The stored value or default if not found
        """
        with self._lock:
            return self.kv.get(key, default)

    def delete(self, key: str) -> None:
        """Delete a key-value pair from the registry.

        Args:
            key: The key to delete
        """
        with self._lock:
            self.kv.pop(key, None)

    def _clear(self) -> None:
        """Clear all key-value pairs from the registry."""
        with self._lock:
            self.kv.clear()

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the registry.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        with self._lock:
            return key in self.kv

    def get_all(self) -> Dict[str, Any]:
        """Get all key-value pairs in the registry.

        Returns:
            Dictionary containing all registry key-value pairs
        """
        with self._lock:
            return self.kv.copy()

    def dump(self):
        """Dump the registry contents into a serializable dictionary format.

        Converts registry values into JSON-serializable formats:
        - Lists and sets are converted to lists
        - Pydantic models are converted to dictionaries
        - Other values are kept as-is

        Returns:
            dict: A dictionary containing all registry data in a serializable format
        """
        with self._lock:
            result = {}
            for k, v in self.kv.items():
                if isinstance(v, (list, set)):
                    result[k] = list(v)
                elif isinstance(v, BaseModel):
                    result[k] = v.model_dump()
                else:
                    result[k] = v
        return result

    def update(self, data: Dict[str, Any]) -> None:
        """Load registry data from a dictionary.

        Args:
            data: Dictionary containing key-value pairs to load into registry
        """
        with self._lock:
            self.kv.clear()
            self.kv.update(data)

    @staticmethod
    def load(data: Dict[str, Any]) -> Memory:
        """Load registry data from a dictionary and return a new Memory instance.

        Args:
            data: Dictionary containing key-value pairs to load into registry

        Returns:
            Memory: A new Memory instance with the loaded data
        """
        memory = Memory()
        memory.update(data)
        return memory

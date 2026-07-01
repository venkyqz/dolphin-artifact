from typing import Any, Callable, Dict, List

from sweagent.agent.models import AbstractModel


def filter_system_role(item: Dict[str, Any]) -> bool:
    """Filter out system role messages."""
    return item.get("role") != "system"


def filter_assistant_role(item: Dict[str, Any]) -> bool:
    """Keep only assistant role messages."""
    return item.get("role") == "assistant"


class Summarizer:
    default_prompt = """
Please summarize the conversation in bullet points.
"""

    NO_SYSTEM = filter_system_role
    ONLY_ASSISTANT = filter_assistant_role

    def __init__(self, history: List[Dict[str, Any]], model: AbstractModel):
        """
        Initialize the Summary class with a history and model.

        Each item in history is a dict like
        {
            "role": "user",
            "content": step.observation,
            "agent": self.name,
            "message_type": "tool",
        }

        Args:
            history: The history content to be summarized
            model: The model instance used for querying summaries
        """
        self.history: List[Dict[str, Any]] = history
        self.model: AbstractModel = model

    def summarize(
        self, prompt=None, filters: List[Callable[[Dict[str, Any]], bool]] = None
    ):
        """
        Summarize the history content based on given parameters.

        Args:
            prompt (str, optional): Prompt to be appended at the end of history
            filters (List[Callable], optional): List of filter functions to apply on history items

        Returns:
            str: The summarized content
        """
        if not self.history:
            return ""

        # Apply all filters if provided
        history_list = self.history
        if filters:
            for filter_func in filters:
                history_list = [item for item in history_list if filter_func(item)]

        if not history_list:
            return ""

        prompt = prompt or self.default_prompt
        prompt_items = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # Append prompt items at the end
        history_list.extend(prompt_items)

        # Query model for summary
        query_result = self.model.query(history_list)
        summary_content = query_result["message"]

        return summary_content

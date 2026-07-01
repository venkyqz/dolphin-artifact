from sweagent.codequery.tools import BaseToolManager
from sweagent.config import env
from sweagent.tools.tool_desc import Arg, Tool


class ContextTools(BaseToolManager):
    """
    Deal with the context set up for agents execution.
    The context includes the problem statements, the execution environment, the config settings, and the execution states.
    """

    @Tool(
        help="Retrieve the current issue or problem statement that needs to be addressed",
        tip="Call at the beginning of your workflow to understand the task, review requirements, and guide your approach.",
    )
    def fetch_issue(self):
        """
        Fetches the current problem statement or issue description from the environment settings.
        This provides context about what task or problem the agent should be working on.
        """
        return env.settings.problem

    @Tool(
        help="Signal that all assigned tasks have been completed, whether successful or not",
        is_completion=True,
        tip="Use when all assigned tasks are finished, regardless of success or failure. Ensure you've attempted all required work before signaling completion.",
    )
    def job_done(self):
        """
        Indicates that all jobs and tasks have been completed.
        Use this when the entire workflow or project is finished, regardless of outcome.
        """
        return "All tasks have been completed."

    # @Tool(help="Notify the control system that the current task has been completed")
    # def completed(self):
    #     """
    #     Used by agents to notify the control program that the current task is completed.
    #     This signals successful completion of the assigned work.
    #     """
    #     return ""

    # @Tool(
    #     help="Request to halt task execution when unable to proceed or encountering blocking issues",
    #     is_completion=True,
    #     tip="Use when encountering blocking issues, missing dependencies, or unclear requirements. Provide clear explanation in the 'thought' parameter and only use after exhausting alternatives."
    # )
    # def halt(self, thought: str = Arg(description="The thought to explain why to halt")):
    #     """
    #     Signals that task execution should be interrupted due to inability to proceed.
    #     Use this when encountering blocking issues or when manual intervention is needed.
    #     """
    #     return "System halted and mission terminated.\n\n" + thought

    # @Tool(
    #     help="Skip the current task and proceed to the next one in the workflow",
    #     is_completion=True,
    #     tip="Use when the current task cannot be completed but workflow should continue. Consider if the task is essential and document your reasoning."
    # )
    # def skip(self):
    #     """
    #     Skips the current task and moves to the next one.
    #     Use this when a task cannot be completed but the workflow should continue,
    #     or when the agent can only provide analysis without taking action.
    #     """
    #     return "Skip the current task and proceed to the next one."

    @Tool(
        help="Accept and approve the current code patch or changes",
        is_completion=True,
        tip="Use when code changes meet requirements, quality standards, and solve the problem. Thoroughly review before accepting and check for potential side effects.",
    )
    def accept_code_patch(self):
        """
        Accepts the current code patch or set of changes as satisfactory.
        Use this to approve code modifications, bug fixes, or feature implementations
        that meet the requirements and quality standards.
        """
        return "Accept the current code patch."

    @Tool(
        help="Request further modifications to the code before proceeding",
        is_completion=True,
        tip="Use when code has issues or doesn't meet requirements. Provide specific, constructive feedback about needed changes and explain why they're necessary.",
    )
    def request_code_revision(self):
        """
        Signals that the current code implementation needs further modifications.
        Used by reviewers to indicate that additional changes are required before the task can be considered completed.
        """
        return "Request revision for the submitted code."

    @Tool(
        help="Enable sequential thinking to break down complex problems and analyze issues step-by-step",
        tip="Use to explicitly verbalize your thinking process for complex problems and decisions. Helps verify reasoning, consider alternatives, and document analysis steps before taking action.",
    )
    def thinking(
        self,
        thought: str = Arg(
            description="Current thought or analysis step in the sequential thinking process."
            "Use this to document each step of problem analysis, solution consideration, "
            "or decision making.",
        ),
    ):
        """
        A tool for structured thinking and problem analysis.
        Helps break down complex problems into manageable steps,
        consider multiple solutions, analyze root causes,
        and document the thought process.
        Can be used multiple times to enhance solution quality.

        Args:
            thought (str): The current thought or analysis step

        Returns:
            str: The documented thought/analysis
        """
        return thought

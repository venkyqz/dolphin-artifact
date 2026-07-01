import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from unidiff import PatchSet, UnidiffParseError

from sweagent.codequery.tools import BaseToolManager
from sweagent.codequery.tools.internal import search_replace as aider_edit
from sweagent.codequery.tools.internal.flake8_utils import format_flake8_output
from sweagent.environment.swe_env import SWEEnv
from sweagent.sop.registry import Memory
from sweagent.tools.tool_desc import Arg, Tool
from sweagent.utils.telemetry import logger


class FenceDiff(BaseModel):
    """Data class to store fence diff format."""

    filepath: str
    search: str
    replace: str
    symbol: str = ""

    def to_string(self) -> str:
        """Format as a fenced diff-style string."""
        return (
            f"{self.filepath} {self.symbol}\n"
            "```\n"
            "<<<<<<< SEARCH\n"
            f"{self.search}\n"
            "=======\n"
            f"{self.replace}\n"
            ">>>>>>> REPLACE\n"
            "```\n"
        )

    @staticmethod
    def from_git_diff(git_diff, filepath) -> List["FenceDiff"]:
        """
        Use unidiff library to parse git diff output.
        Handles all edge cases properly.
        """
        try:
            # Parse the git diff output
            patch_set = PatchSet(git_diff)
        except UnidiffParseError as e:
            logger.exception(f"{e}")
            # Log the error and return empty list when parsing fails
            logger.warning(f"Failed to parse git diff for {filepath}: {e}")
            raise e

        fence_diffs = []

        for patched_file in patch_set:
            if patched_file.path.strip() == filepath:
                for hunk in patched_file:
                    # Extract old and new content for this hunk
                    old_lines = []
                    new_lines = []

                    for line in hunk:
                        if line.is_removed:
                            old_lines.append(line.value)
                        elif line.is_added:
                            new_lines.append(line.value)
                        elif line.is_context:
                            old_lines.append(line.value)
                            new_lines.append(line.value)

                    old_content = "".join(old_lines).rstrip()
                    new_content = "".join(new_lines).rstrip()

                    fence_diff = FenceDiff(filepath=filepath, search=old_content, replace=new_content)
                    fence_diffs.append(fence_diff)

        return fence_diffs


class EditRecord(BaseModel):
    """Data class to store edit operation records."""

    filepath: str
    search: str
    replace: str
    symbol: str = ""
    timestamp: str = ""
    success: bool = False
    error_msg: str = ""
    matches_found: int = 0
    operation_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()

    def to_fence_diff(self) -> str:
        """Format the edit record as a fenced diff-style log."""
        fence_diff = FenceDiff(
            filepath=self.filepath,
            search=self.search,
            replace=self.replace,
            symbol=self.symbol,
        )
        return fence_diff.to_string()


class EditTools(BaseToolManager):
    """Tools for editing files with search and replace functionality."""

    GIT_DIFF_CONTEXT_LINES = 5

    def __init__(self, root_path, env: SWEEnv = None, memory: Memory = None):
        super().__init__(root_path, env, memory)
        # Log all edit operations using new dataclass
        self.edit_log: List[EditRecord] = []
        # Stash list to store reverted changes
        self.stash_list: List[List[EditRecord]] = []

    def _log_edit_operation(
        self,
        filepath: str,
        search: str,
        replace: str,
        success: bool,
        error_msg: str = "",
        matches_found: int = 0,
        symbol: str = "",
    ):
        """Log edit operation for debugging and tracking."""
        edit_record = EditRecord(
            timestamp=datetime.now().isoformat(),
            filepath=filepath,
            symbol=symbol,
            search=search,
            replace=replace,
            success=success,
            error_msg=error_msg,
            matches_found=matches_found,
        )
        self.edit_log.append(edit_record)
        logger.info(f"Edit operation logged: {edit_record.to_dict()}")

        if success:
            logger.warning(f"Edit operation on:\n{edit_record.to_fence_diff()}")

        # FIXME: we may add lock for memory set and get
        # Initialize CHANGE_FILE_LIST in memory if not exists
        if self.memory:
            change_file_list = self.memory.get("CHANGE_FILE_LIST", default=set())
            if not isinstance(change_file_list, set):
                change_file_list = set()

            # Record filepath to CHANGE_FILE_LIST if operation was successful
            if success:
                change_file_list.add(filepath)
                self.memory.set("CHANGE_FILE_LIST", change_file_list)

        return edit_record

    @Tool(help="Clear local edit history.")
    def clear_edit_log(self):
        """Clear local edit log, used to record edit operations for individual agents."""
        self.edit_log = []
        return ""

    def _get_abs_file_path(self, filepath: str) -> str:
        """Get the full absolute path of a file from a relative path."""
        if self.env:
            if Path(filepath).is_absolute():
                abspath = filepath
            else:
                abspath = Path(self.env.get_cwd()) / filepath
                abspath = abspath.absolute()
            return abspath
        else:
            # Fallback to direct file reading
            full_path = Path(self.root_path) / filepath if not os.path.isabs(filepath) else Path(filepath)
            return str(full_path)

    def _read_file(self, filepath: str) -> str:
        """Read file content using SWEEnv if available, otherwise read directly."""
        abspath = self._get_abs_file_path(filepath)
        if self.env:
            return self.env.read_file(abspath)
        else:
            # Fallback to direct file reading
            with open(abspath, "r", encoding="utf-8") as f:
                return f.read()

    def _write_file(self, filepath: str, content: str) -> Tuple[str, str]:
        """Write file content using SWEEnv if available, otherwise write directly.
        return [message, written path] tuple
        """
        abspath = self._get_abs_file_path(filepath)
        if self.env:
            self.env.write_file(abspath, content)
            return f"File written via SWEEnv: {filepath} ({abspath})", abspath
        else:
            # Fallback to direct file writing
            full_path = Path(abspath)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File written directly: {full_path}", str(full_path)

    def _flake8(self, filepath: str) -> str:
        """Run flake8 on a given file and return the output as a string"""
        if Path(filepath).suffix != ".py":
            return ""

        # F821	undefined name name
        # F822	undefined name name in __all__
        # F831	duplicate argument name in function definition
        # E111  Indentation is not a multiple of four
        # E112  Expected an indented block
        # E113  Unexpected indentation
        # E999  SyntaxError
        # Since the edit can be performed in multiple steps, we only check single step error
        cmd = f"flake8 --show-source --isolated --select=F831,E112,E113,E999 {filepath}"

        # don't use capture_output because it's not compatible with python3.6
        if self.env:
            return self.env.communicate(cmd, check="ignore")
        else:
            out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return out.stdout.decode()

    def _write_file_lint(
        self, filepath: str, content: str, window_original: str = "", window_applied: str = "", new_file: bool = False
    ) -> Tuple[str, str]:
        """Write file content using SWEEnv if available, otherwise write directly.
        return [message, written path] tuple, and run pylint on a given file and return the output as a string"""

        # abspath = self._get_abs_file_path(filepath)

        # Get pre-edit linting errors
        old_content = ""
        if not new_file:
            old_content = self._read_file(filepath)
        # pre_edit_lint = self._flake8(abspath)

        # Perform the edit
        write_message, write_path = self._write_file(filepath, content)

        # Check for new linting errors
        post_edit_lint = self._flake8(write_path)

        if post_edit_lint:
            # undo edit
            self._write_file(filepath, old_content)

            # Show error and revert changes
            write_message = f"""
    Your proposed edit has introduced new syntax error(s). Can you read this error message carefully and then fix the errors:

    ERRORS:
    {post_edit_lint}

    This is how your edit would have looked if applied
    ------------------------------------------------
    {window_applied}
    ------------------------------------------------

    This is the original code before your edit
    ------------------------------------------------
    {window_original}
    ------------------------------------------------

    Your changes have not been applied. Please fix your edit command and try again.
    You are highly recommended to re-examine the original code to ensure the edit correctness.
    You MUST change the search string for replacing because running the same command will lead to the same error.
    """

            raise Exception(write_message)

        return write_message, write_path

    def _check_search_uniqueness(self, content: str, search: str) -> Optional[str]:
        """Check if search string is unique and return error message if not unique."""
        matches = list(re.finditer(re.escape(search), content))
        if len(matches) != 1:
            return f"Search text found {len(matches)} times. Please make search more specific."
        return None

    def _validate_edit_params(self, filepath: str, search: str, replace: str) -> Optional[str]:
        """Validate edit parameters and return error message if invalid."""
        if not filepath or not filepath.strip():
            return "Filepath cannot be empty"

        if not search:
            return "Search string cannot be empty"

        return None

    @Tool(help="Create a code file specifically for reproducing bugs or testing scenarios")
    def create_reproduction_script(
        self,
        # language: str = Arg(description="Programming language (e.g., 'python', 'javascript', 'java')"),
        filepath: str = Arg(
            description="Relative path where to save the file (e.g., 'test_script.py', 'test/reproduce_bug.js')"
        ),
        code: str = Arg(
            description="Complete source code with descriptive comments explaining the reproduction scenario",
            newline=True,
        ),
    ):
        """
        Creates a code file specifically designed for reproducing bugs or testing scenarios.

        This method is optimized for creating reproduction scripts that help demonstrate
        issues, test fixes, or validate behavior. It automatically handles directory
        creation and delegates to the write_file method.

        Args:
            # language: Programming language identifier for context
            filepath: Relative path where the file should be saved
            code: Complete source code with explanatory comments

        Returns:
            str: Confirmation message with the created file path
        """

        self.create_file(filepath, code)
        return f"Create reproduce file {filepath}:\n```\n{code}\n```\n"

    @Tool(help="Create or overwrite a file with specified content, automatically creating parent directories")
    def create_file(
        self,
        filepath: str = Arg(description="Relative path for the new file (e.g., 'config.json', 'docs/readme.md')"),
        content: str = Arg(description="Complete file content to write", newline=True),
    ):
        """
        Creates or overwrites a file with the specified content.

        This method provides a convenient way to create files with automatic directory
        creation. If the parent directories don't exist, they will be created automatically.
        The file path is relative to the root_path of the tool manager.

        Args:
            filepath: Relative path for the new file
            content: Complete content to write to the file

        Returns:
            str: Confirmation message with the full path of the created file
        """

        try:
            self._write_file_lint(filepath, content, new_file=True)
        except Exception as e:
            # Handle linting errors and other exceptions from _write_file_lint
            error_msg = f"File creation failed:\n{str(e)}"
            logger.error(error_msg)
            return error_msg

        return "File created successfully: " + str(filepath) + "\n"

    @Tool(help="Create or overwrite a file only in current working directory with specified content")
    def create_file_cwd(
        self,
        filename: str = Arg(
            description="Filename for the new file in current working directory (e.g., 'config.json', 'readme.md')"
        ),
        content: str = Arg(description="Complete file content to write", newline=True),
    ):
        """
        Creates or overwrites a file in the current working directory with the specified content.

        Args:
            filename: Filename for the new file in current working directory
            content: Complete content to write to the file

        Returns:
            str: Confirmation message with the full path of the created file
        """

        try:
            cwd = self.env.get_cwd() if self.env else os.getcwd()
            filepath = os.path.join(cwd, filename)
            self._write_file_lint(filepath, content, new_file=True)
        except Exception as e:
            # Handle linting errors and other exceptions from _write_file_lint
            error_msg = f"File creation failed:\n{str(e)}"
            logger.error(error_msg)
            return error_msg

        return "File created successfully: " + str(filepath) + "\n"

    @Tool(
        help="Edit a file by replacing old text with new text.",
        tip="""
    Best practice:
        1. The `search` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
        2. If the `search` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `search` to make it unique.
        3. The `replace` parameter should contain the edited lines that should replace the `search`
        4. It is suggested to include the entire line, including all the leading whitespaces in the `search` parameter.
        """,
    )
    def edit(
        self,
        filepath: str = Arg(description="Path to the file to edit (relative to project root or absolute path)"),
        symbol: str = Arg(
            description="Code symbol or identifier to specify the exact location (e.g. function/class full name)",
        ),
        search: str = Arg(
            description="Text to search for in the file. Must match EXACTLY one or more consecutive lines, including whitespace. Include enough context to make it unique."
        ),
        replace: str = Arg(
            description="Text to replace the matched lines with. If empty, the matched lines will be removed.",
        ),
    ) -> str:
        """
        Edit a file by searching for specific text and replacing it.

        Basic workflow:
        1. Read file content using SWEEnv or direct file access
        2. Search for the specified text
        3. Replace with new text
        4. Write the modified content back to file

        Args:
            filepath (str): Path to the file to edit
            search (str): Text to search for
            replace (str): Text to replace with
            symbol (str): symbol location

        Returns:
            str: Result message indicating success or failure
        """

        try:
            # Validate parameters
            if error := self._validate_edit_params(filepath, search, replace):
                self._log_edit_operation(filepath, search, replace, False, error, symbol=symbol)
                return f"Error: {error}"

            try:
                content = self._read_file(filepath)
            except FileNotFoundError:
                return f"Error: File not found: {filepath}"
            except Exception as e:
                return f"Error accessing file {filepath}: {str(e)}"

            # Check if search text exists
            if search in content:
                # --- Show before/after code window ---
                # Find the start and end index of the search text
                search_start = content.find(search)
                search_end = search_start + len(search)
                # Split content into lines
                lines = content.splitlines(keepends=True)
                # Find the line numbers for the search region
                char_count = 0
                start_line = end_line = None
                for i, line in enumerate(lines):
                    if start_line is None and char_count + len(line) > search_start:
                        start_line = i
                    if end_line is None and char_count + len(line) >= search_end:
                        end_line = i
                        break
                    char_count += len(line)
                # Get before/after windows (10 lines before and after)
                before_start = max(0, start_line - 10)
                after_end = min(len(lines), end_line + 11)
                before_window = "".join(lines[before_start:start_line])
                search_window = "".join(lines[start_line : end_line + 1])
                after_window = "".join(lines[end_line + 1 : after_end])
                before_code_window = before_window + search_window + after_window
                # Prepare after code window
                new_lines = lines.copy()
                new_lines[start_line : end_line + 1] = [replace]
                after_code_window = "".join(new_lines[before_start:after_end])

                # Check search uniqueness
                if error := self._check_search_uniqueness(content, search):
                    self._log_edit_operation(filepath, search, replace, False, error, symbol=symbol)
                    return f"Error: {error}"

                # Perform replacement
                new_content = content.replace(search, replace)
            else:
                # try aider
                after_code_window = ""
                before_code_window = ""
                strategies = [
                    (aider_edit.search_and_replace, aider_edit.all_preprocs),
                ]
                new_content = aider_edit.flexible_search_and_replace([search, replace, content], strategies)

            if not new_content:
                error_msg = f"Search text not found in file: '{search}'"
                self._log_edit_operation(filepath, search, replace, False, error_msg, symbol=symbol)
                return f"Error: {error_msg}"

            # Write modified content back to file
            try:
                # message, write_path = self._write_file(filepath, new_content)
                message, write_path = self._write_file_lint(
                    filepath, new_content, window_original=before_code_window, window_applied=after_code_window
                )
                edit_record = self._log_edit_operation(filepath, search, replace, True, "", 1, symbol=symbol)
                return f"Success: File edited successfully. \n\n{edit_record.to_fence_diff()}\n{message}"
            except Exception as e:
                error_msg = f"Failed to write file: {str(e)}"
                self._log_edit_operation(filepath, search, replace, False, error_msg, symbol=symbol)
                return f"Error: {error_msg}"

        except Exception as e:
            error_msg = f"Unexpected error during edit operation: {str(e)}"
            self._log_edit_operation(filepath, search, replace, False, error_msg, symbol=symbol)
            logger.error(error_msg)
            return f"Error: {error_msg}"

    def git_reset(self, filepath: str) -> bool:
        if self.env:
            try:
                cmd = f"git reset --hard"
                self.env.communicate(cmd, check="ignore")

                if self.env.repo and self.env.repo.base_commit:
                    cmd = f"git checkout {self.env.repo.base_commit} {filepath} -f"
                else:
                    cmd = f"git checkout HEAD {filepath} -f"
                self.env.communicate(cmd, check="ignore")
                return True
            except:
                ...
            return False

        return False

    def git_stash(self) -> bool:
        if self.env:
            try:
                cmd = f"git stash"
                self.env.communicate(cmd, check="ignore")
                return True
            except:
                ...
            return False

        return False

    @Tool(help="Stash all current changes by reverting them and saving the edit records to stash list")
    def stash_changes(
        self,
        message: str = Arg(
            description="Optional message to describe the stashed changes",
            default="Stashed changes",
        ),
    ) -> str:
        """
        Stash all current changes by reverting them and saving edit records.

        This method:
        1. Reverts all successful edits in reverse order
        2. Saves the current edit log to stash list
        3. Clears the current edit log
        4. Updates the CHANGE_FILE_LIST in memory

        Args:
            message: Optional description for the stashed changes

        Returns:
            str: Result message indicating success or failure
        """
        if not self.edit_log:
            return "No changes to stash."

        successful_edits = [record for record in self.edit_log if record.success]
        if not successful_edits:
            return "No successful changes to stash."

        try:
            self.git_stash()

            # Add current edit log to stash with message
            stash_entry = self.edit_log.copy()
            # Add metadata to first record
            if stash_entry:
                stash_entry[0].error_msg = f"STASH_MESSAGE: {message}"
            self.stash_list.append(stash_entry)

            # Clear current edit log
            self.edit_log.clear()

            # Clear CHANGE_FILE_LIST in memory
            if self.memory:
                self.memory.set("CHANGE_FILE_LIST", set())

            return f"Successfully stashed {len(successful_edits)} changes. Stash message: {message}"

        except Exception as e:
            error_msg = f"Failed to stash changes: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    @Tool(help="Generate formatted diff patch text from edit records")
    def generate_patch(
        self,
        include_failed: bool = Arg(
            description="Whether to include failed edit operations in the diff",
            default=False,
        ),
        # stash_index: int = Arg(
        #     description="Index of stash to generate diff for (-1 for current changes)",
        #     default=-1,
        # ),
        context_lines: int = Arg(
            description="Number of context lines to include in git diff (default: 3)",
            default=GIT_DIFF_CONTEXT_LINES,
        ),
    ) -> str:
        """
        Generate formatted diff patch text using git diff command.

        Args:
            include_failed: Whether to include failed operations (ignored when using git diff)
            stash_index: Index of stash to use (-1 for current changes)
            context_lines: Number of context lines to include in git diff

        Returns:
            str: Formatted diff patch text from git diff
        """
        # For current changes, use git diff
        if self.env:
            try:
                # Get list of modified files from edit log
                modified_files = set()
                for record in self.edit_log:
                    if record.success:
                        modified_files.add(record.filepath)

                if not modified_files:
                    return "No successfully modified files found."

                # Execute git diff with context lines for all modified files
                diff_results = []
                for filepath in sorted(modified_files):
                    # Use git diff with specified context lines
                    git_diff_cmd = f"GIT_PAGER=cat git diff HEAD --no-color -U{context_lines} -- {filepath}"
                    result = self.env.execute_command(git_diff_cmd, cwd=self.env.get_cwd())

                    if result.stdout:
                        # Convert git diff to fence_diff format
                        fence_diffs = FenceDiff.from_git_diff(result.stdout, filepath)
                        diff_results.extend(fence_diffs)

                if not diff_results:
                    return "No differences found in modified files."

                return "\n\n".join([i.to_string() for i in diff_results])

            except Exception as e:
                logger.error(f"Failed to execute git diff: {str(e)}")
                raise
        else:
            return self._gen_debug_diff(include_failed)

    def _gen_debug_diff(self, include_failed: bool) -> str:
        """
        Fallback to original diff generation using edit_log.

        Args:
            include_failed: Whether to include failed operations

        Returns:
            str: Formatted diff using edit records
        """
        if not self.edit_log:
            return "No edit records found in current changes."

        # Filter records based on include_failed flag
        records_to_show = self.edit_log if include_failed else [r for r in self.edit_log if r.success]

        if not records_to_show:
            return f"No {'records' if include_failed else 'successful records'} found in current changes."

        # Generate diff text using to_fence_diff
        diff_lines = []
        for record in records_to_show:
            diff_lines.append(record.to_fence_diff())
            if not record.success and record.error_msg:
                diff_lines.append(f"ERROR: {record.error_msg}")
            diff_lines.append("")

        return "\n".join(diff_lines)

    @Tool(
        help="Submit changes by generating git diff for all modified files from edit records",
        is_completion=False,
        is_submission=True,
    )
    def submit(
        self,
    ) -> str:
        """
        Submit changes by executing git diff commands for modified files.

        This method:
        1. Collects all successfully modified files from edit records
        2. Executes git diff command for each file via SWEEnv
        3. Returns the combined git diff output

        Args:

        Returns:
            str: Git diff output for all modified files
        """
        try:
            # Collect all successfully modified files
            modified_files = set()
            for record in self.edit_log:
                if record.success:
                    modified_files.add(record.filepath)

            if not modified_files:
                return "No successfully modified files to submit."

            # Generate git diff for each file
            diff_results = []
            if self.env:
                # Use SWEEnv to execute git commands
                for filepath in sorted(modified_files):
                    try:
                        # Add file to git staging
                        self.env.execute_command(f"git add {filepath}", cwd=self.env.get_cwd())
                        # Get diff of staged changes
                        git_diff_cmd = f"GIT_PAGER=cat git diff HEAD --no-color -U{self.GIT_DIFF_CONTEXT_LINES} --cached -- {filepath}"
                        result = self.env.execute_command(git_diff_cmd, cwd=self.env.get_cwd())
                        if result.stdout:
                            diff_results.append(result.stdout)

                    except Exception as e:
                        raise

                return "".join(diff_results)
            else:
                # Fallback: provide file list and manual instructions
                diff_results.append("SWEEnv not available. Modified files:")
                for filepath in sorted(modified_files):
                    diff_results.append(f"  - {filepath}")
                diff_results.append("")
                diff_results.append("To get git diff, run:")
                for filepath in sorted(modified_files):
                    diff_results.append(f"  git diff {filepath}")

                return "\n".join(diff_results)

        except Exception as e:
            error_msg = f"Failed to submit changes: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

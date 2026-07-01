import json
import shlex
from pathlib import Path

import requests

from sweagent.codequery.tools import BaseToolManager
from sweagent.tools.tool_desc import Arg, Tool
from swerex.deployment.docker import DockerMountDeployment


class FeatureCommonTools(BaseToolManager):
    """
    Standard file system interaction tools.
    Allows the agent to explore directories and manage files without raw bash access.
    """

    def __init__(self, root_path, env=None, memory=None):
        super().__init__(root_path, env, memory)

        if self.env and not isinstance(self.env.deployment, DockerMountDeployment):
            # We strictly enforce Docker deployment to ensure path mapping works expectedly
            message = "FeatureCommonTools requires DockerMountDeployment."
            raise ValueError(message)

    def _get_abs_file_path(self, filepath: Path) -> Path:
        """Resolve path relative to the container's working directory."""
        if filepath.is_absolute():
            return filepath
        if self.env:
            return Path(self.env.get_cwd()) / filepath
        return Path(self.root_path) / filepath

    def _run_cmd(self, cmd: str, timeout: int = 30) -> str:
        """Execute command in the container context."""
        if isinstance(self.env.deployment, DockerMountDeployment):
            try:
                return self.env.communicate(cmd, check="ignore", timeout=timeout)
            except requests.RequestException as e:
                return f"Tool runtime connection failed: {e}"
        return ""

    @Tool(
        help="Submit final stage-1 feature semantics JSON and finish the task.",
        is_completion=True,
        tip="Call this exactly once with the final JSON string. It writes feature_implementation.json in the current workspace.",
    )
    def submit_feature_semantics(
        self,
        feature_semantics_json: str = Arg(description="Final feature semantics JSON string.", newline=True),  # type: ignore
    ) -> str:
        text = feature_semantics_json.strip() if isinstance(feature_semantics_json, str) else feature_semantics_json
        if isinstance(text, str) and text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        payload = json.loads(text) if isinstance(text, str) else text
        if not isinstance(payload, dict):
            return "Error: feature semantics must be a JSON object."

        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        target_path = self._get_abs_file_path(Path("feature_implementation.json"))
        if self.env:
            self.env.write_file(str(target_path), content)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")

        return "Feature semantics submitted to feature_implementation.json. All tasks have been completed."

    @Tool(
        help="List files and directories.",
        tip="Set path to '.' to list current directory. Useful for exploring project structure.",
    )
    def list_directory(
        self,
        path: str = Arg(description="Path to the directory.", default="."),  # type: ignore
    ) -> str:
        """
        Lists directory contents with `ls -R` command.
        """
        target_path = self._get_abs_file_path(Path(path))

        # 1. Check if path exists and is a directory
        check = self._run_cmd(
            f"if [ -d '{target_path}' ]; then echo 'DIR'; elif [ -f '{target_path}' ]; then echo 'FILE'; else echo 'NONE'; fi"
        )

        if "NONE" in check:
            return f"Error: Path '{path}' does not exist."
        if "FILE" in check:
            return f"{path} is a file."

        # 2. Construct ls command
        # -F: Append indicator (one of */=>@|) to entries
        flags = "-F"

        cmd = f"ls {flags} '{target_path}'"
        output = self._run_cmd(cmd)

        if not output:
            return f"Directory '{path}' is empty."

        return output

    @Tool(
        help="Search for a text pattern within files or directories.",
        tip="Use this to find Macro definitions (e.g. '#define MACRO') or usages. If you know the specific file, pass it in 'path' to speed up the search.",
    )
    def grep_search(
        self,
        query: str = Arg(description="The string or regex pattern to search for (e.g. 'WITH_LZMA')."),
        path: str = Arg(description="The file or directory path to search in.", default="."),
        include_pattern: str = Arg(
            description="Filter files when searching a directory (e.g. '*.c', 'configure.ac'). Ignored if path is a file.",
            default="*",
        ),
        ignore_case: bool = Arg(description="Case insensitive search.", default=False),
        context_lines: int = Arg(
            description="Number of context lines to show around the match.",
            default=0,
        ),
    ) -> str:
        """
        Smart grep wrapper with IO protection.
        Uses shell pipes to strictly limit output size, preventing agent timeouts.
        """
        # [GUARDRAIL 1] Prevent wildcard/empty queries
        # An empty query or '.' in a large repo is a DOS attack on the agent.
        if not query or query.strip() in [".", ".*", ""]:
            return "Error: Query matches everything. Please provide a specific search term."

        # 1. Resolve path
        target_path = self._get_abs_file_path(Path(path))
        safe_path = shlex.quote(str(target_path))

        # [GUARDRAIL 2] Check existence using Shell (avoids host/container mismatch)
        check_exist_cmd = f"test -e {safe_path} && echo 'EXISTS'"
        if self._run_cmd(check_exist_cmd).strip() != "EXISTS":
            return f"Error: Path '{path}' does not exist."

        # 2. Determine if directory
        type_check_cmd = f"if [ -d {safe_path} ]; then echo 'DIR'; else echo 'FILE'; fi"
        path_type = self._run_cmd(type_check_cmd).strip()

        # 3. Construct Grep Flags
        # -I: Ignore binary files
        # -n: Line numbers
        # -H: Print file name
        # --color=never: Avoid ANSI codes confusing the agent
        flags = ["-n", "-I", "-H", "--color=never"]

        if ignore_case:
            flags.append("-i")

        # Models sometimes pass malformed JSON argument values, e.g.
        # {"context_lines": "foo"}. Treat invalid context as zero instead of
        # letting a bad tool argument crash the whole SWE-agent run.
        try:
            context_line_count = int(context_lines)
        except (TypeError, ValueError):
            context_line_count = 0

        # Limit context to avoid exploding output width
        if context_line_count > 0:
            safe_context = min(context_line_count, 5)
            flags.append(f"-C {safe_context}")

        # 4. Recursion & Exclusions
        if path_type == "DIR":
            flags.append("-r")

            # [CRITICAL] Exclude massive/irrelevant directories
            # 'result', 'output', 'tmp' are common culprits in build environments
            exclude_dirs = [
                ".git",
                ".svn",
                ".hg",
                "node_modules",
                "build",
                "dist",
                "vendor",
                "__pycache__",
                "result",
                "output",
                "tmp",
                "logs",
            ]
            for ed in exclude_dirs:
                flags.append(f"--exclude-dir={ed}")

            # Exclude noisy file extensions
            exclude_exts = [
                "html",
                "md",
                "txt",
                "json",
                "xml",
                "map",
                "lock",
                "css",
                "svg",
                "png",
                "jpg",
                "po",
                "pot",
                "log",
            ]
            for ext in exclude_exts:
                flags.append(f"--exclude=*.{ext}")

            # Handle include pattern
            if include_pattern and include_pattern != "*":
                flags.append(f"--include={shlex.quote(include_pattern)}")

        # 5. Build Command with Safety Pipe
        flags_str = " ".join(flags)
        safe_query = shlex.quote(query)

        # Max lines to return.
        # Reducing this helps performance significantly.
        # If the agent needs >100 lines, it should narrow the search.
        MAX_LINES = 100

        # `head` exits after MAX_LINES, so grep can receive SIGPIPE while it is
        # still writing matches. Suppress grep stderr to keep the benign
        # "Broken pipe" warning out of the agent observation.
        cmd = f"grep {flags_str} {safe_query} {safe_path} 2>/dev/null | head -n {MAX_LINES}"

        try:
            output = self._run_cmd(cmd)
        except Exception as e:
            return f"Search execution failed. Error: {str(e)}"

        if not output:
            return f"No matches found for '{query}' in '{path}'."

        # Check if we hit the limit to inform the user
        line_count = len(output.splitlines())
        if line_count >= MAX_LINES:
            output += (
                f"\n\n... [Output truncated by Tool at {MAX_LINES} lines. Please narrow your search query or path.]"
            )

        return output

    @Tool(
        help="Read the content of a file. LIMITATION: You cannot read entire files at once.",
        tip="Defaults to reading the first 50 lines. To read more, verify the file size first or specify a strict `start_line` and `end_line`.",
    )
    def read_file(
        self,
        filepath: str = Arg(description="Path to the file to read."),  # type: ignore
        start_line: int = Arg(description="Start reading from this line (1-based).", default=1),  # type: ignore
        end_line: int = Arg(
            description="Stop reading at this line (inclusive). Default is start_line + 50. -1 is NOT allowed.",
            default=0,  # Changed from -1 to 0 to signal "use default logic"
        ),  # type: ignore
    ) -> str:
        """
        Reads file content with strict safety limits to prevent system crashes.
        """
        MAX_PREVIEW_LINES = 50
        MAX_BATCH_LINES = 500

        target_path = self._get_abs_file_path(Path(filepath))

        # 1. Check existence
        check = self._run_cmd(f"if [ -f '{target_path}' ]; then echo 'EXISTS'; else echo 'MISSING'; fi")
        if "MISSING" in check:
            return f"Error: File '{filepath}' does not exist or is not a regular file."

        # 2. Logic for defaults and safety
        # Case A: User passed -1 (Tried to read all) -> REJECT
        if end_line == -1:
            return (
                "Error: Reading the entire file at once is FORBIDDEN for performance safety.\n"
                "Please specify a specific range (e.g., start_line=1, end_line=100).\n"
                "Use `list_directory` to see file sizes first."
            )

        # Case B: User used default (end_line=0) -> Set to Default Preview
        if end_line == 0:
            actual_end = start_line + MAX_PREVIEW_LINES
        else:
            actual_end = end_line

        # Case C: Safety Check (Range too big)
        count = actual_end - start_line
        if count > MAX_BATCH_LINES:
            return (
                f"Error: You are trying to read {count} lines at once. The limit is {MAX_BATCH_LINES}.\n"
                f"Please read in smaller chunks (e.g., {start_line} to {start_line + MAX_BATCH_LINES})."
            )

        # 3. Construct command (sed)
        # sed -n 'start,endp' file
        cmd = f"sed -n '{start_line},{actual_end}p' '{target_path}'"
        output = self._run_cmd(cmd)

        # 4. Check results
        if not output:
            # Check if file is empty or just range out of bounds
            line_count = self._run_cmd(f"wc -l < '{target_path}'").strip()
            if line_count == "0":
                return "(File is empty)"
            return f"No output. The file has {line_count} lines. Your range {start_line}-{actual_end} might be out of bounds."

        # Add a footer to remind the agent there might be more content
        footer = ""
        if end_line == 0:  # If using default preview
            footer = f"\n\n... (Showing first {MAX_PREVIEW_LINES} lines. Specify start_line/end_line to read more) ..."

        return output + footer

    @Tool(
        help="Find where a C Macro is used to guard code blocks.",
        tip="Use this to quickly identify which files and functions are controlled by a Feature Flag Macro.",
    )
    def find_macro_usages(
        self,
        macro: str = Arg(description="The macro name (e.g. CCITT_SUPPORT)."),
        path: str = Arg(description="Directory to search.", default="."),
    ) -> str:
        """
        Locates usages of a macro in #ifdef / #if defined(...) directives.
        Returns a summarized list of files and context.
        """
        target_path = self._get_abs_file_path(Path(path))

        # 1. Use grep to find potential matches (recursive, line numbers, no binaries)
        # We search specifically for preprocessor directives to reduce noise
        cmd = rf"grep -r -n -I '^\s*#.*{shlex.quote(macro)}' {shlex.quote(str(target_path))}"
        raw_output = self._run_cmd(cmd)

        if not raw_output:
            return f"No preprocessor usages found for '{macro}'."

        # 2. Parse and Summarize
        hits = {}
        lines = raw_output.splitlines()

        for line in lines:
            # Format: filename:line: content
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue

            fpath, lnum, content = parts[0], parts[1], parts[2].strip()

            # Filter: Ignore trivial matches or obvious comments if grep missed them
            if "//" in content and content.index("//") < content.index(macro):
                continue

            # Group by file
            rel_path = str(Path(fpath).relative_to(target_path))
            if rel_path not in hits:
                hits[rel_path] = []

            hits[rel_path].append(f"Line {lnum}: {content}")

        # 3. Format Output for Agent
        output = [f"Found usages of '{macro}' in {len(hits)} files:\n"]

        for fname, contexts in hits.items():
            output.append(f"📄 {fname}")
            # Show first 3 matches per file to avoid flooding
            for ctx in contexts[:3]:
                output.append(f"  └─ {ctx}")
            if len(contexts) > 3:
                output.append(f"  └─ ... ({len(contexts) - 3} more)")
            output.append("")  # Empty line

        return "\n".join(output)

    @Tool(
        help="Analyzes how a specific build flag controls compilation logic.",
        tip="Use this to trace a feature flag (e.g. '--enable-ssl') to its Macros and Source Files. It automatically handles build system detection.",
    )
    def trace_build_flag_logic(
        self,
        flag: str = Arg(description="The feature flag to trace (e.g. '--with-zlib' or 'ENABLE_SSL')."),
        root_path: str = Arg(description="The root directory of the build script.", default="."),
        system_type: str = Arg(
            description="Optional hint: 'autotools', 'cmake', 'nginx'. If empty, auto-detects.", default=""
        ),
    ) -> str:
        """
        Smart Build Tracer.
        Locates the definition and usage of a build flag safely, avoiding massive output floods.
        Returns a JSON string containing the analysis results.
        """
        import json
        import re
        import shlex
        from pathlib import Path

        # 1. Guardrails
        if not flag or len(flag) < 2:
            return json.dumps({"error": "Flag is too short or empty."})

        root = Path(root_path).resolve()

        # Check existence safely
        check_cmd = f"test -e {shlex.quote(str(root))} && echo 'EXISTS'"
        if self._run_cmd(check_cmd).strip() != "EXISTS":
            return json.dumps({"error": f"Path '{root_path}' does not exist inside the container."})

        # 2. Heuristic System Detection
        if not system_type:
            ls_out = self._run_cmd(f"ls {shlex.quote(str(root))}").lower()
            if "cmakelists.txt" in ls_out:
                system_type = "cmake"
            elif "configure.ac" in ls_out or "configure.in" in ls_out:
                system_type = "autotools"
            elif "auto" in ls_out:  # Nginx usually has an 'auto' dir
                system_type = "nginx"
            else:
                system_type = "generic"

        # Initialize output dictionary
        output_data = {"analyzing_mode": system_type.upper(), "flag": flag, "root_path": str(root)}

        # 3. Clean Flag (remove --enable, --with)
        # This helps matching internal variables (e.g. --with-lzma -> LZMA)
        clean_name = re.sub(r"^--?(enable|with|disable|have|use)-", "", flag.lower()).replace("-", "_").upper()
        output_data["clean_name"] = clean_name

        # 4. Execution Strategy

        # --- CMake Strategy ---
        if system_type == "cmake":
            # Search CMakeLists.txt and .cmake files
            # Look for OPTION or SET
            grep_def = rf"grep -r -n -i 'option.*{clean_name}\|set.*{clean_name}' {shlex.quote(str(root))} --include=CMakeLists.txt --include=*.cmake | head -n 20"
            output_data["definitions"] = self._run_cmd(grep_def)

            # Look for IF usage
            grep_use = f"grep -r -n -i -A 5 'if.*{clean_name}' {shlex.quote(str(root))} --include=CMakeLists.txt --include=*.cmake | head -n 50"
            output_data["usage_blocks"] = self._run_cmd(grep_use)

        # --- Autotools Strategy ---
        elif system_type == "autotools":
            # Search configure.ac / configure.in / Makefile.am
            # Use fixed strings for safety when possible
            search_term = flag.replace("--", "")  # e.g. with-lzma

            # 1. Find Arg Definition (AC_ARG_WITH / AC_ARG_ENABLE)
            grep_arg = f"grep -n '{search_term}' {shlex.quote(str(root))}/configure.ac {shlex.quote(str(root))}/configure.in 2>/dev/null | head -n 20"
            output_data["arg_definitions"] = self._run_cmd(grep_arg)

            # 2. Find Macro Definition (AC_DEFINE) using the Clean Name (e.g. LZMA)
            # Context -C 2 is usually enough
            grep_macro = f"grep -r -n -C 2 'AC_DEFINE.*{clean_name}' {shlex.quote(str(root))} --include=configure.ac --include=*.m4 | head -n 50"
            output_data["macro_definitions"] = self._run_cmd(grep_macro)

            # 3. Find File Inclusion (Makefile.am)
            grep_file = f"grep -r -n -C 2 '{clean_name}' {shlex.quote(str(root))} --include=Makefile.am | head -n 50"
            output_data["makefile_logic"] = self._run_cmd(grep_file)

        # --- Generic / Nginx Strategy ---
        else:
            # Fallback but safe: exclude large dirs and limit output
            exclude = "--exclude-dir=.git --exclude-dir=test --exclude-dir=doc --exclude=*.html --exclude=*.po"

            # Search for the exact flag in 'configure' or 'auto/options'
            cmd_def = f"grep -r -n '{flag}' {shlex.quote(str(root))} {exclude} | head -n 20"
            output_data["flag_search"] = self._run_cmd(cmd_def)

            # Search for the variable (clean_name)
            cmd_var = f"grep -r -n -C 2 '{clean_name}' {shlex.quote(str(root))} {exclude} | head -n 50"
            output_data["variable_search"] = self._run_cmd(cmd_var)

        return json.dumps(output_data, indent=2)

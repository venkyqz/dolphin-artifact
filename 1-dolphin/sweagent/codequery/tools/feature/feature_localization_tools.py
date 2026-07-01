import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sweagent.codequery.tools import BaseToolManager
from sweagent.tools.tool_desc import Arg, Tool
from swerex.deployment.docker import DockerMountDeployment

try:
    import tree_sitter_c
    from tree_sitter import Language, Node, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


@dataclass
class CodeRegion:
    name: str
    start_line: int
    end_line: int
    node: Any


class FeatureLocalizationTools(BaseToolManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.env and not isinstance(self.env.deployment, DockerMountDeployment):
            raise ValueError("ASTFeatureTools requires DockerMountDeployment.")
        self._parser: Optional["Parser"] = None
        self._is_initialized = False

    def _ensure_parser(self) -> str:
        if self._is_initialized:
            return ""
        if not TREE_SITTER_AVAILABLE:
            return "Error: tree-sitter not installed."
        try:
            self._parser = Parser(Language(tree_sitter_c.language()))
            self._is_initialized = True
            return ""
        except Exception as e:
            return f"Error initializing Tree-sitter: {e}"

    def _get_abs_local_file_path(self, filepath: Path) -> Path:
        abs_remote_path = Path(filepath) if Path(filepath).is_absolute() else Path(self.env.get_cwd()) / filepath
        if isinstance(self.env.deployment, DockerMountDeployment):
            try:
                rel_path = abs_remote_path.relative_to(self.env.remote_root)
                return self.env.local_root / rel_path
            except ValueError:
                return abs_remote_path
        return abs_remote_path

    def _read_file_bytes(self, filepath: str) -> tuple[Optional[bytes], Optional[str]]:
        err = self._ensure_parser()
        if err:
            return None, err

        path_obj = Path(filepath)
        local_path = self._get_abs_local_file_path(path_obj)
        if not local_path.exists():
            return None, f"Error: File not found at {local_path}"

        try:
            with open(local_path, "rb") as f:
                return f.read(), None
        except Exception as e:
            return None, f"Error reading file: {e}"

    def _extract_text(self, node: "Node", source_bytes: bytes) -> str:
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _get_function_name(self, node: "Node", source_bytes: bytes) -> str:
        declarator = node.child_by_field_name("declarator")
        if not declarator:
            return "unknown"
        curr = declarator
        while curr:
            if curr.type == "identifier":
                return self._extract_text(curr, source_bytes)
            if curr.type == "function_declarator":
                curr = curr.child_by_field_name("declarator")
            elif curr.type in ("pointer_declarator", "parenthesized_declarator"):
                curr = curr.child_by_field_name("declarator")
            else:
                break
        return "unknown"

    def _scan_all_functions(self, root_node: "Node", source_bytes: bytes) -> List[CodeRegion]:
        functions = []
        cursor = root_node.walk()
        visited_children = False

        while True:
            if not visited_children:
                if cursor.node.type == "function_definition":
                    name = self._get_function_name(cursor.node, source_bytes)
                    functions.append(
                        CodeRegion(
                            name=name,
                            start_line=cursor.node.start_point[0],
                            end_line=cursor.node.end_point[0],
                            node=cursor.node,
                        )
                    )
                    visited_children = True

            if not visited_children and cursor.goto_first_child():
                visited_children = False
            elif cursor.goto_next_sibling():
                visited_children = False
            elif cursor.goto_parent():
                visited_children = True
            else:
                break

        return functions

    def _find_macro_regions(self, root_node: "Node", source_bytes: bytes, macro_name: str) -> List[Dict[str, Any]]:
        regions = []
        cursor = root_node.walk()
        visited_children = False

        while True:
            if not visited_children:
                node = cursor.node
                is_match = False
                guard_type = ""

                if node.type == "preproc_ifdef":
                    name_node = node.child_by_field_name("name")
                    if name_node and self._extract_text(name_node, source_bytes) == macro_name:
                        is_match = True
                        guard_type = "ifdef"
                elif node.type == "preproc_ifndef":
                    name_node = node.child_by_field_name("name")
                    if name_node and self._extract_text(name_node, source_bytes) == macro_name:
                        is_match = True
                        guard_type = "ifndef"
                elif node.type in ("preproc_if", "preproc_elif"):
                    cond_node = node.child_by_field_name("condition")
                    if cond_node:
                        cond_text = self._extract_text(cond_node, source_bytes)
                        if re.search(r"\b" + re.escape(macro_name) + r"\b", cond_text):
                            is_match = True
                            guard_type = "if_not_defined" if "!" in cond_text else "if_condition"

                if is_match:
                    regions.append(
                        {
                            "type": guard_type,
                            "start_line": node.start_point[0],
                            "end_line": node.end_point[0]
                        }
                    )

            if not visited_children and cursor.goto_first_child():
                visited_children = False
            elif cursor.goto_next_sibling():
                visited_children = False
            elif cursor.goto_parent():
                visited_children = True
            else:
                break

        return regions

    @Tool(
        help="Analyzes C code to structurally classify functions as ADDED, MODIFIED, or REMOVED by a macro.",
        tip="Pass macro_name='ENTIRE_FILE_ADDED' for added modules, 'ENTIRE_FILE_REMOVED' for excluded modules, or the specific MACRO_NAME for internal #ifdefs.",
    )
    def classify_function_by_macro(
        self,
        filepath: str = Arg(description="Path to the .c file."),  # type: ignore
        macro_name: str = Arg(description="Macro name, 'ENTIRE_FILE_ADDED', or 'ENTIRE_FILE_REMOVED'."),  # type: ignore
    ) -> str:
        """
        Performs spatial analysis to determine function impact, tailored for finding binary structural anchors.
        """
        # Strict enforcement: binary functions live in implementation files
        if not filepath.endswith(".c"):
            return f"Error: Only C source files (.c) are supported for function boundary extraction. Given: {filepath}."

        source_bytes, err = self._read_file_bytes(filepath)
        if err:
            return err

        tree = self._parser.parse(source_bytes)
        root_node = tree.root_node
        all_functions = self._scan_all_functions(root_node, source_bytes)

        if macro_name == "ENTIRE_FILE_ADDED":
            return json.dumps(
                {
                    "filepath": str(filepath),
                    "added_functions": [f.name for f in all_functions],
                    "modified_functions": [],
                    "removed_functions": [],
                },
                indent=2,
            )
        
        if macro_name == "ENTIRE_FILE_REMOVED":
            return json.dumps(
                {
                    "filepath": str(filepath),
                    "added_functions": [],
                    "modified_functions": [],
                    "removed_functions": [f.name for f in all_functions],
                },
                indent=2,
            )

        macro_regions = self._find_macro_regions(root_node, source_bytes, macro_name)

        if not macro_regions:
            return json.dumps(
                {
                    "status": "macro_not_found",
                    "filepath": str(filepath),
                    "added_functions": [],
                    "modified_functions": [],
                    "removed_functions": [],
                },
                indent=2,
            )

        added_funcs = []
        modified_funcs = []
        removed_funcs = []

        for func in all_functions:
            f_start, f_end = func.start_line, func.end_line
            func_status = None

            for region in macro_regions:
                m_start, m_end = region["start_line"], region["end_line"]
                m_type = region["type"]

                # Macro fully wraps the function -> Binary Anchor (Added/Removed)
                if m_start <= f_start and m_end >= f_end:
                    if m_type in ["ifndef", "if_not_defined"]:
                        func_status = "removed"
                    else:
                        func_status = "added"
                    break
                # Macro is inside the function -> Minor Modification (Often noise for stripped binaries)
                elif f_start <= m_start and f_end >= m_end:
                    if func_status is None:
                        func_status = "modified"

            if func_status == "added":
                added_funcs.append(func.name)
            elif func_status == "modified":
                modified_funcs.append(func.name)
            elif func_status == "removed":
                removed_funcs.append(func.name)

        return json.dumps(
            {
                "filepath": str(filepath),
                "added_functions": sorted(list(set(added_funcs))),
                "modified_functions": sorted(list(set(modified_funcs))),
                "removed_functions": sorted(list(set(removed_funcs))),
            },
            indent=2,
        )
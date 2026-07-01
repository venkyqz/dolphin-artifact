import argparse
import json
import logging
import shutil
from pathlib import Path

import tqdm
from util.feature_map import FEATURE_MAP, prepare_project_source

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SourcePreprocessor")

try:
    import tree_sitter_c
    from tree_sitter import Language, Parser

    TS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️  tree-sitter or tree-sitter-c not installed. Run `pip install tree-sitter tree-sitter-c`.")
    TS_AVAILABLE = False

"""
This script performs offline source code preprocessing.
It extracts functions from C projects into a flat directory structure and generates metadata.
Target: To create a 'source_database' that can be mounted into agents later.
"""

METADATA_NAME = "source_metadata.json"
SOURCE_CODE_DIR_NAME = "source_code"


# ==============================================================================
# SOURCE PREPROCESSOR (Tree-sitter Logic)
# ==============================================================================
class SourcePreprocessor:
    def __init__(self):
        if TS_AVAILABLE:
            self.LANGUAGE = Language(tree_sitter_c.language())
            self.parser = Parser(self.LANGUAGE)
        else:
            self.parser = None

    def get_node_text(self, source_bytes, node):
        """Helper to decode node text."""
        if not node:
            return ""
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def extract_identifier(self, node):
        """
        Recursively extract function name identifier from a declarator node.
        """
        curr = node
        while curr:
            if curr.type == "identifier":
                return curr

            # 1. Standard declarator field
            child = curr.child_by_field_name("declarator")
            if child:
                curr = child
                continue

            # 2. Handle pointer/parenthesis wrappers
            if curr.type in (
                "function_declarator",
                "parenthesized_declarator",
                "pointer_declarator",
                "array_declarator",
            ):
                found_next = False
                for c in curr.children:
                    if c.type.endswith("declarator") or c.type == "identifier":
                        curr = c
                        found_next = True
                        break
                if found_next:
                    continue

            break
        return None

    def _analyze_body(self, body_node, content, strings_set, callees_set):
        """
        Deep traversal of function body to extract strings and calls.
        """
        stack = [body_node]
        while stack:
            curr = stack.pop()

            # Extract String Literals
            if curr.type == "string_literal":
                raw_text = self.get_node_text(content, curr)
                if len(raw_text) >= 2:
                    clean_text = raw_text[1:-1]
                    strings_set.add(clean_text)

            # Extract Function Calls
            elif curr.type == "call_expression":
                func_node = curr.child_by_field_name("function")
                if func_node:
                    callee_name = self.get_node_text(content, func_node)
                    callees_set.add(callee_name)

            stack.extend(reversed(curr.children))

    def parse_file(self, file_path):
        """
        Parse a single C file and return function metadata list.
        """
        if not self.parser:
            return []

        try:
            with open(file_path, "rb") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []

        tree = self.parser.parse(content)
        root = tree.root_node

        functions_data = []
        stack = [root]

        while stack:
            node = stack.pop()

            # 1. Found a Function Definition
            if node.type == "function_definition":
                declarator = node.child_by_field_name("declarator")
                if declarator:
                    id_node = self.extract_identifier(declarator)
                    if id_node:
                        func_name = self.get_node_text(content, id_node)

                        # FILTER: Ignore UPPERCASE function names (likely macros)
                        if not func_name.isupper():
                            func_source = self.get_node_text(content, node)

                            # Note: NO FORMATTING here to ensure speed

                            strings = set()
                            callees = set()

                            body_node = node.child_by_field_name("body")
                            if body_node:
                                self._analyze_body(body_node, content, strings, callees)

                            functions_data.append(
                                {
                                    "function_name": func_name,
                                    "source_code": func_source,
                                    "strings": list(strings),
                                    "callees": list(callees),
                                }
                            )

                continue

            # 2. Continue Traversal
            stack.extend(reversed(node.children))

        return functions_data


# ==============================================================================
# MAIN LOGIC
# ==============================================================================


def analyze_source_code(source_root: Path, output_db_dir: Path):
    """
    Statically analyzes source code using SourcePreprocessor.
    """
    if not TS_AVAILABLE:
        logger.error("Tree-sitter not available. Skipping analysis.")
        return

    logger.info(f"🔍 [Pre-process] Analyzing source at {source_root} -> {output_db_dir}...")

    # Setup directories
    source_flat_dir = output_db_dir / SOURCE_CODE_DIR_NAME
    if source_flat_dir.exists():
        shutil.rmtree(source_flat_dir)
    source_flat_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = output_db_dir / METADATA_NAME
    preprocessor = SourcePreprocessor()
    metadata_list = []

    count_files = 0
    c_files = list(source_root.rglob("*.c"))

    for c_file in tqdm.tqdm(c_files, desc=f"Parsing {source_root.name}"):
        try:
            rel_filename = c_file.relative_to(source_root).as_posix()
        except ValueError:
            rel_filename = c_file.name

        extracted_funcs = preprocessor.parse_file(c_file)

        for func_data in extracted_funcs:
            func_name = func_data["function_name"]

            # Generate flat filename: funcname+filename.c
            flat_filename = f"{func_name}+{c_file.stem}.c"
            flat_file_path = source_flat_dir / flat_filename

            # Write source snippet
            with open(flat_file_path, "w", encoding="utf-8") as out_f:
                # out_f.write(f"// File: {rel_filename}\n")
                out_f.write(f"// Function: {func_name}\n")
                out_f.write(func_data["source_code"])

            # Append metadata
            metadata_list.append(
                {
                    "file_name": c_file.stem,
                    "snippet_path": f"{SOURCE_CODE_DIR_NAME}/{flat_filename}",
                    "function_name": func_name,
                    "strings": func_data["strings"],
                    "callees": func_data["callees"],
                }
            )
        count_files += 1

    # Save Metadata
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    logger.info(f"✅ [Done] Saved DB to {output_db_dir} (Scanned {count_files} files, {len(metadata_list)} functions)")


def prepare_source_database(source_dir, db_output_dir):
    """
    Ensures the source database (flat code + metadata) exists.
    """
    metadata_path = db_output_dir / METADATA_NAME
    if not metadata_path.exists():
        analyze_source_code(source_dir, db_output_dir)
    else:
        logger.info(f"⏭️  Database exists for {db_output_dir.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Localization Source Preprocessor")
    parser.add_argument("--oss_root", default="oss", help="Directory containing source tarballs")
    args = parser.parse_args()

    # Base Paths
    BASE_DIR = Path.cwd()
    TEMP_SRC_DIR = BASE_DIR / "temp_sources"
    SOURCE_DATABASE_DIR = BASE_DIR / "source_code"

    for p in [TEMP_SRC_DIR, SOURCE_DATABASE_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    oss_path = Path(args.oss_root)

    # Locate all tarballs
    tar_files = list(oss_path.glob("*.tar.gz"))
    if not tar_files:
        logger.warning(f"No .tar.gz files found in {oss_path}")

    logger.info("📦 [Step 1] Preparing Source Codes & Preprocessing...")

    for tar_file in tar_files:
        # 1. Extract Source
        project_name, source_dir = prepare_project_source(tar_file, TEMP_SRC_DIR)

        if not project_name or project_name not in FEATURE_MAP:
            continue

        # 2. Preprocess (Analyze Source & Generate DB)
        # Output: source_code/nginx/source_metadata.json etc.
        db_output_dir = Path(SOURCE_DATABASE_DIR) / project_name
        db_output_dir.mkdir(parents=True, exist_ok=True)

        prepare_source_database(source_dir, db_output_dir)

    logger.info("🏁 All Preprocessing Completed.")

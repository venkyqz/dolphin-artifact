import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RECOVERED_CONFIGURATION_FILE = "recovered_configuration.json"
RECOVERY_EVIDENCE_FILE = "recovery_evidence.json"
PRIORITIZED_PATHS_FILE = "prioritized_paths.json"
RETRIEVAL_METADATA_FILE = "retrieval_metadata.json"
MAX_DECOMPILED_WINDOW_LINES = 80


class FeatureSandbox:
    """Minimal stage-2 sandbox for objective prioritized path verification.

    Acts as a pure data provider. Source-code reading capabilities are removed
    to prevent hallucinations. The LLM relies on the feature contract, optional
    feature hints exposed by the hint-aware start tool, fast-path string evidence,
    and prioritized decompiled binary code.
    """

    def __init__(self):
        self.context: dict[str, Any] | None = None
        self.feature_definition: dict[str, Any] = {}
        self.retrieval_metadata: dict[str, Any] = {}
        self.paths: list[dict[str, Any]] = []
        self.binary_records: dict[str, dict[str, Any]] = {}
        self.target_strings: list[str] | None = None

        self.path_cursor = 0
        self.node_cursor = 0
        self.line_cursor = 0
        self.read_binary_ids: list[str] = []
        self.root_path: Path | None = None
        self.env = None

    def workspace_root(self) -> Path:
        if self.env and hasattr(self.env, "local_root"):
            return Path(self.env.local_root)
        return Path(self.root_path) if self.root_path else Path()

    def ensure_loaded(self) -> tuple[bool, str]:
        if self.context is not None:
            return True, "already loaded"

        problem_path = self.workspace_root() / "problem_statement.md"
        if not problem_path.exists():
            return False, "problem_statement.md missing"

        context_path = self._resolve_context_path(
            problem_path.read_text(encoding="utf-8", errors="replace")
        )
        if not context_path or not context_path.exists():
            return False, "tool context not found"

        self.context = json.loads(
            context_path.read_text(encoding="utf-8", errors="replace")
        )
        feature_path = Path(str(self.context.get("feature_definition_path", "")))
        embedding_dir = Path(str(self.context.get("embedding_dir", "")))

        if not feature_path.exists() or not embedding_dir.exists():
            return False, "Required directories/files not found."

        self.feature_definition = json.loads(
            feature_path.read_text(encoding="utf-8", errors="replace")
        )

        try:
            self._load_prioritized_paths(embedding_dir)
            self._load_records(
                embedding_dir / "binary_records.json", self.binary_records
            )

            meta_path = embedding_dir / RETRIEVAL_METADATA_FILE
            if meta_path.exists():
                self.retrieval_metadata = json.loads(
                    meta_path.read_text(encoding="utf-8", errors="replace")
                )
        except Exception as exc:
            self.context = None
            return False, f"Failed to load evidence: {exc}"

        if not self.paths:
            return False, "No prioritized paths found."

        active_path = self.paths[0]
        has_code = any(
            self.binary_records.get(node.get("binary_id"), {}).get("code")
            for node in active_path.get("binary_path", [])
            if isinstance(node, dict)
        )
        if not has_code:
            return False, "prioritized path code missing; rerun path prioritization"

        return True, "ok"

    def _resolve_context_path(self, problem_text: str) -> Path | None:
        for pat in [
            r"Tool context JSON:\s*`?([^`\n]+)`?",
            r"TOOL_CONTEXT_JSON:\s*`?([^`\n]+)`?",
        ]:
            match = re.search(pat, problem_text)
            if match:
                return Path(match.group(1).strip()).expanduser()

        for pat in [
            r"Tool context ID:\s*`?([^`\n]+)`?",
            r"Task ID:\s*`?([^`\n]+)`?",
        ]:
            match = re.search(pat, problem_text)
            if match:
                return (
                    self.workspace_root().parent
                    / "_tool_data"
                    / match.group(1).strip()
                    / "context.json"
                )

        return None

    def _load_prioritized_paths(self, embedding_dir: Path) -> None:
        path = embedding_dir / PRIORITIZED_PATHS_FILE
        if not path.exists():
            raise FileNotFoundError(path)

        content = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        loaded = [content] if isinstance(content, dict) else content
        loaded.sort(key=lambda x: int(x.get("path_rank") or 10**9))
        self.paths = loaded[:1]

    def _load_records(self, path: Path, target_dict: dict[str, dict[str, Any]]) -> None:
        if not path.exists():
            return

        records = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        for record in records:
            if not isinstance(record, dict):
                continue
            if record.get("id"):
                target_dict[str(record.get("id"))] = record
            if record.get("name"):
                target_dict[str(record.get("name"))] = record

    def get_start_payload(self) -> str:
        """Return blind/no-hint start context for baseline verification."""
        active_path = self.paths[0] if self.paths else {}
        contract = self._minimal_feature_contract()
        instruction = (
            "Call `check_fast_path_strings` FIRST. If it returns "
            "`FAST_PATH_DECISION`, submit that decision. Only use "
            "`explore_prioritized_path` when it returns `FAST_PATH_UNAVAILABLE`."
        )

        return json.dumps(
            {
                "status": "READY",
                "hint_mode": False,
                "feature_contract": contract,
                "prioritized_path_node_count": len(
                    active_path.get("binary_path", [])
                    if isinstance(active_path, dict)
                    else []
                ),
                "verification_mode": (
                    active_path.get("verification_mode", "path_presence")
                    if isinstance(active_path, dict)
                    else "path_presence"
                ),
                "instruction": instruction,
            },
            indent=2,
        )

    def get_start_payload_hint(self) -> str:
        """Return hint-aware start context.

        Feature hints are exposed as semantic verification criteria. They are not
        evidence that the prioritized path is correct and do not prove feature
        presence by themselves.
        """
        active_path = self.paths[0] if self.paths else {}
        contract = self._hinted_feature_contract(active_path)
        instruction = (
            "Call `check_fast_path_strings` FIRST. If it returns "
            "`FAST_PATH_DECISION`, submit that decision. Only use "
            "`explore_prioritized_path` when it returns `FAST_PATH_UNAVAILABLE`. "
            "Feature hints are included as semantic verification criteria; "
            "they are not retrieval proof."
        )
        if contract.get("gate_hint_functions"):
            instruction += (
                " Gate hints are present; verify gate/dispatch/config/default-"
                "initialization semantics before using helper-only evidence for ON."
            )

        return json.dumps(
            {
                "status": "READY",
                "hint_mode": True,
                "feature_contract": contract,
                "prioritized_path_node_count": len(
                    active_path.get("binary_path", [])
                    if isinstance(active_path, dict)
                    else []
                ),
                "verification_mode": (
                    active_path.get("verification_mode", "path_presence")
                    if isinstance(active_path, dict)
                    else "path_presence"
                ),
                "instruction": instruction,
            },
            indent=2,
        )

    def check_fast_path_strings_payload(self) -> str:
        active_path = self.paths[0] if self.paths else {}
        fast_evidence = active_path.get("fast_path_evidence", {})
        on_cands = fast_evidence.get("on_strings", [])
        off_cands = fast_evidence.get("off_strings", [])

        if not on_cands and not off_cands:
            return json.dumps(
                {
                    "status": "FAST_PATH_UNAVAILABLE",
                    "instruction": (
                        "No unique global strings exist for this feature's "
                        "implementation. You MUST fallback to "
                        "`explore_prioritized_path` to verify decompiled "
                        "semantics manually."
                    ),
                },
                indent=2,
            )

        try:
            target_text = "\n".join(self._load_target_strings())
        except Exception as exc:
            return json.dumps(
                {"error": f"Failed to load binary strings dump: {exc}"}, indent=2
            )

        found_on = [s for s in on_cands if s in target_text]
        found_off = [s for s in off_cands if s in target_text]

        if on_cands:
            if set(on_cands).issubset(found_on):
                return json.dumps(
                    {
                        "status": "FAST_PATH_DECISION",
                        "decision": "on",
                        "is_exist": True,
                        "found_on_strings": found_on,
                        "checked_on_strings": on_cands,
                        "evidence": (
                            "Found all unique feature-enabled implementation "
                            "strings in the target binary."
                        ),
                        "instruction": (
                            "Submit ON immediately with this evidence. Do not "
                            "explore the decompiled path."
                        ),
                    },
                    indent=2,
                )
            return json.dumps(
                {
                    "status": "FAST_PATH_UNAVAILABLE",
                    "found_on_strings": found_on,
                    "checked_on_strings": on_cands,
                    "instruction": (
                        "Unique feature-enabled strings are absent or partial. "
                        "You MUST fallback to `explore_prioritized_path` to "
                        "verify decompiled semantics manually."
                    ),
                },
                indent=2,
            )

        if off_cands:
            if set(off_cands).issubset(found_off):
                return json.dumps(
                    {
                        "status": "FAST_PATH_DECISION",
                        "decision": "off",
                        "is_exist": False,
                        "found_off_strings": found_off,
                        "checked_off_strings": off_cands,
                        "evidence": (
                            "Found all unique feature-disabled strings in the "
                            "target binary."
                        ),
                        "instruction": (
                            "Submit OFF immediately with this evidence. Do not "
                            "explore the decompiled path."
                        ),
                    },
                    indent=2,
                )
            return json.dumps(
                {
                    "status": "FAST_PATH_UNAVAILABLE",
                    "found_off_strings": found_off,
                    "checked_off_strings": off_cands,
                    "instruction": (
                        "Unique feature-disabled strings are absent or partial. "
                        "You MUST fallback to `explore_prioritized_path` to "
                        "verify decompiled semantics manually."
                    ),
                },
                indent=2,
            )

        return json.dumps(
            {
                "status": "FAST_PATH_UNAVAILABLE",
                "found_on_strings": found_on,
                "found_off_strings": found_off,
                "instruction": (
                    "Unique-string evidence is absent or partial. You MUST "
                    "fallback to `explore_prioritized_path` to verify decompiled "
                    "semantics manually."
                ),
            },
            indent=2,
        )

    def _minimal_feature_contract(self) -> dict[str, Any]:
        """Create the blind/no-hint feature contract."""
        feature_name = str(
            (self.context or {}).get("feature_name")
            or self.feature_definition.get("feature")
            or ""
        )
        contract: dict[str, Any] = {"feature": feature_name}

        for key in ["feature-on-flag", "feature-off-flag", "description", "summary"]:
            if self.feature_definition.get(key) not in (None, "", []):
                contract[key] = self.feature_definition[key]

        return contract

    def _hinted_feature_contract(self, active_path: dict[str, Any]) -> dict[str, Any]:
        """Create a hint-aware feature contract.

        Exposes all feature hints from the feature definition, independent of
        whether their source functions appear in the prioritized path. Path
        relevance is provided only as auxiliary metadata.
        """
        contract = self._minimal_feature_contract()
        contract["hint_mode"] = True
        active_names = self._active_source_function_names(active_path)

        feature_hints: list[dict[str, Any]] = []
        for order, hint in enumerate(
            self.feature_definition.get("feature_hints", []) or []
        ):
            if not isinstance(hint, dict):
                continue
            function_name = hint.get("function_name")
            feature_hints.append(
                {
                    "order": order,
                    "function_name": function_name,
                    "source_file": hint.get("source_file") or hint.get("filepath"),
                    "kind": hint.get("kind"),
                    "expected_binary_observation": hint.get(
                        "expected_binary_observation"
                    ),
                    "semantics": hint.get("semantics"),
                    "path_relevant": bool(
                        function_name and function_name in active_names
                    ),
                }
            )

        if feature_hints:
            contract["feature_hints"] = feature_hints
            contract["path_relevant_hint_functions"] = sorted(
                {
                    str(hint.get("function_name"))
                    for hint in feature_hints
                    if hint.get("function_name") and hint.get("path_relevant")
                }
            )

            gate_terms = (
                "flag",
                "registration",
                "dispatch",
                "parser",
                "config",
                "gate",
                "header",
                "default",
                "initialization",
                "entrypoint",
            )
            gate_hints = [
                hint.get("function_name")
                for hint in feature_hints
                if any(
                    term
                    in str(
                        hint.get("semantics")
                        or hint.get("expected_binary_observation")
                        or hint.get("kind")
                        or ""
                    ).lower()
                    for term in gate_terms
                )
            ]
            if gate_hints:
                contract["gate_hint_functions"] = sorted(
                    {str(name) for name in gate_hints if name}
                )

        return contract

    def _active_source_function_names(self, active_path: dict[str, Any]) -> set[str]:
        names: set[str] = set()

        for node in (
            active_path.get("source_path", []) if isinstance(active_path, dict) else []
        ):
            if not isinstance(node, dict):
                continue
            name = node.get("function_name") or node.get("source_function")
            if name:
                names.add(str(name))

        for bnode in (
            active_path.get("binary_path", []) if isinstance(active_path, dict) else []
        ):
            if not isinstance(bnode, dict):
                continue
            for snode in bnode.get("aligned_source_nodes", []) or []:
                if not isinstance(snode, dict):
                    continue
                name = snode.get("source_function") or snode.get("function_name")
                if name:
                    names.add(str(name))

        return names

    def _load_target_strings(self) -> list[str]:
        if self.target_strings is not None:
            return self.target_strings

        strings_path = Path(str((self.context or {}).get("target_strings_path") or ""))
        if not strings_path.exists():
            error = str((self.context or {}).get("target_strings_error") or "")
            raise FileNotFoundError(
                f"target strings dump missing: {strings_path or '<missing>'} {error}".strip()
            )

        self.target_strings = [
            line
            for line in strings_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line
        ]
        return self.target_strings

    def _active_binary_path(self) -> list[dict[str, Any]]:
        active_path = self.paths[0] if self.paths else {}
        return (
            active_path.get("binary_path", []) if isinstance(active_path, dict) else []
        )

    def explore_prioritized_path_payload(
        self, window_lines: int = 80, advance: str = "window"
    ) -> str:
        if not self.paths:
            return json.dumps({"status": "NO_PRIORITIZED_PATH"}, indent=2)

        binary_path = self._active_binary_path()
        if self.node_cursor >= len(binary_path):
            return json.dumps(
                {
                    "status": "PATH_EXHAUSTED",
                    "instruction": (
                        "No more useful prioritized-path windows are available. "
                        "Submit your final decision."
                    ),
                },
                indent=2,
            )

        action = str(advance or "").strip().lower()
        if action in {"next_function", "skip_function", "skip"}:
            self.node_cursor += 1
            self.line_cursor = 0
            if self.node_cursor >= len(binary_path):
                return json.dumps({"status": "PATH_EXHAUSTED"}, indent=2)
        elif action in {"current_function_start", "function_start", "start"}:
            self.line_cursor = 0

        try:
            window_lines = int(window_lines)
        except (TypeError, ValueError):
            window_lines = MAX_DECOMPILED_WINDOW_LINES
        window_lines = max(1, min(window_lines, MAX_DECOMPILED_WINDOW_LINES))

        node = binary_path[self.node_cursor]
        binary_id = node.get("binary_id")
        record = self.binary_records.get(str(binary_id)) or self.binary_records.get(
            binary_id
        )
        code_lines = (
            record.get("code", "") if record else "// unavailable"
        ).splitlines()

        if binary_id not in self.read_binary_ids:
            self.read_binary_ids.append(binary_id)

        start = self.line_cursor
        end = min(start + window_lines, len(code_lines))
        window_code = "\n".join(code_lines[start:end])
        function_done = end >= len(code_lines)

        self.line_cursor = end
        if function_done:
            self.node_cursor += 1
            self.line_cursor = 0

        return json.dumps(
            {
                "status": "PATH_WINDOW",
                "position": {
                    "node": self.node_cursor + (0 if function_done else 1),
                    "total_nodes": len(binary_path),
                },
                "window": {
                    "start_line": start + 1,
                    "end_line": end,
                    "total_lines": len(code_lines),
                },
                "binary_function": {
                    "binary_name": node.get("binary_name"),
                    "code": window_code,
                },
                "instruction": (
                    "Use only this decompiled code window as evidence. Verify the "
                    "actual semantics against the feature contract. Check exact "
                    "constants, control flow, data flow, calls, state mutation, "
                    "registration/dispatch/config gates, and feature-specific "
                    "behavior. Call advance='window' to continue this function, "
                    "advance='current_function_start' to restart from the function "
                    "prologue, or advance='next_function' if this function is "
                    "clearly generic or unrelated."
                ),
            },
            indent=2,
        )

    def save_decision_and_exit(
        self, is_exist: bool, evidence: str, reasoning: str, confidence: float
    ) -> str:
        decision = "on" if is_exist else "off"
        feature = str((self.context or {}).get("feature_name", "unknown"))
        active_path = self.paths[0] if self.paths else {}

        recovered = {
            "features": {feature: decision},
            "evidence": {feature: evidence},
        }
        evidence_payload = {
            "task_id": (self.context or {}).get("task_id"),
            "feature": feature,
            "decision": decision,
            "confidence": confidence,
            "evidence": evidence,
            "reasoning": reasoning,
            "retrieval_provider": self.retrieval_metadata.get(
                "provider", "b2sfinder-anchor-path"
            ),
            "active_path_id": (
                active_path.get("path_id") if isinstance(active_path, dict) else None
            ),
            "read_binary_ids": self.read_binary_ids,
            "created_at": datetime.now(UTC).isoformat(),
        }

        root = self.workspace_root()
        (root / RECOVERED_CONFIGURATION_FILE).write_text(
            json.dumps(recovered, indent=2) + "\n", encoding="utf-8"
        )
        (root / RECOVERY_EVIDENCE_FILE).write_text(
            json.dumps(evidence_payload, indent=2) + "\n", encoding="utf-8"
        )
        return json.dumps({"status": "SUCCESS", "decision": decision}, indent=2)
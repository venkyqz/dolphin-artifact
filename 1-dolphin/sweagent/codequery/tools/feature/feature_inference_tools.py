import json
from pathlib import Path

from sweagent.codequery.tools import BaseToolManager
from sweagent.tools.tool_desc import Arg, Tool
from swerex.deployment.docker import DockerMountDeployment

from sweagent.codequery.tools.internal.feature_sandbox_state import FeatureSandbox


class FeatureInferenceTools(BaseToolManager):
    """Stage-2 feature recovery tools.

    Provides tools for the agent to evaluate pre-computed fast-path unique
    strings, explore the prioritized decompiled path, and submit a final ON/OFF
    decision.

    Source-code reading is intentionally disabled. The agent relies only on:
      - the feature contract returned by start_investigation*;
      - fast-path unique-string evidence;
      - prioritized decompiled binary path windows.

    This tool manager intentionally exposes no tournament-style tools.
    """

    def __init__(self, root_path, env=None, memory=None):
        super().__init__(root_path, env, memory)
        if self.env and not isinstance(self.env.deployment, DockerMountDeployment):
            raise ValueError("FeatureInferenceTools requires DockerMountDeployment.")
        self._sandbox = FeatureSandbox()
        self._sandbox.root_path = Path(self.root_path)
        self._sandbox.env = self.env

    def _ensure_loaded_or_error(self) -> tuple[bool, str | None]:
        ok, msg = self._sandbox.ensure_loaded()
        if not ok:
            return False, json.dumps({"error": msg}, indent=2)
        return True, None

    @Tool(
        help=(
            "Start stage-2 verification and fetch the objective tool context "
            "without exposing feature hints. Use this for blind/no-hint baselines."
        ),
        tip="Call this first for blind/no-hint verification.",
    )
    def start_investigation(self) -> str:
        ok, error = self._ensure_loaded_or_error()
        if not ok:
            return error or json.dumps({"error": "failed to load sandbox"}, indent=2)
        return self._sandbox.get_start_payload()

    @Tool(
        help=(
            "Start stage-2 verification and fetch the objective tool context "
            "with feature hints exposed as semantic verification criteria. "
            "Hints are not retrieval proof and do not prove feature presence by themselves."
        ),
        tip="Call this first for hint-aware verification.",
    )
    def start_investigation_hint(self) -> str:
        ok, error = self._ensure_loaded_or_error()
        if not ok:
            return error or json.dumps({"error": "failed to load sandbox"}, indent=2)
        return self._sandbox.get_start_payload_hint()

    @Tool(
        help=(
            "Check for the presence of pre-computed unique feature strings "
            "(the fast path). Call this before decompiled-code exploration."
        )
    )
    def check_fast_path_strings(self) -> str:
        ok, error = self._ensure_loaded_or_error()
        if not ok:
            return error or json.dumps({"error": "failed to load sandbox"}, indent=2)
        return self._sandbox.check_fast_path_strings_payload()

    @Tool(
        help=(
            "Explore the prioritized decompiled binary function path in code "
            "windows, from head to tail (the fallback path)."
        )
    )
    def explore_prioritized_path(
        self,
        window_lines: int = Arg(
            default=80,
            description="Number of decompiled code lines to return in this window.",
        ),
        advance: str = Arg(
            default="window",
            description=(
                "Use 'window' for the next chunk, 'current_function_start' to "
                "restart the current binary function from its prologue, or "
                "'next_function' to skip the rest of the current binary function."
            ),
        ),
    ) -> str:
        ok, error = self._ensure_loaded_or_error()
        if not ok:
            return error or json.dumps({"error": "failed to load sandbox"}, indent=2)
        return self._sandbox.explore_prioritized_path_payload(window_lines, advance)

    @Tool(help="Persist the final feature presence decision.", is_completion=True)
    def submit_presence_decision(
        self,
        is_exist: bool = Arg(
            description=(
                "True if executable feature implementation semantics are present "
                "in the binary. False otherwise."
            )
        ),
        evidence: str = Arg(
            description=(
                "Concise evidence summary. Cite either decisive fast-path strings "
                "or concrete decompiled-code behavior."
            )
        ),
        reasoning: str = Arg(
            description=(
                "Evidence-based reasoning explaining how the observed binary "
                "semantics match or contradict the feature contract."
            )
        ),
        confidence: float = Arg(
            default=0.0,
            description="Confidence level from 0.0 to 1.0.",
        ),
    ) -> str:
        ok, error = self._ensure_loaded_or_error()
        if not ok:
            return error or json.dumps({"error": "failed to load sandbox"}, indent=2)
        return self._sandbox.save_decision_and_exit(
            is_exist=is_exist,
            evidence=evidence,
            reasoning=reasoning,
            confidence=confidence,
        )

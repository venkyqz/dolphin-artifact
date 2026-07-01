import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby, zip_longest
from pathlib import Path

import tqdm
from util.feature_map import FEATURE_MAP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Runner")

BINARY_UNDER_TEST = "target"


def get_formatted_feature_list(project_name):
    if project_name not in FEATURE_MAP:
        return None
    lines = []
    for idx, (name, flag) in enumerate(FEATURE_MAP[project_name], 0):
        lines.append(f"[{idx}] {name} (Flag: {flag})")
    return "\n".join(lines)


def get_json_template(project_name):
    if project_name not in FEATURE_MAP:
        return "{}"
    template = {name: "on/off" for name, _ in FEATURE_MAP[project_name]}
    json_str = json.dumps(template, indent=4)
    return json_str[1:-1]


# ==============================================================================
# WORKER FUNCTION
# ==============================================================================
def run_recovery_agent(target_dir, config):
    target_path = Path(target_dir)
    project_name = target_path.parent.name
    sha_id = target_path.name

    # 1. Basic Setup & Validation
    if project_name not in FEATURE_MAP:
        return None  # Skip unknown projects

    # Declare variables for Prompt
    feature_details = get_formatted_feature_list(project_name)
    json_template = get_json_template(project_name)

    BASE_DIR = Path(config["BASE_DIR"])
    worker_workspace = BASE_DIR / "workspace" / config["WORKDIR_NAME"] / sha_id

    os.chdir(worker_workspace)

    if not worker_workspace.exists():
        return f"❌ ERROR: {sha_id} - Workspace not found. Ensure Phase 2 completed."

    if (worker_workspace / "recovered_configuration.json").exists():
        return {"skipped": True, "status": f"⏭️  SKIPPING: {sha_id} - Result exists."}

    # logger.info(f"🔧 Setting up workspace for {sha_id}...")

    # find the first executable linkable file in target_path
    # by checking the magic number
    bin_name = BINARY_UNDER_TEST

    # 2. Setup Logging & Paths
    # custom_id = f"{project_name}_{sha_id}"
    # log_path = Path(config["LOG_DIR"]) / f"{custom_id}.log"
    # traj_dir = Path(config["TRAJ_DIR"]) / config["WORKDIR_NAME"] / custom_id
    # expected_result_file = worker_workspace / "recovered_configuration.json"

    custom_id = f"{project_name}_{sha_id}"
    log_path = Path(config["LOG_DIR"]) / f"{custom_id}.log"
    traj_dir = Path(config["TRAJ_DIR"]) / config["WORKDIR_NAME"] / custom_id
    expected_result_file = worker_workspace / "recovered_configuration.json"

    if expected_result_file.exists():
        return f"⏭️  SKIPPING: {sha_id} - Result exists."

    # --------------------------------------------------------------------------
    # 3. [BRIDGE] Import Feature Definitions (Phase 1 Output)
    # --------------------------------------------------------------------------
    source_defs_repo = Path(config["SOURCE_DEFS_DIR"]) / project_name
    workspace_defs_dir = worker_workspace / "feature_definitions"
    workspace_defs_dir.mkdir(exist_ok=True)

    project_features = FEATURE_MAP.get(project_name, [])

    for feature_tuple in project_features:
        feat_name = feature_tuple[0]
        src_json = source_defs_repo / f"{feat_name}.json"

        if src_json.exists():
            shutil.copy(src_json, workspace_defs_dir / f"{feat_name}.json")

        else:
            # You might want to log missing definitions here
            pass

    # --------------------------------------------------------------------------
    # 4. [BRIDGE] Import Source Code Repo
    # --------------------------------------------------------------------------
    # This copies the preprocessed source code from source_code to the worker workspace,
    # allowing the agent to perform s2b analysis or grep source.
    source_db_dir = Path(config["SOURCE_DATABASE_DIR"])

    # 定义目标路径 (Destination)
    dest_src_code_dir = worker_workspace / "source_code"
    dest_metadata_file = worker_workspace / "source_metadata.json"

    if not dest_src_code_dir.exists():
        # 定义源路径 (Source)
        project_src_code_dir = source_db_dir / project_name / "source_code"
        project_metadata_file = source_db_dir / project_name / "source_metadata.json"

        # 1. 复制目录: 使用 copytree 而不是 copy
        # dirs_exist_ok=True 允许目录已存在的情况下覆盖内容 (Python 3.8+)
        if project_src_code_dir.exists():
            shutil.copytree(project_src_code_dir, dest_src_code_dir, dirs_exist_ok=True)

        # 2. 复制 Metadata 文件: 修复变量名覆盖问题
        if project_metadata_file.exists():
            shutil.copy(project_metadata_file, dest_metadata_file)

    # --------------------------------------------------------------------------
    # 5. Construct Problem Statement
    # --------------------------------------------------------------------------
    problem_file = worker_workspace / "problem_statement.md"

    problem_content = textwrap.dedent(
        f"""
        TARGET CONTEXT
        - Project: {project_name}
        - Target ID: {sha_id}

        GOAL
        Recover the build configuration of the binary located at:
        `target`

        RESOURCES
        1. Feature Queue: Managed automatically by your tools.
        2. Decompiled Code: `decompiled_code/`
        3. Source Code: `source_code/`

        YOUR JOB
        You are a worker in a loop.
        Simply initialize the environment, then pull tasks one by one until the queue is empty.

        Verify each feature using the evidence tools provided, and strictly follow the state machine defined in your system prompt.
        """
    ).strip()

    problem_file.write_text(problem_content)

    cwd = worker_workspace
    # 6. Run SWE-Agent
    cmd = [
        "uv",
        "run",
        "sweagent",
        "run",
        f"--output_dir={traj_dir}",
        "--env.deployment.type=mount",
        f"--env.deployment.local_mount_path={worker_workspace}",
        f"--config={config['AGENT_CONFIG']}",
        f"--problem_statement.path={problem_file}",
    ]

    # NOTE: Temporarily disabled execution for testing purposes.

    with open(log_path, "w") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=cwd)

    # 7. Check Result
    if expected_result_file.exists():
        return f"✅ SUCCESS: {sha_id} ({project_name})"
    else:
        return f"❌ FAILED:  {sha_id} ({project_name}) -> Check log: {log_path}"


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Configuration Recovery Agent")
    parser.add_argument("workdir_name", help="Name of the agent (e.g., gpt4o-mini)")
    parser.add_argument("config_path", help="Path to agent config.yaml")
    parser.add_argument("bench_root", help="Path to bench root")
    parser.add_argument("--oss_root", default="oss", help="Directory containing source tarballs")
    parser.add_argument("-j", "--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    # Base Paths
    BASE_DIR = Path.cwd()

    config = {
        "WORKDIR_NAME": args.workdir_name,
        "AGENT_CONFIG": args.config_path,
        "BENCH_ROOT": args.bench_root,
        "BASE_DIR": str(BASE_DIR),
        # Directories
        "STRING_DATABASE_DIR": str(BASE_DIR / "string_databases"),
        "SOURCE_DEFS_DIR": str(BASE_DIR / "feature_definitions_source"),
        "LOG_DIR": str(BASE_DIR / f"logs/{args.workdir_name}"),
        "TRAJ_DIR": str(BASE_DIR / f"trajectories/{args.workdir_name}"),
        "SOURCE_DATABASE_DIR": str(BASE_DIR / "source_code"),
    }

    oss_path = Path(args.oss_root)
    # Locate all tarballs
    tar_files = list(oss_path.glob("*.tar.gz"))
    if not tar_files:
        logger.warning(f"No .tar.gz files found in {oss_path}")
    # return

    # Ensure directories exist
    for k in ["LOG_DIR", "TRAJ_DIR"]:
        Path(config[k]).mkdir(parents=True, exist_ok=True)

    logger.info(f"======================================================================")
    logger.info(f"🚀 PHASE 3: RECOVERY AGENT START")
    logger.info(f"   Agent:       {config['WORKDIR_NAME']}")
    logger.info(f"   Sources:     {config['SOURCE_DATABASE_DIR']}")
    logger.info(f"   Definitions: {config['SOURCE_DEFS_DIR']}")
    logger.info(f"======================================================================")

    variant_root = Path(config["BENCH_ROOT"]) / "variant"
    if not variant_root.exists():
        logger.error(f"❌ Error: {variant_root} not found.")
        sys.exit(1)

    targets = [p for p in variant_root.glob("*/*") if p.is_dir()]
    if not targets:
        logger.warning("❌ No targets found.")
        sys.exit(1)

    logger.info(f"📋 Found {len(targets)} targets. Starting parallel execution...")

    # Group targets by project (parent dir name), then flatten in grouped order

    targets_sorted = sorted(targets, key=lambda p: p.parent.name)
    grouped_lists = [list(group) for _, group in groupby(targets_sorted, key=lambda p: p.parent.name)]
    interleaved_batches = zip_longest(*grouped_lists)
    ordered_targets = [t for batch in interleaved_batches for t in batch if t is not None]
    logger.info(f"🔄 Reordered targets using Round-Robin strategy (Count: {len(ordered_targets)})")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_recovery_agent, t, config) for t in ordered_targets]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Recovering"):
                tqdm.tqdm.write(str(future.result()))

    except KeyboardInterrupt:
        logger.warning("\n🛑 Stopped by user.")
        sys.exit(1)

    logger.info("🏁 Phase 3 Completed.")

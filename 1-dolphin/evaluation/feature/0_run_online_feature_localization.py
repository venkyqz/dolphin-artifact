import argparse
import shutil
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm
from util.feature_map import FEATURE_MAP, prepare_project_source

"""
This script automates the process of localizing features in various open-source
projects using a Multi-Agent System (SWE-Agent).
Optimization: Source code analysis is performed ONCE per project.
"""


FINAL_RESULT_NAME = "feature_implementation.json"


# ==============================================================================
# WORKER FUNCTION
# ==============================================================================
def localize_feature(source_dir, project_name, feature_name, feature_flag, config):
    """
    Runs the Multi-Agent System. Now simply copies the pre-computed DB.
    """
    task_id = f"{project_name}_{feature_name}"
    BASE_DIR = Path(config["BASE_DIR"])
    worker_workspace = BASE_DIR / "workspace" / "feature_localization" / task_id

    feature_output_dir = Path(config["FEATURE_DEFINITION_DIR"]) / project_name

    feature_output_file = feature_output_dir / f"{feature_name}.json"

    # preprocessed_source_code_dir = Path(config["SOURCE_PREPROCESSING_DIR"]) / project_name / "source_code"

    # preprocessed_source_metadata_file = Path(config["SOURCE_PREPROCESSING_DIR"]) / project_name / "source_metadata.json"

    if feature_output_file.exists():
        return f"⏭️  SKIPPING: {task_id} - Result exists."

    if not worker_workspace.exists():
        # Copy Source Code Repository
        shutil.copytree(source_dir, Path(worker_workspace / "repository"), dirs_exist_ok=True)

        # Copy Preprocessed Source Code
        # shutil.copytree(
        #     preprocessed_source_code_dir,
        #     Path(worker_workspace / "source_code"),
        #     dirs_exist_ok=True,
        # )

        # # Copy Preprocessed Source Metadata
        # shutil.copy2(
        #     preprocessed_source_metadata_file,
        #     Path(worker_workspace / "source_metadata.json"),
        # )

    log_path = Path(config["LOG_DIR"]) / f"{task_id}.log"
    traj_dir = Path(config["TRAJ_DIR"]) / task_id

    problem_file = worker_workspace / "problem_statement.md"
    problem_content = textwrap.dedent(
        f"""
        # Feature Localization Task

        Target Repo: /workspace
        Project: {project_name}
        Target Feature Name: {feature_name}
        Target Build Flag: `{feature_flag}`
        """
    ).strip()
    problem_file.write_text(problem_content)

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

    with open(log_path, "w") as log_file:
        try:
            subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=config["BASE_DIR"],
                check=False,
            )
        except Exception as e:
            return f"❌ ERROR: {task_id} - Execution failed: {str(e)}"

    generated_file = worker_workspace / FINAL_RESULT_NAME
    if generated_file.exists():
        feature_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(generated_file, feature_output_file)
        shutil.rmtree(worker_workspace, ignore_errors=True)
        return f"✅ SUCCESS: {task_id} -> {feature_output_file}"
    else:
        return f"❌ FAILED:  {task_id} -> No JSON generated. Check logs."


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Localization Runner")
    parser.add_argument("--config", default="config/feature/1_feature_localization.yaml")
    parser.add_argument("--oss_root", default="oss")
    parser.add_argument("-j", "--workers", type=int, default=4)
    args = parser.parse_args()

    # Base Paths
    BASE_DIR = Path.cwd()

    TEMP_SRC_DIR = BASE_DIR / "temp_sources"
    FEATURE_DEFINITION_DIR = BASE_DIR / "feature_definitions_source"
    LOG_DIR = BASE_DIR / "logs" / "feature_localization"
    TRAJ_DIR = BASE_DIR / "trajectories" / "feature_localization"

    SOURCE_PREPROCESSING_DIR = BASE_DIR / "source_code"

    for p in [
        TEMP_SRC_DIR,
        FEATURE_DEFINITION_DIR,
        LOG_DIR,
        TRAJ_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    config = {
        "BASE_DIR": str(BASE_DIR),
        "AGENT_CONFIG": args.config,
        "FEATURE_DEFINITION_DIR": str(FEATURE_DEFINITION_DIR),
        "LOG_DIR": str(LOG_DIR),
        "TRAJ_DIR": str(TRAJ_DIR),
        "SOURCE_PREPROCESSING_DIR": str(SOURCE_PREPROCESSING_DIR),
    }

    # 1. Prepare Projects & Databases (Sequential, Once per project)
    tasks = []
    oss_path = Path(args.oss_root)

    print("📦 Preparing Source Codes & String Databases...")
    for tar_file in oss_path.glob("*.tar.gz"):
        # This now ensures extraction + analysis happens ONCE
        project_name, source_dir = prepare_project_source(tar_file, TEMP_SRC_DIR)

        if not project_name:
            continue

        if project_name not in FEATURE_MAP:
            continue

        for feat_name, feat_flag in FEATURE_MAP[project_name]:
            tasks.append(
                {
                    "source_dir": source_dir,
                    "project_name": project_name,
                    "feature_name": feat_name,
                    "feature_flag": feat_flag,
                }
            )

    print(f"🚀 Starting Feature Localization on {len(tasks)} features with {args.workers} workers.")

    # 2. Execute Workers (Parallel)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                localize_feature,
                t["source_dir"],
                t["project_name"],
                t["feature_name"],
                t["feature_flag"],
                config,
            )
            for t in tasks
        ]

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            tqdm.tqdm.write(str(future.result()))

    print("🏁 Feature Localization Completed.")

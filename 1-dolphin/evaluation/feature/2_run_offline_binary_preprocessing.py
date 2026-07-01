import argparse
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import tqdm

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Import Feature Map
try:
    from evaluation.feature.util.feature_map import FEATURE_MAP
except ImportError:
    try:
        from util.feature_map import FEATURE_MAP
    except ImportError:
        print("❌ Error: Could not import FEATURE_MAP. Check python path.")
        sys.exit(1)

# Configure Console Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Manager")


# ==============================================================================
# WORKER LOGIC
# ==============================================================================


def preprocess_binary(target_path, config, task_log_file: Path):
    """
    Worker function to process a single binary using IDA Pro wrapper.
    """
    project_name = target_path.parent.name
    sha_id = target_path.name

    # 1. Setup Workspace
    BASE_DIR = Path(config["BASE_DIR"])
    worker_workspace = BASE_DIR / "workspace" / config["WORKDIR_NAME"] / sha_id
    worker_workspace.mkdir(parents=True, exist_ok=True)

    # Change CWD to worker workspace
    os.chdir(worker_workspace)

    with open(task_log_file, "w") as f:
        f.write(f"=== Log for {project_name}/{sha_id} ===\n")

    # 2. Prepare Binary
    bin_path = target_path / project_name
    if bin_path.exists():
        bin_source_path = bin_path
    else:
        try:
            bin_source_path = next(
                f for f in target_path.glob("*") if f.is_file() and os.access(f, os.X_OK) and f.suffix != ".sh"
            )
        except StopIteration:
            with open(task_log_file, "a") as f:
                f.write(f"[ERROR] No executable found in {target_path}\n")
            return False, worker_workspace

    dest_bin_path = worker_workspace / "target"
    if not dest_bin_path.exists():
        shutil.copy2(bin_source_path, dest_bin_path)

    # 3. Check Cache
    final_code_dir = worker_workspace / "decompiled_code"
    final_meta = worker_workspace / "decompiled_metadata.json"

    # Cache hit check
    if final_code_dir.exists() and final_meta.exists() and final_meta.stat().st_size > 10:
        # Quick check if dir is empty
        if any(final_code_dir.iterdir()):
            with open(task_log_file, "a") as f:
                f.write("\n[CACHE HIT] Skipping analysis.\n")
            return True, worker_workspace

    # 4. Prepare Output Directory
    # [FIX] Explicitly create and pass the subdirectory
    if final_code_dir.exists():
        shutil.rmtree(final_code_dir)
    final_code_dir.mkdir(parents=True, exist_ok=True)

    # Create EMPTY targets.json to trigger "Decompile ALL"
    targets_json_path = worker_workspace / "all_targets.json"
    with open(targets_json_path, "w") as f:
        json.dump([], f)

    ida_wrapper = config["IDA_WRAPPER"]

    # [FIX] Pass final_code_dir as the output directory to wrapper
    cmd = [
        ida_wrapper,
        str(dest_bin_path.absolute()),  # <bin_abspath>
        str(final_code_dir.absolute()),  # <out_dir_abspath> (The decompiled_code folder)
        str(targets_json_path.absolute()),  # <json_abspath>
    ]

    # 5. Run IDA Wrapper
    try:
        with open(task_log_file, "a") as f:
            f.write(f"\n>>> EXEC: {' '.join(cmd)}\n")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=1200,  # 20 mins timeout per binary
        )

        with open(task_log_file, "a") as f:
            if result.stdout:
                f.write(f"[STDOUT]\n{result.stdout}\n")
            if result.stderr:
                f.write(f"[STDERR]\n{result.stderr}\n")
            f.write(f"[EXIT CODE] {result.returncode}\n")

    except subprocess.TimeoutExpired:
        with open(task_log_file, "a") as f:
            f.write("\n[ERROR] Timeout expired.\n")
        return False, worker_workspace
    except Exception as e:
        with open(task_log_file, "a") as f:
            f.write(f"\n[ERROR] Exception: {e}\n")
        return False, worker_workspace

    # 6. Post-Processing
    # The wrapper generated 'decompile_index.json' INSIDE 'decompiled_code' directory.
    # We want to move it to root as 'decompiled_metadata.json' for easier access.

    generated_index = final_code_dir / "decompile_index.json"

    if not generated_index.exists():
        with open(task_log_file, "a") as f:
            f.write("\n[ERROR] decompile_index.json not found. Analysis likely failed.\n")
        return False, worker_workspace

    # Move index to workspace root and rename
    shutil.move(str(generated_index), str(final_meta))

    # Clean up dummy target json
    if targets_json_path.exists():
        os.remove(targets_json_path)

    # Clean up database files if any
    for db_file in worker_workspace.glob("*.i64"):
        os.remove(db_file)

    for call_graph in worker_workspace.glob("callgraph.json"):
        shutil.move(str(call_graph), str(worker_workspace / "callgraph.json"))

    # 7. Final Verification
    func_count = len(list(final_code_dir.glob("*.c")))
    with open(task_log_file, "a") as f:
        f.write(f"\n[SUCCESS] Preprocessing complete.\n")
        f.write(f"Functions dumped: {func_count} to {final_code_dir}\n")
        f.write(f"Metadata saved: {final_meta}\n")

    return True, worker_workspace


def run_preprocessing_task(target_dir, config):
    target_path = Path(target_dir)
    project_name = target_path.parent.name
    sha_id = target_path.name

    if project_name not in FEATURE_MAP:
        return None

    # Define task-specific log file
    log_dir = Path(config["LOG_DIR"])
    task_log_file = log_dir / f"{sha_id}.log"

    try:
        success, workspace_path = preprocess_binary(target_path, config, task_log_file)
    except Exception as e:
        return {"success": False, "status": f"❌ EXCEPTION: {sha_id} - {e}"}

    if not success:
        return {"success": False, "status": f"❌ FAIL: {sha_id}"}

    return {
        "success": True,
        "skipped": False,
        "status": f"✅ {sha_id}",
        "context": {
            "worker_workspace": str(workspace_path),
            "project_name": project_name,
            "sha_id": sha_id,
        },
    }


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Binary Preprocessing (IDA Pro)")
    parser.add_argument("workdir_name", help="Name of workspace (e.g. recovery_v1)")
    parser.add_argument("bench_root", help="Path to benchmark root")
    parser.add_argument("--wrapper", required=True, help="Absolute path to ida_decomp_binary shell script")
    parser.add_argument("-j", "--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    # Validate Wrapper
    wrapper_path = Path(args.wrapper).resolve()
    if not wrapper_path.exists() or not os.access(wrapper_path, os.X_OK):
        logger.error(f"Wrapper not found or not executable: {wrapper_path}")
        sys.exit(1)

    BASE_DIR = Path.cwd()
    config = {
        "WORKDIR_NAME": args.workdir_name,
        "BENCH_ROOT": args.bench_root,
        "IDA_WRAPPER": str(wrapper_path),
        "BASE_DIR": str(BASE_DIR),
        "LOG_DIR": str(BASE_DIR / f"logs/{args.workdir_name}"),
        "TRAJ_DIR": str(BASE_DIR / f"trajectories/{args.workdir_name}"),
    }

    for k in ["LOG_DIR", "TRAJ_DIR"]:
        Path(config[k]).mkdir(parents=True, exist_ok=True)

    variant_root = Path(config["BENCH_ROOT"]) / "variant"
    if not variant_root.exists():
        logger.error(f"Error: {variant_root} not found.")
        sys.exit(1)

    targets = [p for p in variant_root.glob("*/*") if p.is_dir()]

    logger.info(f"🚀 Starting IDA Preprocessing with {args.workers} workers.")
    logger.info(f"🔧 Wrapper: {config['IDA_WRAPPER']}")
    logger.info(f"📄 Logs: {config['LOG_DIR']}/<sha_id>.log")

    valid_contexts = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_preprocessing_task, t, config): t for t in targets}

        for future in tqdm.tqdm(as_completed(futures), total=len(targets), desc="Processing"):
            try:
                result = future.result()
                if not result:
                    continue

                if result["success"]:
                    valid_contexts.append(result["context"])
                else:
                    logger.error(result["status"])

            except Exception as e:
                logger.error(f"Main loop exception: {e}")

    output_summary = Path(config["TRAJ_DIR"]) / "stage1_summary.json"
    with open(output_summary, "w") as f:
        json.dump(valid_contexts, f, indent=2)

    logger.info(f"🏁 Done. Success rate: {len(valid_contexts)}/{len(targets)}")
    logger.info(f"📄 Summary saved to {output_summary}")

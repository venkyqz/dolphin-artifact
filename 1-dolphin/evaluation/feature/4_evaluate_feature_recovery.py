import argparse
import json
import os
from typing import Dict, List, Set, Tuple, Union

from tabulate import tabulate  # Requires: pip install tabulate


def load_json(filepath: str) -> dict:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def normalize_function_list(func_list: List) -> Set[Tuple[str, str]]:
    """
    Normalizes a list of [filename, function_name] or just function_name strings
    into a set of (filename, function_name) tuples for comparison.

    Handles two formats found in your examples:
    1. Ground Truth format: [ ["file.c", "func1"], ["file.c", "func2"] ]
    2. Recovered format (sometimes): [ "func1", "func2" ] (if filename is implicit or lost)
       OR the Impacted Files format:
       {
         "impacted_files": [
            { "filename": "A.c", "changes": { "added_functions": ["f1"] } }
         ]
       }
    """
    normalized = set()

    # Direct list of lists/tuples: [ ["file.c", "func"], ... ]
    if isinstance(func_list, list):
        for item in func_list:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                # Normalizing filename: remove leading paths, keep basename for safer comparison
                fname = os.path.basename(item[0])
                func = item[1]
                normalized.add((fname, func))
            elif isinstance(item, str):
                # If only function name is provided (less strict comparison)
                # We use a wildcard or empty string for filename if strict file checking isn't possible
                # But for this specific task, we assume the ground truth has files.
                # If the Source input has only strings, we might need a different strategy.
                # For now, let's assume valid tuple inputs or handle the specific Source format below.
                pass

    return normalized


def extract_functions_from_recovered(data: dict) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Extracts added and modified functions from the Recovered Feature JSON structure.

    Expected Recovered Structure (based on your input):
    {
      "impacted_files": [
        {
          "filename": "src/module.c",
          "changes": {
            "added_functions": ["funcA", "funcB"],
            "modified_functions": ["funcC"]
          }
        }
      ]
    }
    """
    added = set()
    modified = set()

    if "impacted_files" in data:
        for file_info in data["impacted_files"]:
            # Normalize path to basename for comparison
            raw_filename = file_info.get("filename", "")

            if not raw_filename:
                raw_filename = file_info.get("filepath", "")

            if not raw_filename.endswith((".c", ".cpp")):
                continue  # Skip non-source files

            fname = os.path.basename(raw_filename.split("/")[-1])

            changes = file_info.get("changes", {})

            for func in changes.get("added_functions", []):
                added.add((fname, func))

            for func in changes.get("modified_functions", []):
                modified.add((fname, func))

    return {"added": added, "modified": modified}


def extract_functions_from_ground_truth(data: dict) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Extracts added and modified functions from the Ground Truth JSON structure.

    Expected Ground Truth Structure:
    {
      "ground_truth_superset": {
        "added_functions": [ ["file.c", "funcA"], ... ],
        "modified_functions": [ ["file.c", "funcB"], ... ]
      }
    }
    """
    added = set()
    modified = set()

    superset = data.get("ground_truth_superset", {})

    # Added Functions
    raw_added = superset.get("added_functions", [])
    for item in raw_added:
        if len(item) >= 2:
            added.add((os.path.basename(item[0]), item[1]))

    # Modified Functions
    raw_modified = superset.get("modified_functions", [])
    for item in raw_modified:
        if len(item) >= 2:
            modified.add((os.path.basename(item[0]), item[1]))

    return {"added": added, "modified": modified}


def format_metrics_display(metrics: Dict) -> List[Union[str, float]]:
    """Formats P, R, F1, and Count display, using '-' if GT count is 0."""
    gt_count = metrics["gt_count"]
    if gt_count == 0:
        # If there's no ground truth data, display dashes for P, R, F1
        p, r, f1 = "-", "-", "-"
    else:
        p = f"{metrics['precision']:.2f}"
        r = f"{metrics['recall']:.2f}"
        f1 = f"{metrics['f1']:.2f}"

    count_display = f"({metrics['tp']}/{gt_count})"

    return [p, r, f1, count_display]


def calculate_metrics(ground_truth: Set, recovered: Set):
    tp = len(ground_truth.intersection(recovered))
    fp = len(recovered - ground_truth)
    fn = len(ground_truth - recovered)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- ADDED LINE: Calculate gt_count ---
    gt_count = len(ground_truth)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_count": gt_count,  # <--- ADDED KEY
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Feature Recovery vs Ground Truth")
    parser.add_argument("--gt", required=True, help="Path to ground_truth.json")
    parser.add_argument("--source", required=True, help="Path to feature_definitions_source directory")
    args = parser.parse_args()

    # 1. Load Ground Truth
    gt_data = load_json(args.gt)

    results_table = []

    # 2. Iterate through libraries and features in Ground Truth
    for lib_name, features in gt_data.items():
        for feat_name, feat_data in features.items():
            # Construct expected path for recovered file
            # Assumption: Structure is source_dir/lib_name/feature_name.json OR similar.
            # Adjust this path joining logic based on your actual file structure
            # Example based on typical structure: source/libtiff/ccitt/feature_implementation.json
            # Or simplified: source/libtiff_ccitt.json.
            # Let's assume a nested structure based on the prompt's context implicitly:
            # source_dir/libname/featurename.json or source_dir/libname_featurename.json

            # Searching for the file
            # Try 1: source/lib/feature.json
            rec_path = os.path.join(args.source, lib_name, f"{feat_name}.json")
            if not os.path.exists(rec_path):
                # Try 2: source/lib_feature.json
                rec_path = os.path.join(args.source, f"{lib_name}_{feat_name}.json")

            # If not found, skip or mark as 0
            if not os.path.exists(rec_path):
                results_table.append(
                    [
                        lib_name,
                        feat_name,
                        "File Not Found",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                    ]
                )
                continue

            # 3. Load Recovered Data
            rec_json = load_json(rec_path)

            # 4. Extract Sets
            gt_funcs = extract_functions_from_ground_truth(feat_data)
            rec_funcs = extract_functions_from_recovered(rec_json)

            # 5. Calculate Metrics
            added_metrics = calculate_metrics(gt_funcs["added"], rec_funcs["added"])
            mod_metrics = calculate_metrics(gt_funcs["modified"], rec_funcs["modified"])

            # Format the metrics using the new display function
            added_display = format_metrics_display(added_metrics)
            mod_display = format_metrics_display(mod_metrics)

            # 6. Format Row

            results_table.append([lib_name, feat_name] + added_display + mod_display)

            # Library | Feature | Type | P | R | F1 | Type | P | R | F1
            # results_table.append(
            #     [
            #         lib_name,
            #         feat_name,
            #         # Added Functions Stats
            #         f"{added_metrics['precision']:.2f}",
            #         f"{added_metrics['recall']:.2f}",
            #         f"{added_metrics['f1']:.2f}",
            #         f"({added_metrics['tp']}/{len(gt_funcs['added'])})",
            #         # Modified Functions Stats
            #         f"{mod_metrics['precision']:.2f}",
            #         f"{mod_metrics['recall']:.2f}",
            #         f"{mod_metrics['f1']:.2f}",
            #         f"({mod_metrics['tp']}/{len(gt_funcs['modified'])})",
            #     ]
            # )

    # 7. Print Table
    headers = [
        "Library",
        "Feature",
        "Add P",
        "Add R",
        "Add F1",
        "Add Count(TP/GT)",
        "Mod P",
        "Mod R",
        "Mod F1",
        "Mod Count(TP/GT)",
    ]

    print(tabulate(results_table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()

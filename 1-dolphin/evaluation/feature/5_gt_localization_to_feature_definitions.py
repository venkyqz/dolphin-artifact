import argparse
import json
import os
from collections import defaultdict


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def group_functions_by_file(func_list):
    """
    Converts list of [filename, funcname] into {filename: [funcnames]}
    """
    file_map = defaultdict(list)
    for item in func_list:
        if isinstance(item, list) and len(item) >= 2:
            filename = item[0]
            funcname = item[1]
            file_map[filename].append(funcname)
    return file_map


def convert_feature(lib_name, feat_name, feat_data):
    """
    Converts a single feature's GT data into the target JSON structure.
    """
    gt_superset = feat_data.get("ground_truth_superset", {})

    # 1. Group functions by file
    added_by_file = group_functions_by_file(gt_superset.get("added_functions", []))
    modified_by_file = group_functions_by_file(gt_superset.get("modified_functions", []))

    # 2. Collect all impacted files
    all_files = set(added_by_file.keys()) | set(modified_by_file.keys())

    impacted_files = []

    for filename in sorted(list(all_files)):
        added_funcs = sorted(added_by_file.get(filename, []))
        mod_funcs = sorted(modified_by_file.get(filename, []))

        # Heuristic: If a file has added functions but NO modified functions (and likely many added),
        # it is often a new module. This is a best-guess for the benchmark.
        # Strict logic: If it appears in 'added_files' list from build trace it is new.
        # But we only have binary diff here.
        # Simple Logic: If >0 added and 0 modified, treat as new module for benchmark purpose,
        # or default to false to be safe. Let's default to False unless ALL funcs in file are new.
        # (Since we don't know the total funcs in file easily here, we'll mark is_new_module=False usually
        # unless you have specific metadata).
        # IMPROVEMENT: Let's assume False for binary diff based GT to avoid false positives.
        is_new = False
        if len(added_funcs) > 0 and len(mod_funcs) == 0:
            # Weak heuristic: heavily populated new file
            is_new = True

        file_entry = {
            "filepath": filename,  # Note: GT usually has relative paths like "tif_fax3.c" or "src/foo.c"
            "is_new_module": is_new,
            "reasoning": "Ground Truth: Function difference detected in binary analysis.",
            "changes": {
                "added_functions": added_funcs,
                "modified_functions": mod_funcs,
                "removed_functions": [],  # Binary diff usually doesn't track removed easily in this context
            },
        }
        impacted_files.append(file_entry)

    # 3. Construct Final Object
    return {
        "feature_flag": feat_data.get("feature_flag", ""),
        "build_macros": ["<OMITTED_IN_GT>"],  # Binary analysis doesn't know macro names
        "summary": {
            "intent": f"Ground Truth for {lib_name} feature: {feat_name}",
            "structural_change": "Automatically generated from binary diff analysis.",
        },
        "impacted_files": impacted_files,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Ground Truth JSON to Feature Definitions Structure")
    parser.add_argument("--input", default="ground_truth_localization_benchmark.json", help="Input Ground Truth JSON")
    parser.add_argument("--output_dir", default="gt_feature_definitions", help="Output directory for converted JSONs")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    data = load_json(args.input)

    print(f"Processing {len(data)} libraries...")

    for lib_name, features in data.items():
        lib_dir = os.path.join(args.output_dir, lib_name)
        ensure_dir(lib_dir)

        for feat_name, feat_data in features.items():
            converted_data = convert_feature(lib_name, feat_name, feat_data)

            output_path = os.path.join(lib_dir, f"{feat_name}.json")

            with open(output_path, "w") as f:
                json.dump(converted_data, f, indent=2)

    print(f"Conversion complete. Files saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

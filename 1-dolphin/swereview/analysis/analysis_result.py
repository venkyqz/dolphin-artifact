import json

import loguru
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from path import Path

from swereview.model.result import SWEResults

log = loguru.logger


def analyse_all_result(path: str):
    path = Path(path)
    # 找到所有的results.json文件，并获取对应的df，合并所有的resolved
    all_results = []

    # Recursively find all results.json files
    results_files = path.walkfiles("results.json")

    for result_file in results_files:
        print(result_file)
        # continue
        try:
            # Load each results file and convert to DataFrame
            df = load_swe_result(result_file)
            all_results.append(df)
        except Exception as e:
            log.error(f"Error processing {result_file}: {e}")

    if not all_results:
        log.warning("No results.json files found")
        return None

    # Concatenate all DataFrames
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df = combined_df[combined_df["resolved"]]

    # Remove duplicate IDs if any
    combined_df = combined_df.drop_duplicates(subset=["id"])

    # Analyze the combined results
    stat(combined_df)
    # show_bar(combined_df)
    # show_failed(combined_df)

    return combined_df


def analyse_result(path: str):
    df = load_swe_result(path)
    stat(df)
    show_bar(df)
    show_failed(df)


def model_to_dataframe(test_results: SWEResults) -> pd.DataFrame:
    # Convert model to dict
    data_dict = test_results.dict()

    # Get all unique IDs across all lists
    all_ids = set()
    for values in data_dict.values():
        all_ids.update(values)

    # Create a DataFrame with unique IDs
    df = pd.DataFrame({"id": sorted(list(all_ids))})

    # Add boolean columns for each category
    for category, values in data_dict.items():
        df[category] = df["id"].isin(values)

    return df


def load_swe_result(path: str):
    if Path(path).is_dir():
        file = path + "/results/results.json"
    else:
        file = path

    with open(file) as fd:
        data = json.load(fd)
        res = SWEResults(**data)

    return model_to_dataframe(res)


def stat(df):
    log.info(f"result corresponding to {len(df)} tasks")
    # print(df.head())


def show_bar(df):
    # 计算每一列的True/False比例
    columns = [
        "applied",
        "generated",
        "install_fail",
        "no_apply",
        "no_generation",
        "reset_failed",
        "resolved",
        "test_errored",
        "test_timeout",
        "with_logs",
    ]

    # 创建结果字典
    results = {}
    for col in columns:
        value_counts = df[col].value_counts(normalize=True)
        results[col] = {
            "True": value_counts.get(True, 0),
            "False": value_counts.get(False, 0),
        }

    # 转换为DataFrame便于绘图
    plot_df = pd.DataFrame(results).T

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(columns))
    width = 0.35

    # 绘制True和False的柱状图
    rects1 = ax.bar(x - width / 2, plot_df["True"], width, label="True")
    rects2 = ax.bar(x + width / 2, plot_df["False"], width, label="False")

    # 在柱子上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2%}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3点垂直偏移
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.title("True/False Proportion for Each Column")
    plt.xlabel("Columns")
    plt.ylabel("Proportion")
    plt.xticks(x, columns, rotation=45, ha="right")
    plt.legend(title="Value")
    plt.tight_layout()
    plt.show()


def show_failed(df):
    df_failed = df[not df["resolved"]]
    print(len(df_failed))
    print(df_failed)


if __name__ == "__main__":
    df = analyse_all_result("/path/to/swe-agent-log/evaluation/test")
    df.to_csv("results.test.csv")

    df = analyse_all_result("/path/to/swe-agent-log/evaluation/lite")
    df.to_csv("results.lite.csv")

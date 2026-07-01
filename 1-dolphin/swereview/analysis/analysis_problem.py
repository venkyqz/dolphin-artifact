import loguru
import pandas as pd
import typer

log = loguru.logger


def search_with_id(id: str, df) -> pd.DataFrame:
    # Search for 'id' in column `instance_id` and create a new DataFrame
    return df[df["instance_id"].astype(str).str.contains(id, na=False)]


def search_with(s, df):
    # Search for `s` in all columns and create a new DataFrame
    return df[df.astype(str).apply(lambda x: x.str.contains(s, na=False)).any(axis=1)]


def main(name: str, dataset: str = typer.Option("data/swe-lite.test.csv", "-d", "--dataset")):
    df = pd.read_csv(dataset)
    df_filtered = search_with_id(name, df)

    # Display the number of matching rows
    log.info(f"Found {len(df_filtered)} rows containing '{name}'")

    if len(df_filtered) == 0:
        log.warning(f"No rows found for '{name}'")
        return

    # Save the filtered DataFrame to a new CSV file
    log.info(f"Saving filtered data to data/problem/{name}.json")
    df_filtered.to_json(f"data/problem/{name}.json", index=False, indent=2, orient="records")


if __name__ == "__main__":
    typer.run(main)

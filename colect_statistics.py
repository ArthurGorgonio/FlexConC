import argparse
from glob import glob
from os import makedirs

import pandas as pd


def create_table(
    metric: list,
    name: str,
    percentage: bool = True,
) -> str:
    """Calculate the average ± standard deviation and quartiles."""

    if percentage:
        metric = [m * 100 for m in metric]

    msg = (
        f"*{name}: {round(metric[0], 2)} ± {round(metric[1], 2)}*\n"
        f"Q1: {round(metric[2], 2)}\n"
        f"Q2: {round(metric[3], 2)}\n"
        f"Q3: {round(metric[4], 2)}\n"
        f"Q4: {round(metric[5], 2)}\n\n"
    )

    return msg


def generate_basic_metrics(
    file: str,
    df: pd.DataFrame,
    verbose: bool = False,
) -> list:
    """Print the metrics

    Args:
        file (str): Filename.
        df (pd.DataFrame): Dataframe with all metrics.
    """
    interesting_metrics = [1, 2, 4, 5, 6, 7]
    acc, fscore, kappa, time = [], [], [], []
    ensemble = []

    for i in interesting_metrics:
        metrics = (df.describe().iloc[i, 3:]).tolist()
        acc.append(metrics[0])
        fscore.append(metrics[1])
        kappa.append(metrics[2])
        time.append(metrics[3])
        ensemble.append((df.describe().iloc[i, 0]).tolist())

    if verbose:
        print(f"Método: {file.split('_')[1]}")
        print(f"Reator: {file.split('_')[2]}")
        print(f"Strategy: {file.split('_')[3]}")
        print(f"Detecções: {sum(df.iloc[:, 2])}")
        print(
            f'{create_table(acc, "ACC"):>20}'
            + f'{create_table(fscore, "F1"):>20}'
            + f'{create_table(kappa, "Kappa", False):>20}'
            + f'{create_table(time, "Time", False):>20}'
            + f'{create_table(ensemble, "Comitê", False):>20}'
        )

    msg = [
        sum(df.iloc[:, 2]),
    ]
    msg += generate_metrics(acc)
    msg += generate_metrics(fscore)
    msg += generate_metrics(kappa)
    msg += generate_metrics(time)
    msg += generate_metrics(ensemble)

    return msg


def generate_metrics(metric: list) -> str:
    return [
        round(metric[0], 4),
        round(metric[2], 4),
        round(metric[3], 4),
        round(metric[4], 4),
        round(metric[5], 4),
    ]


def get_dataset_names(data_path: str) -> set:
    """get dataset names to filter"""
    base_files = glob(f"{data_path}/*.txt")
    dataset_names = {data.split("_")[-1].split(".")[0] for data in base_files}

    return dataset_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_path",
        help="Path to results files",
    )
    parser.add_argument(
        "--all",
        nargs="+",
        type=int,
        default=[0, 4, 5, 6, 7],
        help="Which columns should be saved, based on data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    names = ["size", "acc", "f1", "kappa", "time", "csv"]

    for n in names:
        makedirs(
            f"{args.results_path}/{n}",
            exist_ok=True,
        )

    print(args)
    datasets = get_dataset_names(args.results_path)

    for dataset in datasets:
        files = glob(f"{args.results_path}/*{dataset}.txt")
        files.sort()

        data = [pd.DataFrame() for _ in range(len(args.all))]
        metrics = pd.DataFrame()

        for file in files:
            print("*" * 30)
            print(f"Dataset: {dataset}")
            df = pd.read_csv(file, delimiter=r"\s+", skiprows=16, header=None)
            col_name = file.split("/")[-1].split('_')[0]
            metrics[col_name] = generate_basic_metrics(
                file.split("/")[-1], df, args.verbose
            )

            for metric_df, col in zip(data, args.all):
                metric_df[col_name] = df.iloc[:, col]

        metrics.to_csv(
            f"{args.results_path}/csv/{dataset}.csv",
            sep="\t",
            index=None,
        )

        for metric_df, name in zip(data, names):
            metric_df.to_csv(
                f"{args.results_path}/{name}/results_{dataset}.dat",
                sep="\t",
                index=None,
            )

import argparse
import operator
from glob import glob
from os.path import abspath
from subprocess import CalledProcessError, run

import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv


def get_dataset_names(data_path: str) -> set:
    """get dataset names to filter"""
    base_files = glob(f"{data_path}/*.dat")

    return sorted(base_files)


def generate_cd_diagrm(
    dt_name: str,
    pos: list[int],
    lbs: list[str],
    plot_name: str,
    plot_path: str,
    decreasing: bool = False,
):
    cmd = [
        "Rscript",
        "--vanilla",
        "/home/arthur/workspace/cd-plots/cdplot/mainPlot.R",
        dt_name,
        "--location",
        abspath(plot_path) + "/",
        "--head",
        "--cd",
        "--only",
        "".join(str(pos)[1:-1].split(" ")),
        "--col",
        "".join(str(lbs)[1:-1].replace("'", "").split(" ")),
        "--suffix",
        plot_name,
    ]

    if decreasing:
        cmd.insert(7, "--decreasing")

    print(f"O comando chamado foi:\n{cmd}")
    try:
        run(cmd, check=True, shell=True, capture_output=True)
    except CalledProcessError as ex:
        print(f"The {dt_name} not run. Probally all columns are equal!\n")


def create_plot(
    df: DataFrame,
    labels: list[str],
    colors: list[str],
    title: str,
    plot_name: str,
    plot_path: str,
    resize: bool = False,
):
    if resize:
        plt.tight_layout()
        plt.figure(figsize=(12.8, 4.8))
    else:
        plt.figure(figsize=(6.4, 4.8))

    p = plt.boxplot(df, labels=labels, patch_artist=True)

    for patch, color in zip(p["boxes"], colors):
        patch.set_facecolor(color)

    dir_name = plot_path.split("/")

    if "acc" in dir_name or "f1" in dir_name:
        plt.ylim(0, 1)
    elif "kappa" in dir_name:
        plt.ylim(-1, 1)
    elif "size" in dir_name:
        plt.ylim(0, 10)
    plt.title(f"{title}")

    plt.savefig(f"{plot_path}/{plot_name.lower()}.png")
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="Path to results files")
    parser.add_argument(
        "--decrease",
        "-d",
        action="store_true",
        help="Is the metric is best when resuls are lowers? (Default 1)"
        "Pass 0 to False. Any value to True.",
    )
    args = parser.parse_args()

    labels = [
        "std-F-E-D",
        "std-F-E-S",
        "std-N-E-D",
        "std-N-E-S",
        "F-E-D",
        "F-E-S",
        # "F-P-D",
        # "F-P-S",
        # "F-V-D",
        # "F-V-S",
        "N-E-D",
        "N-E-S",
        # "N-P-D",
        # "N-P-S",
        # "N-V-D",
        # "N-V-S",
        # "S-E-D",
        # "S-E-S",
        # "S-P-D",
        # "S-P-S",
        # "S-V-D",
        # "S-V-S",
    ]

    cols = [
        "#ffc533ff",  # "std-F-E-D"
        "#ffc533ff",  # "std-F-E-S"
        "#D579FFff",  # "std-N-E-D"
        "#D579FFff",  # "std-N-E-S"
        "#ffc53399",  # "F-E-D"
        "#ffc53399",  # "F-E-S"
        # "#EAB76099",  # "F-P-D"
        # "#EAB76099",  # "F-P-S"
        # "#ffee93ff",  # "F-V-D"
        # "#ffee93ff",  # "F-V-S"
        "#D579FFaa",  # "N-E-D"
        "#D579FFaa",  # "N-E-S"
        # "#ddb0dd99",  # "N-P-D"
        # "#ddb0dd99",  # "N-P-S"
        # "#C9A0DBaa",  # "N-V-D"
        # "#C9A0DBaa",  # "N-V-S"
        # "#19ffb2cc",  # "S-E-D"
        # "#19ffb2cc",  # "S-E-S"
        # "#ADF7B699",  # "S-P-D"
        # "#ADF7B699",  # "S-P-S"
        # "#ABF5A6ff",  # "S-V-D"
        # "#ABF5A6ff",  # "S-V-S"
    ]
    datasets = get_dataset_names(args.files)


for data in datasets:
    df = read_csv(data, sep="\t")
    data_name = data.split(".")[0].split("_")[-1]
    create_plot(df, labels, cols, data_name, data_name, args.files, True)
    generate_cd_diagrm(
        data,
        list(range(1, len(labels) + 1)),
        labels,
        data_name,
        args.files,
        args.decrease,
    )

    # Detector: Fixed Threshold
    _lst = [0, 1, 4, 5]
    print(
        f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
        f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
        "\n\n",
    )
    create_plot(
        df.iloc[:, _lst],
        list(operator.itemgetter(*_lst)(labels)),
        list(operator.itemgetter(*_lst)(cols)),
        data_name + " FixedThreshold",
        data_name + "_fixedthreshold",
        args.files,
    )
    generate_cd_diagrm(
        data,
        [1, 2, 5, 6],
        list(operator.itemgetter(*_lst)(labels)),
        "fixedthreshold",
        args.files,
        args.decrease,
    )

    # Detector: Normal
    _lst = [2, 3, 6, 7]
    print(
        f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
        f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
        "\n\n",
    )
    create_plot(
        df.iloc[:, _lst],
        list(operator.itemgetter(*_lst)(labels)),
        list(operator.itemgetter(*_lst)(cols)),
        data_name + " Normal",
        data_name + "_normal",
        args.files,
    )
    generate_cd_diagrm(
        data,
        [3, 4, 7, 8],
        list(operator.itemgetter(*_lst)(labels)),
        "normal",
        args.files,
        args.decrease,
    )

    # # Detector: Statistical
    # _lst = list(range(12, 18))
    # print(
    #     f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
    #     f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
    #     "\n\n",
    # )
    # create_plot(
    #     df.iloc[:, _lst],
    #     list(operator.itemgetter(*_lst)(labels)),
    #     list(operator.itemgetter(*_lst)(cols)),
    #     data_name + " Statistical",
    #     data_name + "_statistical",
    #     args.files,
    # )
    # generate_cd_diagrm(
    #     data,
    #     list(range(13, 19)),
    #     list(operator.itemgetter(*_lst)(labels)),
    #     "statistical",
    #     args.files,
    #     args.decrease,
    # )

    # # Reactor: Exchange
    # _lst = [0, 1, 6, 7, 12, 13]
    # print(
    #     f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
    #     f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
    #     "\n\n",
    # )
    # create_plot(
    #     df.iloc[:, _lst],
    #     list(operator.itemgetter(*_lst)(labels)),
    #     list(operator.itemgetter(*_lst)(cols)),
    #     data_name + " Exchange",
    #     data_name + "_exchange",
    #     args.files,
    # )
    # generate_cd_diagrm(
    #     data,
    #     [1, 2, 7, 8, 13, 14],
    #     list(operator.itemgetter(*_lst)(labels)),
    #     "exchange",
    #     args.files,
    #     args.decrease,
    # )

    # # Reactor: Pareto
    # _lst = [2, 3, 8, 9, 14, 15]
    # print(
    #     f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
    #     f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
    #     "\n\n",
    # )
    # create_plot(
    #     df.iloc[:, _lst],
    #     list(operator.itemgetter(*_lst)(labels)),
    #     list(operator.itemgetter(*_lst)(cols)),
    #     data_name + " Pareto",
    #     data_name + "_pareto",
    #     args.files,
    # )
    # generate_cd_diagrm(
    #     data,
    #     [3, 4, 9, 10, 15, 16],
    #     list(operator.itemgetter(*_lst)(labels)),
    #     "pareto",
    #     args.files,
    #     args.decrease,
    # )

    # # Reactor: Volatile Exchange
    # _lst = [4, 5, 10, 11, 16, 17]
    # print(
    #     f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
    #     f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
    #     "\n\n",
    # )
    # create_plot(
    #     df.iloc[:, _lst],
    #     list(operator.itemgetter(*_lst)(labels)),
    #     list(operator.itemgetter(*_lst)(cols)),
    #     data_name + " VolatileExchange",
    #     data_name + "_volatile",
    #     args.files,
    # )
    # generate_cd_diagrm(
    #     data,
    #     [5, 6, 11, 12, 17, 18],
    #     list(operator.itemgetter(*_lst)(labels)),
    #     "volatile",
    #     args.files,
    #     args.decrease,
    # )

    # Strategy: Simple
    _lst = list(range(1, 8, 2))
    print(
        f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
        f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
        "\n\n",
    )
    create_plot(
        df.iloc[:, _lst],
        list(operator.itemgetter(*_lst)(labels)),
        list(operator.itemgetter(*_lst)(cols)),
        data_name + " simple",
        data_name + "_simple",
        args.files,
    )
    generate_cd_diagrm(
        data,
        list(range(2, 9, 2)),
        list(operator.itemgetter(*_lst)(labels)),
        "simple",
        args.files,
        args.decrease,
    )

    # Strategy: Drift
    _lst = list(range(0, 8, 2))
    print(
        f"\nLabels: {list(operator.itemgetter(*_lst)(labels))}",
        f"\nColors: {list(operator.itemgetter(*_lst)(cols))}",
        "\n\n",
    )
    create_plot(
        df.iloc[:, _lst],
        list(operator.itemgetter(*_lst)(labels)),
        list(operator.itemgetter(*_lst)(cols)),
        data_name + " drift",
        data_name + "_drift",
        args.files,
    )
    generate_cd_diagrm(
        data,
        list(range(1, 8, 2)),
        list(operator.itemgetter(*_lst)(labels)),
        "drift",
        args.files,
        args.decrease,
    )

    # aggregated = [read_csv(data, sep="\t") for data in datasets]
    # create_plot(
    #     aggregated,
    #     labels,
    #     cols,
    #     "general",
    #     "general",
    #     args.files,
    # )

    # create_plot(
    #     aggregated.iloc[:, [0, 1, 4, 5, 8, 9]],
    #     list(operator.itemgetter(*[0, 1, 4, 5, 8, 9])(labels)),
    #     cols2,
    #     "general Exchange",
    #     "general_exchange",
    #     args.files,
    # )

    # create_plot(
    #     aggregated.iloc[:, [2, 3, 6, 7, 10, 11]],
    #     list(operator.itemgetter(*[2, 3, 6, 7, 10, 11])(labels)),
    #     cols2,
    #     "general Pareto",
    #     "general_pareto",
    #     args.files,
    # )

    # create_plot(
    #     aggregated.iloc[:, [1, 3, 5, 7, 9, 11]],
    #     list(operator.itemgetter(*[1, 3, 5, 7, 9, 11])(labels)),
    #     cols2,
    #     "general simple",
    #     "general_simple",
    #     args.files,
    # )

    # create_plot(
    #     aggregated.iloc[:, [0, 2, 4, 6, 8, 10]],
    #     list(operator.itemgetter(*[0, 2, 4, 6, 8, 10])(labels)),
    #     cols2,
    #     "general drift",
    #     "general_drift",
    #     args.files,
    # )

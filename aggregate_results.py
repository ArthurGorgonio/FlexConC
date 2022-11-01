import glob

from numpy import array, where
from pandas import DataFrame, read_csv

# base variables
base_path = "results/detailed/"
base_name = [
    "DyDaSL N - ",
    "DyDaSL FT - ",
    "DyDaSL W - ",
    "DyDaSL H - ",
    "DyDaSL C - ",
]
base_name.sort()
base_batch = [100, 250, 500, 750, 1000, 2500, 5000]
drift_names = [
    "Drift – N",
    "Drift – FT",
    "Drift – W",
    "Drift – H",
    "Drift – C",
]
drift_names.sort()
header = [
    "CPSSDS",
    "Hinkley",
    "DyDaSL N",
    "DyDaSL FT",
    "DyDaSL W",
    "Mean C",
    "Mean H",
    "Mean N",
    "Mean FT",
    "Mean W",
]

# new base variables
base_path = "./"
base_batch = [500]
base_name = [
    "DyDaSL N - ",
    "DyDaSL FT - ",
    "DyDaSL S - ",
]
drift_names = [
    "Drift – N",
    "Drift – FT",
    "Drift – S",
]
base_name.sort()
drift_names.sort()

files = glob.glob(base_path + "DyDaSL*.txt")

data_name = [i.split("_")[-2].split(".")[0] for i in files]
datasets = list(set(data_name))
print(datasets)
header = [
    "DyDaSL FT",
    "Mean FT",
    "DyDaSL N",
    "Mean N",
    "DyDaSL S",
    "Mean S",
]


def save_data(path: str, values: list, column_names: list):
    df = DataFrame(values).transpose()
    df.columns = column_names
    df.to_csv(path, sep="\t")


def cumulative_sum(path: str, values: list, column_names: list):
    data = DataFrame(values).transpose()
    data.columns = column_names
    # data["Drift – C"] = where(data["Drift – C"], 1, 0)
    # data["Drift – H"] = where(data["Drift – H"], 1, 0)
    data["Drift – FT"] = where(data["Drift – FT"], 1, 0)
    data["Drift – N"] = where(data["Drift – N"], 1, 0)
    data["Drift – S"] = where(data["Drift – S"], 1, 0)
    # data["C Frequency"] = data["Drift – C"].cumsum() / (
    #     sum(data["Drift – C"]) if sum(data["Drift – C"]) > 0 else 1
    # )
    # data["H Frequency"] = data["Drift – H"].cumsum() / (
    #     sum(data["Drift – H"]) if sum(data["Drift – H"]) > 0 else 1
    # )
    data["FT Frequency"] = data["Drift – FT"].cumsum() / (
        sum(data["Drift – FT"]) if sum(data["Drift – FT"]) > 0 else 1
    )
    data["N Frequency"] = data["Drift – N"].cumsum() / (
        sum(data["Drift – N"]) if sum(data["Drift – N"]) > 0 else 1
    )
    data["S Frequency"] = data["Drift – S"].cumsum() / (
        sum(data["Drift – S"]) if sum(data["Drift – S"]) > 0 else 1
    )
    data.to_csv(path, sep="\t")


approach_name = [i + str(j) for j in base_batch for i in base_name]
step = len(base_name)

for pattern in datasets:
    print(f"Pattern procurado: {pattern}")
    acc_mean = []
    f1_mean = []
    kappa_mean = []
    drift_count = []
    files = glob.glob(base_path + "*" + pattern + "*.txt")
    files.sort()
    # data_name = [i.split("/")[-1].split("_")[0][6:] for i in files]
    data_name = ["-".join(i.split("/")[-1].split("_")[0:2]) for i in files]

    for batch in base_batch:
        files = glob.glob(
            base_path + "*" + pattern + "*_" + str(batch) + ".txt"
        )
        files.sort()
        print("\n", files)
        acc_batch = [[0] for _ in range(len(base_name) * 2)]
        f1_batch = [[0] for _ in range(len(base_name) * 2)]
        kappa_batch = [[0] for _ in range(len(base_name) * 2)]
        drift_batch = [[0] for _ in range(len(base_name))]

        for i, file_name in zip(range(0, len(files) * 2, 2), files):
            print(i, "\tFile: ", file_name)
            # TODO: alterar para 10
            data = read_csv(file_name, sep=r"\s+", skiprows=10)
            data.fillna(0, inplace=True)
            acc_mean.append(round(data.describe().iloc[1, 3], 4))
            f1_mean.append(round(data.describe().iloc[1, 4], 4))
            kappa_mean.append(round(data.describe().iloc[1, 5], 4))
            drift_count.append(sum(data.iloc[:, 2]))
            acc = data.iloc[:, 4]
            f1 = data.iloc[:, 5]
            kappa = data.iloc[:, 6]
            drift = data.iloc[:, 2]
            acc_batch[i] = array(acc)
            f1_batch[i] = array(f1)
            kappa_batch[i] = array(kappa)
            drift_batch[i // 2] = array(drift)
            acc_batch[i + 1] = array(round(acc.expanding().mean(), 4))
            f1_batch[i + 1] = array(round(f1.expanding().mean(), 4))
            kappa_batch[i + 1] = array(round(kappa.expanding().mean(), 4))
        save_data(
            "plots/acc/" + pattern + "_acc-" + str(batch) + ".csv",
            acc_batch,
            header,
        )
        save_data(
            "plots/f1/" + pattern + "_f1-" + str(batch) + ".csv",
            f1_batch,
            header,
        )
        save_data(
            "plots/kappa/" + pattern + "_kappa-" + str(batch) + ".csv",
            kappa_batch,
            header,
        )
        cumulative_sum(
            "plots/drift/" + pattern + "_drift-" + str(batch) + ".csv",
            drift_batch,
            drift_names,
        )
    print(
        "Acc: ",
        len(acc_mean),
        "\nF1S: ",
        len(f1_mean),
        "\nKap: ",
        len(kappa_mean),
        "\nDri: ",
        len(drift_count),
        "\nNam: ",
        len(data_name),
    )
    DataFrame(
        zip(data_name, acc_mean, f1_mean, kappa_mean, drift_count),
        columns=["Data", "Acc", "F1", "Kappa", "Drifts"],
        index=approach_name,
    ).to_csv("plots/medias" + pattern + ".csv")

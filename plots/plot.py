import glob
import os
from pathlib import Path

files = glob.glob("[!dri]*/*.csv")
files.sort()

metric_names = list(set([i.split("/")[0] for i in files]))
metric_names.sort()

drift = glob.glob("drift/*.csv")
drift.sort()

titles = ["Acurácia Média", "F-Score Médio", "Kappa Médio"]
batches = ["100", "250", "500", "750", "1000", "2500", "5000"]
batches = ["500"]
batches.sort()

plot_path = list(
    set([i.split("/")[1].split("_")[0].lower() for i in files])
) * len(set(batches))
plot_path.sort()
batches = batches * len(set(plot_path))

for i in set(plot_path):
    Path(i).mkdir(parents=True, exist_ok=True)

for metric, title in zip(metric_names, titles):
    output_data = [data for data in files if metric in data]

    for data, dots, batch, path in zip(output_data, drift, batches, plot_path):
        plot_name = title + " - Batches " + batch
        fig_name = path + "/" + plot_name
        print("Data: ", data, "\tDrift: ", dots, "\nFigName: ", fig_name)
        os.system(
            "gnuplot -c plot.gnuplot '{}' '{}' '{}' '{}' '{}' '{}'".format(
                data,
                dots,
                plot_name + "Best Fit",
                fig_name + "best-fit.png",
                0,
                1,
            )
        )
        os.system(
            "gnuplot -c plot.gnuplot '{}' '{}' '{}' '{}' '{}' '{}'".format(
                data,
                dots,
                plot_name + "Best Fit - Não Agregado",
                fig_name + "not-aggregated-best-fit.png",
                0,
                2,
            )
        )

        if "kappa" not in title.lower():
            os.system(
                "gnuplot -c plot.gnuplot '{}' '{}' '{}' '{}' '{}' '{}'".format(
                    data, dots, plot_name, fig_name + ".png", 1, 1
                )
            )
        else:
            os.system(
                "gnuplot -c plot.gnuplot '{}' '{}' '{}' '{}' '{}' '{}'".format(
                    data, dots, plot_name, fig_name + ".png", 2, 1
                )
            )
        plot_name = "Frequência de Drifts - Batches " + batch
        fig_name = "frequencia/" + path + plot_name.lower().strip()
        os.system(
            "gnuplot -c plot.gnuplot '{}' '{}' '{}' '{}' '{}' '{}'".format(
                dots, dots, plot_name, fig_name + ".png", 1, 3
            )
        )

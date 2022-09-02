import warnings

from glob import glob
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score as kappa,
    f1_score,
)
from skmultiflow.data import DataStream

from src.core.core import Core
from src.detection.fixed_threshold import FixedThreshold
from src.reaction.exchange import Exchange
from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon
from src.utils import Log

dydasl = Core(Ensemble, FixedThreshold, Exchange)
dydasl.configure_params(SelfFlexCon)
dydasl.add_metrics('acc', accuracy_score)
dydasl.add_metrics('f1', f1_score)
dydasl.add_metrics('kappa', kappa)
# datasets = glob('datasets/*.csv')
# datasets.sort()
datasets = [
    'Connect-4.csv',
    'Electricity.csv',
    'Fars.csv',
    'ForestCover.csv',
    'GEARS2C2D.csv',
    'Poker.csv',
    'Shuttle.csv',
    'UG2C3D.csv',
]

for dataset in datasets:
    dataframe = pd.read_csv(
        dataset if 'datasets/' in dataset else 'datasets/' + dataset
    )
    Log().filename = dataset.split('.')[0].split('/')[-1]
    # depende do dataset
    dim = dataframe.shape
    array = dataframe.values
    instances = array[: dim[0], : dim[1] - 1]
    target = array[: dim[0], dim[1] - 1]
    class_set = np.unique(target)
    class_count = np.unique(target).shape[0]
    stream = DataStream(
        instances,
        target,
        target_idx=-1,
        n_targets=class_count,
        cat_features=None,  # Categorical features?
        name=None,
        allow_nan=True,
    )
    dydasl.add_metrics("acc", accuracy_score)
    dydasl.add_metrics("f1", f1_score)
    dydasl.add_metrics("kappa", kappa)
    Log().write_archive_header()
    dydasl.run(stream)

# print(
#     f"ACC: {round(accuracy_score(y_true, y_pred), 4)}%\n"
#     f'F1-Score: {round(f1_score(y_true, y_pred, average="macro"), 4)}%\n'
#     f"Motivo da finalização: {comite.ensemble[0].termination_condition_}\n"
#     f"Valor do teste estatístico é de {alpha}, significante? {alpha <= 0.05}"
# )

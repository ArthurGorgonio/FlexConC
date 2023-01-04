from numpy import ndarray

from src.reaction.interfaces.reactor import Reactor
from src.ssl.ensemble import Ensemble


class Pareto(Reactor):
    """Reator de drift baseado na fronteira de Pareto.

    Parameters
    ----------
    params : Dict[str, Any]
        parâmetros de configuração do reator.

        - minimization : bool
            - indica se será utilizado a minimização ou maximização na
            otimização de Pareto. Default False.
    """

    def __init__(self, **params) -> None:
        self.minimization = params.get("minimization", False)
        self.computation = params.get(
            "pareto_strategy", "classifier_classifier"
        )
        self.q_measure_params = params.get(
            "q_measure", {"absolute": True, "average": False}
        )

    def react(
        self,
        ensemble: Ensemble,
        instances: ndarray,
        labels: ndarray
    ) -> Ensemble:
        ensemble.compute_pareto_frontier(
            instances,
            labels,
            self.computation,
            self.minimization,
            **self.q_measure_params,
        )
        ensemble.swap(
            ensemble.ensemble,
            list(range(len(ensemble.ensemble))),
            instances,
            labels,
            True,
        )

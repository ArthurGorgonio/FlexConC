from src.ssl.flexcon import BaseFlexCon


class CoFlexCon(BaseFlexCon):
    def __init__(self, base_estimator):
        super().__init__(base_estimator=base_estimator)

    def fit(self, X, y):
        pass

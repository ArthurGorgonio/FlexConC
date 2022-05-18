from src.reaction.interfaces.IReaction import IReaction


class Exchange(IReaction):
    def __init__(self):
        self.oracle = None

    def swap_ensemble(self, oracle, ensemble):
        pass

    def measure_ensemble(self, ensemble):
        pass

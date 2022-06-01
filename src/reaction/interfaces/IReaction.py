from abc import abstractmethod


class IReaction:
    def __init__(self):
        pass

    @abstractmethod
    def react(self, ensemble):
        ...

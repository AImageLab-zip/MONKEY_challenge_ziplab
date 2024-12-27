from .AbstractExperiment import AbstractExperiment

class BaselineDetectronExperiment(AbstractExperiment):
    def __init__(self, args, config):
        super().__init__(args, config) # Call the constructor of the parent class
        

    def train(self):
        pass

    def test(self):
        pass

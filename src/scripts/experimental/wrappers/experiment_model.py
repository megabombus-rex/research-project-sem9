import pytorch_lightning as L

class BaseModel:
    def __init__(self, model: L.LightningModule):
        self.model = model
        pass

    def __call__(self, x):
        pass

    # fitting, training etc.
    def fit(self, X, y):
        pass

    def predict(self, x):
        pass
    
    # will probably do the same, but without batch
    def predict_batch(self, x, batch_size):
        pass

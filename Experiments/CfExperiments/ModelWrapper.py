from abc import ABC, abstractmethod
import pickle
import numpy as np
import pandas as pd


class ModelWrapper(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict_proba(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class RfWrapper(ModelWrapper):
    def __init__(self, model_path, pp_path):
        with open(model_path,'rb') as f:
            self.clf = pickle.load(f)
        with open(pp_path,'rb') as f:
            self.pp = pickle.load(f)

    def predict_proba(self,x):
        if type(x) == pd.DataFrame:
            x =x.astype(np.float32)
            return self.clf.predict_proba(self.pp.transform(x.values))
        else:
            return self.clf.predict_proba(self.pp.transform(x))

    def predict(self,x):
        return self.predict_proba(x).argmax(axis=1)

    def __call__(self,x):
        return self.predict_proba(x)

class AeWrapper():
    def __init__(self, ae_path, pp_path):
        with open(ae_path, 'rb') as f:
            self.ae= pickle.load(f)
        with open(pp_path, 'rb') as f:
            self.pp = pickle.load(f)

    def predict(self,x):
        return self.ae.predict(self.pp.transform(x))


    def __call__(self,x):
        reconstruction = self.predict(x)
        original= self.pp.transform(x)
        rmse = ((original-reconstruction)**2).sum(axis=1)
        return rmse





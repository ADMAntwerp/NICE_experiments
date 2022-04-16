from Experiments.Modeling.AutoEncoder import AeTrainer
from Experiments.PathGenerator import path_generator
from Experiments.DataLoader import TabularDataLoader
import pickle


class ImTrainer(AeTrainer):
    def __init__(self, dataset:str, label:int):
        self.paths = path_generator(dataset)
        self.save_path = self.paths[f'ae_{label}']
        self.dataset = TabularDataLoader(self.paths['dataset'])
        with open(self.paths['pp'],'rb') as f:
            self.pp = pickle.load(f)
        self.input_shape = self.pp.transform(self.dataset.X_train).shape[1]
        self.dataset.X_train = self.dataset.X_train[self.dataset.y_train==label,:]

    def save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.best_ae, f)
from Experiments.PathGenerator import path_generator
from Experiments.DataLoader import TabularDataLoader
from Experiments.CfExperiments.ModelWrapper import RfWrapper
import pickle
from math import ceil
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

AE_GRID = {'learning_rate_init': [0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001]}




class AeTrainer:
    def __init__(self, dataset:str):
        self.paths = path_generator(dataset, 'RF')
        self.dataset = TabularDataLoader(self.paths['dataset'])
        with open(self.paths['pp'],'rb') as f:
            self.pp = pickle.load(f)
        self.input_shape = self.pp.transform(self.dataset.X_train).shape[1]

    def grid_search(self, grid):
        latent_size = 2 if int(ceil(self.input_shape/4))<4 else 4
        hidden_shapes = (int(ceil(self.input_shape/2)), int(ceil(self.input_shape/4)), latent_size,
                         int(ceil(self.input_shape/4)), int(ceil(self.input_shape/2)))
        print(hidden_shapes)
        self.ae = MLPRegressor(
            hidden_layer_sizes= hidden_shapes,
            activation= 'tanh',
            solver= 'adam',
            learning_rate_init=0.005,
            learning_rate='adaptive',
            max_iter= 100,
            tol= 0.00001,
            verbose= 0,
            validation_fraction=0.2,
            early_stopping=True,
            n_iter_no_change=5
        )
        gs = GridSearchCV(self.ae,grid, n_jobs= -2, verbose=0)
        gs.fit(self.pp.transform(self.dataset.X_train), self.pp.transform(self.dataset.X_train))
        self.best_ae = gs.best_estimator_
        self.best_ae.fit(self.pp.transform(self.dataset.X_train), self.pp.transform(self.dataset.X_train))

    def save(self):
        with open(self.paths['ae'], 'wb') as f:
            pickle.dump(self.best_ae, f)
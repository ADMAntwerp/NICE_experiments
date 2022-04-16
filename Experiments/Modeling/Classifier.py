from abc import ABC,abstractmethod
from Experiments.PathGenerator import path_generator
from Experiments.DataLoader import TabularDataLoader
import pandas as pd
from library import PARAMETERS_GRID
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

MODELS = {'RF':RandomForestClassifier,
          'ANN': MLPClassifier}
STATIC_PARAMS = {
    'RF': {'class_weight': 'balanced'},
    'ANN': {
        'solver': 'lbfgs',
        'learning_rate': 'adaptive',
        'early_stopping': True,
        'max_iter': 1000,
        'tol': 0.0001
}}


class Modeler(ABC):
    @abstractmethod
    def grid_search(self):
        pass

    @abstractmethod
    def save(self):
        pass



class SklearnTabularModeler(Modeler):
    def __init__(self,dataset_name:str, model):
        self.dataset_name= dataset_name
        self.paths = path_generator(dataset_name,model)
        self.model = MODELS[model](**STATIC_PARAMS[model])
        self.dataset = TabularDataLoader(self.paths['dataset'])
        with open(self.paths['pp'], 'rb') as f:
            self.pp = pickle.load(f)

    def grid_search(self, grid):
        X_train_pp = self.pp.transform(self.dataset.X_train)
        gs = GridSearchCV(self.model,grid,verbose=0, cv=5, scoring= 'roc_auc', n_jobs=-1)
        gs.fit(X_train_pp,self.dataset.y_train)
        self.best_model = gs.best_estimator_
        self.best_model.fit(X_train_pp, self.dataset.y_train)

    def save(self):
        with open(self.paths['model'], 'wb') as f:
            pickle.dump(self.best_model, f)

    def save_stats(self):
        stats = pd.DataFrame()
        y_score= self.best_model.predict_proba(self.pp.transform(self.dataset.X_test))[:,1]
        y_pred = self.best_model.predict(self.pp.transform(self.dataset.X_test))

        stats.loc['Imbalance',self.dataset_name]= self.dataset.y_train.mean()
        stats.loc['Train size', self.dataset_name] = self.dataset.X_train.shape[0]
        stats.loc['Test size', self.dataset_name] = self.dataset.X_test.shape[0]
        stats.loc['n features', self.dataset_name] = self.dataset.X_test.shape[1]
        stats.loc['n cat features'] = len(self.dataset.cat_feat)
        stats.loc['n con features'] = len(self.dataset.con_feat)

        stats.loc['auc', self.dataset_name]= roc_auc_score(self.dataset.y_test, y_score)
        stats.loc['Accuracy']= accuracy_score(self.dataset.y_test,y_pred)
        stats.loc['Precision', self.dataset_name] = precision_score(self.dataset.y_test,y_pred)
        stats.loc['F1', self.dataset_name] = f1_score(self.dataset.y_test, y_pred)

        stats.loc['Recall', self.dataset_name] = recall_score(self.dataset.y_test, y_pred)
        cf_matrix = confusion_matrix(self.dataset.y_test,y_pred)
        cf_matrix = cf_matrix/np.sum(cf_matrix)
        stats.loc['TP test', self.dataset_name] = cf_matrix[0, 0]
        stats.loc['FP test', self.dataset_name] = cf_matrix[1, 0]
        stats.loc['FN test', self.dataset_name] = cf_matrix[0, 1]
        stats.loc['TN test', self.dataset_name] = cf_matrix[1, 1]

        cf_matrix = confusion_matrix(
            self.dataset.y_train,
            self.best_model.predict(self.pp.transform(self.dataset.X_train)))
        cf_matrix = cf_matrix / np.sum(cf_matrix)
        stats.loc['TP train', self.dataset_name] = cf_matrix[0, 0]
        stats.loc['FP train', self.dataset_name] = cf_matrix[1, 0]
        stats.loc['FN train', self.dataset_name] = cf_matrix[0, 1]
        stats.loc['TN train', self.dataset_name] = cf_matrix[1, 1]

        for k,v in self.best_model.get_params().items():
            stats.loc[k, self.dataset_name]= v

        stats.to_csv(self.paths['model_stats'])

def dynamic_MLP_layers(dataset_name,n_options,max_scale):
    paths = path_generator(dataset_name)
    dataset = TabularDataLoader(paths['dataset'])
    with open(paths['pp'], 'rb') as f:
        pp = pickle.load(f)

    input_size = pp.transform(dataset.X_train).shape[1]
    max_size = int(input_size*max_scale)
    step = int(np.ceil((max_size - 2)/n_options))
    grid = list(range(2,max_size,step))
    grid = [(i,) for i in grid]
    return grid














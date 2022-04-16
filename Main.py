from Experiments.CfExperiments.CfRunner import CfRunner
from Experiments.Modeling.Classifier import SklearnTabularModeler, dynamic_MLP_layers
from Experiments.PathGenerator import path_generator
from Experiments.DataLoader import TabularDataLoader
from Experiments.Modeling.AutoEncoder import AeTrainer
from library import datasets_pmlb, datasets_kaggle, PARAMETERS_GRID, AE_GRID
from Experiments.DataGathering.Fetchers import PmlbFetcher
import argparse
from Experiments.Modeling.Preprocessing import PpModeler
from tqdm import tqdm

models=  ['ANN','RF']
CF_WRAPPERS = ['dice','nicespars','niceprox','niceplaus','cfproto','cbr','geco','nicenone','wit','sedc']

def fetch_datasets(dataset_name):
    fetcher = PmlbFetcher(dataset_name)
    fetcher.save()

def run_preprocessing(dataset_name):
    pp = PpModeler(dataset_name)
    pp.save()

def run_modeling(dataset_name, model, grid):
    modeler= SklearnTabularModeler(dataset_name, model)
    if model == 'ANN':
        grid['hidden_layer_sizes']= dynamic_MLP_layers(dataset_name,10,1.5)
    modeler.grid_search(grid)
    modeler.save()
    modeler.save_stats()

def run_ae_modeling(dataset):
    ae_modeler = AeTrainer(dataset)
    ae_modeler.grid_search(AE_GRID)
    ae_modeler.save()

def run_cf_experiment(dataset, model, cf):
    experiment = CfRunner(cf, model, dataset)
    experiment.run_experiment()


if __name__ == '__main__':

    for dataset in datasets_pmlb:
        print(dataset)
        fetch_datasets(dataset)
        run_preprocessing(dataset)
        run_ae_modeling(dataset)
        for model in models:
            run_modeling(dataset,model, PARAMETERS_GRID[model])
            for cf in cfs:
                run_cf_experiment(dataset, model, cf)



from joblib import Parallel, delayed
from library import datasets_pmlb
import os
import warnings
from Experiments.CfExperiments.CfRunner import CfRunnerParallel, CfRunner
from Experiments.PathGenerator import path_generator
import pickle
from random import shuffle
import argparse
warnings.filterwarnings('ignore')



def run_cf_experiment(dataset, model, cf):
    experiment = CfRunner(cf, model, dataset)
    experiment.run_experiment(overwrite=True)



arguments = list()
cfs = [['wit'],['cbr'],['geco']]
for cf in cfs:
    for model in ['ANN','RF']:
        for dataset in datasets_pmlb:
                    arg = (dataset,model,cf)
                    arguments.append(arg)
                    print(arg)

print(len(arguments))


def thread_function(arg):
    dataset = arg[0]
    model = arg[1]
    cfs= arg[2]
    for cf in cfs:
        try:
            run_cf_experiment(dataset,model,cf)
        except:
            print(dataset,model,cfs, 'failed')



parser = argparse.ArgumentParser()
parser.add_argument('n_jobs', type=int)
n_jobs = parser.parse_args().n_jobs

Parallel(n_jobs=n_jobs,verbose=11)(
    delayed(thread_function)(i) for i in arguments
)


from Experiments.ResultProcessing import ModelStats, ResultDataLoader, ProgressChecker, SummaryStats
from Experiments.ResultProcessing.IM import ImTrainer
from library import datasets_pmlb
from Experiments.Modeling.AutoEncoder import AE_GRID

ss = SummaryStats(dataset_names=datasets_pmlb)
ss.generate_tables()

rdl = ResultDataLoader(model='ANN')
rdl.generate_tables()
rdl = ResultDataLoader(model='RF')
rdl.generate_tables()
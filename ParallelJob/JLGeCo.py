from joblib import Parallel, delayed
from Main import run_cf_experiment
from library import datasets_pmlb
import os
import warnings
from Experiments.CfExperiments.CfRunner import CfRunnerParallel, CfRunnerMultipleCf
from Experiments.PathGenerator import path_generator
import pickle
from random import shuffle
warnings.filterwarnings('ignore')

def run_cf_experiment(dataset, model, cf,n):
    experiment = CfRunnerMultipleCf(cf, model, dataset,n)
    experiment.run_experiment()



arguments = list()
cf = 'geco'

for model in ['ANN','RF']:
    for dataset in datasets_pmlb:
        for n in [3]:
            if not os.path.isfile(f'./Data/{dataset}/results/{model}_{cf}_{n}.pkl'):
                arg = (dataset,model,cf,n)
                arguments.append(arg)
                print(arg)

print(len(arguments))


def thread_function(arg):
    dataset = arg[0]
    model = arg[1]
    cfs= arg[2]
    n= arg[3]
    run_cf_experiment(dataset,model,cf,n)


Parallel(n_jobs=2,verbose=11)(
    delayed(thread_function)(i) for i in arguments
)



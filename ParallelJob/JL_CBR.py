from joblib import Parallel, delayed
from Main import run_cf_experiment
from library import datasets_pmlb
import os
import warnings
from Experiments.CfExperiments.CfRunner import CfRunnerParallel, CfRunnerTolerance
from Experiments.PathGenerator import path_generator
import pickle
warnings.filterwarnings('ignore')

def run_cf_experiment(dataset, model, cf,tolerance):
    try:
        experiment = CfRunnerTolerance(cf, model, dataset,tolerance)
        experiment.run_experiment()
    except:
        print(dataset,model,cf,tolerance,'failed')



arguments = list()
for i in range(37,50):
    for model in ['ANN','RF']:
        for dataset in datasets_pmlb:
                arg = (dataset,model,'cbr',i)
                arguments.append(arg)
                print(arg)



def thread_function(arg):
    dataset = arg[0]
    model = arg[1]
    cf= arg[2]
    tolerance =arg[3]
    run_cf_experiment(dataset,model,cf,tolerance)

thread_function(arguments[0])

Parallel(n_jobs=-1,verbose=11)(
    delayed(thread_function)(i) for i in arguments
)
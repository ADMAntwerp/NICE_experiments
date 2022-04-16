from joblib import Parallel, delayed
from library import datasets_pmlb
import os
import warnings
from Experiments.CfExperiments.CfRunner import CfRunnerParallel
from Experiments.PathGenerator import path_generator
import pickle
from random import shuffle
import argparse
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('n_jobs', type=int)
n_jobs = parser.parse_args().n_jobs


def run_cf_experiment(dataset, model, cf,i):
    experiment = CfRunnerParallel(cf, model, dataset)
    experiment.run_experiment(i)


arguments = list()
cf = 'dice'
step=10

for dataset in datasets_pmlb:
    for model in ['RF']:
        paths = path_generator(dataset,model,cf)
        if os.path.isfile(paths['results']):
            with open(paths['results'], 'rb') as f:
                results = pickle.load(f)
            if len(results)<200:
                leftovers = list(range(len(results),200))
                path= f'./Data/{dataset}/results/{cf}_{model}'
                if os.path.isdir(path):
                    leftovers_done = os.listdir(path)
                    leftovers_done = [int(i[:-4]) for i in leftovers_done]
                    leftovers = [i for i in leftovers if i not in leftovers_done]

                for i in leftovers:
                    arg = (dataset,model,cf,i)
                    arguments.append(arg)
                    print(arg)
print(len(arguments))

print('reverse')




def thread_function(arg):
    dataset = arg[0]
    model = arg[1]
    cf= arg[2]
    i =[arg[3]]
    run_cf_experiment(dataset,model,cf,i)


Parallel(n_jobs= n_jobs, verbose=11)(
    delayed(thread_function)(i) for i in arguments[::-1]
)



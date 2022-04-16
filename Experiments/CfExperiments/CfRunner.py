from Experiments.CfExperiments.CfWrapper import DiceWrapper, NiceSparsityWrapper, NiceProximityWrapper, \
    CfProtoWrapper, CbrWrapper, GecoWrapper, NiceNoneWrapper, WitWrapper, NicePlausibilityWrapper, WachterWrapper, \
    SedcWrapper
from Experiments.PathGenerator import path_generator
from Experiments.DataLoader import TabularDataLoader
from Experiments.CfExperiments.ModelWrapper import RfWrapper, AeWrapper
from time import time
import pickle
from tqdm import tqdm
import os

CF_WRAPPERS = {'dice': DiceWrapper,
               'nicespars': NiceSparsityWrapper,
               'niceprox': NiceProximityWrapper,
               'niceplaus': NicePlausibilityWrapper,
               'cfproto':CfProtoWrapper,
               'cbr':CbrWrapper,
               'geco': GecoWrapper,
               'nicenone':NiceNoneWrapper,
               'wit': WitWrapper,
               'wachter': WachterWrapper,
               'sedc': SedcWrapper}

MODEL_WRAPPERS = {'RF':RfWrapper,
                  'ANN': RfWrapper}


class CfRunner:
    def __init__(self, cf:str, model:str, dataset:str,verbose = 1):
        print(f'Initiating {dataset}-{model}-{cf}')
        self.cf_name=cf
        self.desc = f'{model} //{dataset} // {cf}'
        self.verbose = verbose
        self.paths = path_generator(dataset, model, cf)
        self.dataset = TabularDataLoader(self.paths['dataset'])
        self.model = MODEL_WRAPPERS[model](self.paths['model'],self.paths['pp'])
        ae = AeWrapper(self.paths['ae'],self.paths['pp'])
        self.CF_algo = CF_WRAPPERS[cf](self.dataset, self.model, auto_encoder=ae)

    def run_experiment(self,overwrite= False):
        save_time = time()
        if os.path.isfile(self.paths['results'])&(not overwrite):
            with open(self.paths['results'], 'rb') as f:
                results = pickle.load(f)
        else:
            results= []

        for row in tqdm(range(len(results),self.dataset.X_explain.shape[0]),desc = self.desc):
            result = {'original': self.dataset.X_explain[row:row + 1, :]}
            result['cf'], result['time'] = self.CF_algo.explain(result['original'])
            results.append(result)
            if (time()-save_time)>180:
                with open(self.paths['results'], 'wb') as f:
                    pickle.dump(results, f)
                save_time=time()

        with open(self.paths['results'], 'wb') as f:
            pickle.dump(results, f)



class CfRunnerParallel(CfRunner):
    def __init__(self, cf:str, model:str, dataset:str,verbose = 1):
        print(f'Initiating {dataset}-{model}-{cf}')
        self.cf_name=cf
        self.desc = f'{model} //{dataset} // {cf}'
        self.verbose = verbose
        self.paths = path_generator(dataset, model, cf)
        self.dataset = TabularDataLoader(self.paths['dataset'])
        self.model = MODEL_WRAPPERS[model](self.paths['model'],self.paths['pp'])
        ae = AeWrapper(self.paths['ae'],self.paths['pp'])
        self.CF_algo = CF_WRAPPERS[cf](self.dataset, self.model, auto_encoder=ae)
        self.save_path = f'./Data/{dataset}/results/{cf}_{model}'
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

    def run_experiment(self,idx_list):
        for row in tqdm(idx_list,desc = self.desc):
            result = {'original': self.dataset.X_explain[row:row + 1, :]}
            result['cf'], result['time'] = self.CF_algo.explain(result['original'])

            with open(f'{self.save_path}/{row}.pkl','wb') as f:
                pickle.dump(result, f)


class CfRunnerTolerance(CfRunner):
    def __init__(self, cf:str, model:str, dataset:str, tolerance):
        print(f'Initiating {dataset}-{model}-{cf}')
        self.tolerance=str(tolerance)
        self.cf_name=cf
        self.desc = f'{model} //{dataset} // {cf}'
        self.paths = path_generator(dataset, model, cf)
        self.dataset = TabularDataLoader(self.paths['dataset'])
        self.model = MODEL_WRAPPERS[model](self.paths['model'],self.paths['pp'])
        ae= None
        self.CF_algo = CF_WRAPPERS[cf](self.dataset, self.model, auto_encoder=ae,tolerance= tolerance/100)

    def run_experiment(self):
        if os.path.isfile(self.paths['results']):
            with open(self.paths['results'], 'rb') as f:
                results = pickle.load(f)
        else:
            results = []

        for row in tqdm(range(len(results), self.dataset.X_explain.shape[0]), desc=self.desc):
            result = {'original': self.dataset.X_explain[row:row + 1, :]}
            result['cf'], result['time'] = self.CF_algo.explain(result['original'])
            results.append(result)
        result_path= self.paths['results'][:-4]
        result_path = f'{result_path}_{self.tolerance}.pkl'
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)

class CfRunnerMultipleCf(CfRunner):
    def __init__(self, cf:str, model:str, dataset:str, n):
        print(f'Initiating {dataset}-{model}-{cf}')
        self.tolerance=str(n)
        self.cf_name=cf
        self.desc = f'{model} //{dataset} // {cf}'
        self.paths = path_generator(dataset, model, cf)
        self.dataset = TabularDataLoader(self.paths['dataset'])
        self.model = MODEL_WRAPPERS[model](self.paths['model'],self.paths['pp'])
        ae= None
        self.CF_algo = CF_WRAPPERS[cf](self.dataset, self.model, auto_encoder=ae,n=n)

    def run_experiment(self):
        if os.path.isfile(self.paths['results']):
            with open(self.paths['results'], 'rb') as f:
                results = pickle.load(f)
        else:
            results = []

        for row in tqdm(range(len(results), self.dataset.X_explain.shape[0]), desc=self.desc):
            result = {'original': self.dataset.X_explain[row:row + 1, :]}
            result['cf'], result['time'] = self.CF_algo.explain(result['original'])
            results.append(result)
        result_path= self.paths['results'][:-4]
        result_path = f'{result_path}_{self.tolerance}.pkl'
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)





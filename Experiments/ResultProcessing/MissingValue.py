import os
import pickle
from Experiments.DataLoader import TabularDataLoader
from Experiments.PathGenerator import path_generator
import numpy as np

def missing_filler(model):
    path = './Data'
    for map in os.listdir(path):
        for file in os.listdir(f'{path}/{map}/results'):
            if model in file:
                pkl_path = f'{path}/{map}/results/{file}'
                if os.path.isfile(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        results = pickle.load(f)
                    if len(results) < 200:
                        cf = file.split('.')[0].split('_')[1]
                        seperate_path = f'{path}/{map}/results/{cf}_{model}'
                        if os.path.isdir(seperate_path):
                            seperate_results = os.listdir(seperate_path)
                            seperate_results = [i[:-4] for i in seperate_results]
                            if len(seperate_results) != (200 - len(results)):
                                missing = [i for i in range(len(results), 200) if str(i) not in seperate_results]
                                dataset = TabularDataLoader(path_generator(map)['dataset'])
                                for i in missing:
                                    result = {'original': dataset.X_explain[i:i+1,:],
                                              'time':np.nan}
                                    if (cf== 'dice') |(cf == 'geco'):
                                        result['cf']= np.tile(np.nan,(3,result['original'].shape[1]))
                                    else:
                                        result['cf'] = np.tile(np.nan, result['original'].shape)
                                    with open(f'{seperate_path}/{i}.pkl', 'wb') as f:
                                        pickle.dump(result,f)
                                    print(map,model,i)
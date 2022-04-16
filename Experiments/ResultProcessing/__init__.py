import os
import pandas as pd
import pickle
import numpy as np
from scipy import stats
from library import cvnt005, cvnt001,cvnt010
from Experiments.PathGenerator import path_generator
from Experiments.CfExperiments.ModelWrapper import RfWrapper
from Experiments.ResultProcessing.distance import HEOM,l2MinMaxDistance, l1MinMaxDistance, l1StandardDistance, \
    l2StandardDistance
from Experiments.PathGenerator import path_generator
from Experiments.DataLoader import TabularDataLoader
from tqdm import tqdm
from Experiments.CfExperiments.ModelWrapper import AeWrapper
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score, accuracy_score



class ModelStats:
    def __init__(self,model):
        dataset_names = os.listdir('./data/')
        self.stats = []
        for name in dataset_names:
            path= f'./data/{name}/{model}_stats.csv'
            if os.path.isfile(path):
                self.stats.append(pd.read_csv(path,index_col=0))
        self.stats = pd.concat(self.stats,axis=1).T

    def save(self, path):
        self.stats.to_csv(path)


class ProgressChecker:
    def __init__(self, path='./Data', model='ANN'):
        self.model = model
        self.results = {}
        for map in os.listdir(path):
            self.results[map] = {}
            for file in os.listdir(f'{path}/{map}/results'):
                if model in file:
                    pkl_path = f'{path}/{map}/results/{file}'
                    if os.path.isfile(pkl_path):
                        with open(pkl_path, 'rb') as f:
                            results = pickle.load(f)
                        if len(results)<200:
                            cf = file.split('.')[0].split('_')[1]
                            seperate_path = f'{path}/{map}/results/{cf}_{model}'
                            if os.path.isdir(seperate_path):
                                if len(os.listdir(seperate_path))==(200-len(results)):
                                    for i in range(len(results),200):
                                        with open(f'{seperate_path}/{i}.pkl','rb') as f:
                                            results.append(pickle.load(f))
                        self.results[map][file.split('.')[0]] = results

    def save_time(self):
        df_time_average = pd.DataFrame()
        df_count = pd.DataFrame()
        df_time_std = pd.DataFrame()
        di_time = {}
        for name, values in self.results.items():
            di_time[name]={}
            for model,v in values.items():
                time =  [v1['time'] for v1 in v]
                di_time[name][model]=time
                df_time_average.loc[name,model] = np.mean(time)
                df_count.loc[name, model] = 200-len(time)
                df_time_std.loc[name, model] = np.std(time)

        remaining_time = (df_time_average*df_count)/3600
        remaining_time.to_csv(f'./Tables/remaining_time_{self.model}.csv')
        df_count.to_csv(f'./Tables/remaining_count_{self.model}.csv')
        df_time_average.to_csv(f'./Tables/Time_average_{self.model}.csv')
        df_time_std.to_csv(f'./Tables/Time_std_{self.model}.csv')

class SummaryStats:
    def __init__(self,dataset_names):
        self.dataset_names= dataset_names

    def generate_tables(self):
        results = pd.DataFrame()
        for name in self.dataset_names:
            paths = path_generator(name)
            data= TabularDataLoader(paths['dataset'])
            results.loc[name,'n training instances']= data.X_train.shape[0]
            results.loc[name, 'n instances']=data.X_train.shape[0]+data.X_test.shape[0]
            results.loc[name,'n features']= data.X_train.shape[1]
            results.loc[name, 'n cat features']= len(data.cat_feat)
            results.loc[name, 'n con features']= len(data.con_feat)
            results.loc[name, 'imbalance']= np.mean(data.y_train)
            for model in ['ANN', 'RF']:
                paths= path_generator(name,model)
                clf= RfWrapper(paths['model'],paths['pp'])
                results.loc[name, f'auc {model}']= roc_auc_score(data.y_test, clf.predict_proba(data.X_test)[:,1])
                results.loc[name, f'accuracy {model}']= accuracy_score(data.y_test, clf.predict(data.X_test))
        results.to_csv('./tables/summary_stats.csv')



class ResultDataLoader:
    def __init__(self,path = './Data', model= 'ANN'):
        self.model = model
        self.results = {}
        for map in os.listdir(path):
            self.results[map] = {}
            for file in os.listdir(f'{path}/{map}/results'):
                if model in file:
                    pkl_path = f'{path}/{map}/results/{file}'
                    if os.path.isfile(pkl_path):
                        with open(pkl_path, 'rb') as f:
                            results = pickle.load(f)
                        if len(results)<200:
                            cf = file.split('.')[0].split('_')[1]
                            seperate_path = f'{path}/{map}/results/{cf}_{model}'
                            if os.path.isdir(seperate_path):
                                if len(os.listdir(seperate_path))==(200-len(results)):
                                    for i in range(len(results),200):
                                        with open(f'{seperate_path}/{i}.pkl','rb') as f:
                                            results.append(pickle.load(f))
                        if len(results)==200:
                            self.results[map][file.split('.')[0]] = results
        self.check_sync()
        self.concat_results()
        self.save_coverage()

    def check_sync(self):
        self.original = {}
        for name, values in self.results.items():
            originals = {}
            for model,v in values.items():
                originals[model] = np.concatenate([[v1['original']] for v1 in v])
            for k1,v1 in originals.items():
                for k2,v2 in originals.items():
                    if np.any(v1!=v2):
                        raise ValueError(f'Originals {k1} {k2} of {name} do not match')
                self.original[name]= v1

    def concat_results(self):
        self.cf = {}
        for name, values in self.results.items():
            self.cf[name]={}
            for model,v in values.items():
                self.cf[name][model] = np.concatenate([v1['cf'][np.newaxis,:,:] for v1 in v],axis=0)

    def save_time(self):
        dic_time = {}
        for name, values in self.results.items():
            df_time = pd.DataFrame()
            for model,v in values.items():
                times = [abs(v1['time']) for v1 in v]#apps for negative time
                df_time.loc[:, model] = times
            dic_time[name] = df_time.copy()
        df_time = pd.concat(dic_time)
        df_time.groupby(level=0).mean().to_csv(f'./Tables/Time_mean_{self.model}.csv')
        df_time.groupby(level=0).std().to_csv(f'./Tables/Time_std_{self.model}.csv')
        df_time = df_time.reset_index(drop=True)
        self.friedman(df_time).to_csv(f'./Tables/Time_ranks_{self.model}.csv')


    def save_coverage(self):
        df_coverage = pd.DataFrame()
        df_correct_coverage = pd.DataFrame()
        for name,v in self.cf.items():
            paths = path_generator(name,self.model)
            clf = RfWrapper(paths['model'], paths['pp'])
            for model, v1 in v.items():
                coverage = ~np.all(np.isnan(v1), axis=2)
                df_coverage.loc[name, model] = np.mean(np.any(coverage,axis=1))
                original_class = clf(self.original[name][:,0,:]).argmax(axis=1)
                for i in range(coverage.shape[1]):
                    if np.any(coverage[:,i]):
                        cf_class = clf(v1[coverage[:,i],i,:]).argmax(axis=1)
                        correct_coverage= original_class[coverage[:,i]]!=cf_class
                        if not np.all(correct_coverage):
                            coverage[coverage[:, i], i] = correct_coverage
                            v1[~coverage[:,i],i,:]= np.nan #change counterfactuals with ^y==^y original to missing
                df_correct_coverage.loc[name, model] = np.mean(~np.all(np.isnan(v1), axis=2))
                self.cf[name][model]=v1

        df_coverage.to_csv(f'./Tables/Coverage_mean_{self.model}.csv')
        df_correct_coverage.to_csv(f'./Tables/Correct_Coverage_mean_{self.model}.csv')

    def generate_tables(self):
        self.save_time()
        metrics = {
            'CM_robustness':{},
            'sparsity':{},
            'HEOM_l1_mm':{},
            'HEOM_l1_std': {},
            'HEOM_l2_mm': {},
            'HEOM_l2_std': {},
            'cosine_dist': {},
            'AE_loss':{},
            'IM1': {},
            'IM2': {},
            'HEOM_5NN':{}
        }
        for name in tqdm(self.original.keys()):
            for key in metrics.keys():
                metrics[key][name]={}

            for model in self.cf[name].keys():
                original = self.original[name]
                cf = self.cf[name][model]
                metrics['CM_robustness'][name][model]= self.CM_robustness(cf,name)
                metrics['sparsity'][name][model] = self.sparsity(original, cf)
                metrics['HEOM_l1_mm'][name][model]= self.HEOM(original, cf, name, l1MinMaxDistance)
                metrics['HEOM_l1_std'][name][model] = self.HEOM(original, cf, name, l1StandardDistance)
                metrics['HEOM_l2_mm'][name][model] = self.HEOM(original, cf, name, l2MinMaxDistance)
                metrics['HEOM_l2_std'][name][model] = self.HEOM(original, cf, name, l2StandardDistance)
                metrics['cosine_dist'][name][model] = self.cosine_distance(original, cf, name)
                metrics['AE_loss'][name][model]= self.AE_loss(cf,name)
                metrics['IM1'][name][model] = self.IM1(original,cf, name, self.model)
                metrics['IM2'][name][model] = self.IM2(original, cf, name, self.model)
                metrics['HEOM_5NN'][name][model] = self.HEOM_5NN(cf, name,k=5)


        di_friedman = {}
        df_metrics =  {k0:pd.concat({k1:pd.DataFrame(v1) for k1,v1 in v0.items()}) for k0,v0 in metrics.items()}
        df_means = pd.DataFrame()
        for metric, df in df_metrics.items():
            #df = pd.concat({k1:pd.DataFrame(v1) for k1,v1 in values.items()})
            df.groupby(level=0).mean().to_csv(f'./Tables/{metric}_average_{self.model}.csv')
            df.groupby(level=0).std().to_csv(f'./Tables/{metric}_std_{self.model}.csv')
            df_means.loc[:,metric]=df.mean(axis=0)
            df = df.reset_index(drop=True)
            di_friedman[metric]= self.friedman(df)
        df_means.to_csv(f'./tables/Means_{self.model}.csv')
        pd.concat(di_friedman,axis=1).T.to_csv(f'./tables/Friedman_{self.model}.csv')
        self.correlation(di_metrics= df_metrics)
        self.percentage_best(df_metrics)

    def percentage_best(self,df_metrics):
        best = pd.DataFrame()
        for k,v in df_metrics.items():
            best.loc[:,k]=(v.rank(axis=1, na_option='bottom', method='min')==1).mean(axis=0)
        best.T.to_csv(f'./tables/percentage_best_{self.model}.csv')

    def correlation(self,di_metrics):
        di_ranks = {metric:df.rank(axis=1, na_option='bottom', method='min') for metric,df in di_metrics.items()}
        flat_ranks = pd.DataFrame({metric:np.ravel(v.values) for metric,v in di_ranks.items()})
        flat_metrics = pd.DataFrame({metric:np.ravel(v.values) for metric,v in di_metrics.items()})
        flat_ranks.corr().to_csv(f'./tables/correlation_ranks_{self.model}.csv')
        flat_metrics.corr().to_csv(f'./tables/correlation_values_{self.model}.csv')



    @staticmethod
    def sparsity(original,cf):
        return np.min(np.sum(original != cf, axis=2), axis=1)

    @staticmethod
    def HEOM(original, cf, dataset_name, numerical_distance):
        path = path_generator(dataset_name)
        dl= TabularDataLoader(path['dataset'])
        dl.num_feat = dl.con_feat
        dl.eps= 0.0000000001
        heom= HEOM(dl,numerical_distance)
        distance= []
        for i in range(original.shape[0]):
            distance.append(np.min(heom.measure(original[i,:,:],cf[i,:,:])))
        return np.array(distance)

    @staticmethod
    def cosine_distance(original, cf, dataset_name):
        path = path_generator(dataset_name)
        with open(path['pp'],'rb') as f:
            pp = pickle.load(f)

        distance = []
        for i in range(original.shape[0]):
            distance_local=[]
            for j in range(cf.shape[1]):
                distance_local.append(np.min(cosine(pp.transform(original[i, :, :]), pp.transform(cf[i, j:j+1, :]))))
            distance.append(np.min(distance_local))
        return np.array(distance)

    @staticmethod
    def AE_loss(cf,dataset_name):
        AE_loss = []
        paths = path_generator(dataset_name)
        ae = AeWrapper(paths['ae'], paths['pp'])
        for i in range(cf.shape[0]):
            cf_i = cf[i,:,:]
            cf_i= cf_i[~np.all(np.isnan(cf_i),axis=1),:]
            if cf_i.shape[0]>0:
                AE_loss.append(np.min(ae(cf_i)))
            else:
                AE_loss.append(np.nan)
        return np.array(AE_loss)

    @staticmethod
    def CM_robustness(cf,dataset_name):
        CM = []
        paths = path_generator(dataset_name,'RF')
        rf = RfWrapper(paths['model'], paths['pp'])
        paths= path_generator(dataset_name,'ANN')
        ann = RfWrapper(paths['model'],paths['pp'])
        CM = np.tile(False,(cf.shape[0],cf.shape[1]))
        for i in range(cf.shape[1]):#0
            cf_i = cf[:,i,:]
            mask = ~np.all(np.isnan(cf_i),axis=1)
            cf_i= cf_i[mask,:]
            if cf_i.shape[0]>0:
                CM[mask,i]=(rf.predict(cf_i)==ann.predict(cf_i))
        CM = np.any(CM,axis=1)
        return CM

    @staticmethod
    def IM1(original, cf, dataset_name, model_name):
        AE_loss = []
        paths = path_generator(dataset_name, model=model_name)
        clf = RfWrapper(paths['model'], paths['pp'])
        ae_0 = AeWrapper(paths['ae_0'], paths['pp'])
        ae_1 = AeWrapper(paths['ae_1'], paths['pp'])
        for i in range(cf.shape[0]):
            original_i = original[i,:,:]
            cf_i = cf[i, :, :]
            cf_i = cf_i[~np.all(np.isnan(cf_i), axis=1), :]
            if cf_i.shape[0] > 0:
                if clf.predict(original_i)[0] == 0:
                    AE_loss.append(np.min(ae_1(cf_i)/ae_0(cf_i)))
                else:
                    AE_loss.append(np.min(ae_0(cf_i) / ae_1(cf_i)))
            else:
                AE_loss.append(np.nan)
        return np.array(AE_loss)

    @staticmethod
    def IM2(original, cf, dataset_name, model_name):
        AE_loss = []
        paths = path_generator(dataset_name, model=model_name)
        clf = RfWrapper(paths['model'], paths['pp'])
        ae_0 = AeWrapper(paths['ae_0'], paths['pp'])
        ae_1 = AeWrapper(paths['ae_1'], paths['pp'])
        ae = AeWrapper(paths['ae'], paths['pp'])
        for i in range(cf.shape[0]):
            original_i = original[i,:,:]
            cf_i = cf[i, :, :]
            cf_i = cf_i[~np.all(np.isnan(cf_i), axis=1), :]
            if cf_i.shape[0] > 0:
                normalize = np.sum(abs(clf.pp.transform(cf_i) - clf.pp.transform(original_i)), axis=1)
                if clf.predict(original_i) == 0:
                    AE_loss.append(np.min(np.sum((ae_0.predict(cf_i)-ae.predict(cf_i))**2,axis=1)/normalize))
                else:
                    AE_loss.append(np.min(np.sum((ae_1.predict(cf_i)-ae.predict(cf_i))**2,axis=1)/normalize))
            else:
                AE_loss.append(np.nan)
        return np.array(AE_loss)



    @staticmethod
    def HEOM_5NN(cf,dataset_name,k=5):
        path = path_generator(dataset_name)
        dl= TabularDataLoader(path['dataset'])
        dl.num_feat = dl.con_feat
        dl.eps= 0.0000000001
        heom= HEOM(dl,l1MinMaxDistance)
        distance_NN= []
        for i in range(cf.shape[0]):
            local_dist=[]
            for j in range(cf.shape[1]):
                distances= heom.measure(cf[i,j:j+1,:], dl.X_train)
                idx= np.argpartition(distances, k)[:k]
                local_dist.append(np.mean(distances[idx]))
            distance_NN.append(np.min(local_dist))
        return np.array(distance_NN)


    @staticmethod
    def friedman(df):
        df = df.rank(axis=1, na_option='bottom', method='min')
        Rj = df.mean(axis=0)
        N, K = df.shape
        Q = 12 * N / (K * (K + 1)) * np.sum((Rj.values - (K + 1) / 2) ** 2)
        critical_f_value = stats.f.ppf(q=1 - .01, dfn=K - 1, dfd=N - 1)
        if Q > critical_f_value:
            Rj.loc['CD1'] = cvnt001[K] * (K * (K + 1) / (6 * N)) ** 0.5
            Rj.loc['CD5'] = cvnt005[K] * (K * (K + 1) / (6 * N)) ** 0.5
            Rj.loc['CD10'] = cvnt010[K] * (K * (K + 1) / (6 * N)) ** 0.5
        else:
            Rj.loc['CD1'] = np.nan
            critical_f_value = stats.f.ppf(q=1 - .05, dfn=K - 1, dfd=N - 1)
            if Q > critical_f_value:
                Rj.loc['CD5'] = cvnt005[K] * (K * (K + 1) / (6 * N)) ** 0.5
                Rj.loc['CD10'] = cvnt010[K] * (K * (K + 1) / (6 * N)) ** 0.5
            else:
                Rj.loc['CD5'] = np.nan
                critical_f_value = stats.f.ppf(q=1 - .1, dfn=K - 1, dfd=N - 1)
                if Q > critical_f_value:
                    Rj.loc['CD10'] = cvnt010[K] * (K * (K + 1) / (6 * N)) ** 0.5
                else:
                    Rj.loc['CD10']



        return Rj

    def save_all(self):
        self.save_coverage()
        self.save_time()
        self.save_sparsity()











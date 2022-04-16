from abc import ABC, abstractmethod
from CBR import CBR
import dice_ml
import dice_ml_local
import pandas as pd
import numpy as np
from nice import NICE
from alibi.explainers import CounterFactualProto
from alibi.utils.mapping import ord_to_ohe
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from time import time
from scipy import stats
from SEDC.explainer import Explainer as SEDCexplainer

CFPROTO_PARAMS = {'beta':0.01, 'c_init':1, 'c_steps':5, 'max_iterations':500, 'theta':10,'use_kdtree': True}

class CfWrapper(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def explain(self):
        pass

class DiceWrapper(CfWrapper):
    def __init__(self,dataset,model, **kwargs):
        self.n = 3
        Xy = np.concatenate((dataset.X_train,dataset.y_train[:,np.newaxis]), axis= 1)
        Xy_test = np.concatenate((dataset.X_test,dataset.y_test[:,np.newaxis]), axis= 1)
        Xy = np.concatenate((Xy,Xy_test),axis=0)
        self.feature_names = dataset.feature_names + ['y']
        df = pd.DataFrame(Xy,columns= self.feature_names)
        con_feat = [dataset.feature_names[i] for i in dataset.con_feat]

        self.d = dice_ml.Data(dataframe = df, continuous_features = con_feat, outcome_name = 'y')
        self.m = dice_ml.Model(model = model, backend='sklearn')
        self.explainer = dice_ml.Dice(self.d, self.m, method='random')

    def explain(self,x):
        df_x = pd.DataFrame(x, columns= self.feature_names[:-1])
        start = time()
        e1 = self.explainer.generate_counterfactuals(df_x, total_CFs=self.n, desired_class='opposite',verbose= False)
        run_time = time()-start
        try:
            explanation = e1.cf_examples_list[0].final_cfs_df.iloc[:,:-1].values.astype(np.float32)
        except:
            explanation = np.tile(np.nan,(self.n, x.shape[1]))
        if explanation.shape[0]<self.n:
            missing= np.tile(np.nan,(self.n-explanation.shape[0],x.shape[1]))
            explanation = np.concatenate((explanation,missing),axis=0)
        return explanation, run_time

class GecoWrapper(CfWrapper):
    def __init__(self,dataset,model,n=3, **kwargs):
        self.n=n
        self.cat_feat= dataset.cat_feat
        Xy = np.concatenate((dataset.X_train,dataset.y_train[:,np.newaxis]), axis= 1)
        self.feature_names = dataset.feature_names + ['y']
        df = pd.DataFrame(Xy,columns= self.feature_names)
        df.iloc[:,self.cat_feat] = df.iloc[:,self.cat_feat].astype(str)
        con_feat = [dataset.feature_names[i] for i in dataset.con_feat]

        d = dice_ml_local.Data(dataframe = df, continuous_features = con_feat, outcome_name = 'y')
        m = dice_ml_local.Model(model = model, backend='sklearn')
        self.explainer = dice_ml_local.Dice(d, m, method= 'genetic')

    def explain(self,x):
        df_x = pd.DataFrame(x, columns= self.feature_names[:-1])
        df_x.iloc[:,self.cat_feat]=df_x.iloc[:,self.cat_feat].astype(str)
        try:
            start = time()
            e1 = self.explainer.generate_counterfactuals(df_x, total_CFs=self.n, desired_class='opposite',verbose= False)
            run_time = time()-start
            results = e1.cf_examples_list[0].final_cfs_df.iloc[:,:-1].values.astype(np.float32)
        except:
            results = np.tile(np.nan,(self.n, x.shape[1]))
            run_time= np.nan
        if results.shape[0]<self.n:
            missing= np.tile(np.nan,(self.n-results.shape[0],x.shape[1]))
            results = np.concatenate((results,missing),axis=0)
        return results, run_time



class NiceWrapper(CfWrapper):
    def explain(self,x):
        start=  time()
        explanation = self.explainer.explain(x)
        run_time = time() - start
        return explanation, run_time

class NiceSparsityWrapper(NiceWrapper):
    def __init__(self,dataset,model, **kwargs):
        self.explainer = NICE(
            X_train= dataset.X_train,
            predict_fn=model,
            y_train= dataset.y_train,
            cat_feat= dataset.cat_feat,
            num_feat= dataset.con_feat,
            optimization='sparsity',
            justified_cf=True
        )

class NiceProximityWrapper(NiceWrapper):
    def __init__(self,dataset,model, **kwargs):
        self.explainer = NICE(
            X_train= dataset.X_train,
            predict_fn=model,
            y_train= dataset.y_train,
            cat_feat= dataset.cat_feat,
            num_feat= dataset.con_feat,
            optimization='proximity',
            distance_metric= 'HEOM',
            num_normalization= 'minmax',
            justified_cf=True
        )

class NicePlausibilityWrapper(NiceWrapper):
    def __init__(self,dataset,model, auto_encoder, **kwargs):
        self.explainer = NICE(
            X_train= dataset.X_train,
            predict_fn=model,
            y_train= dataset.y_train,
            cat_feat= dataset.cat_feat,
            num_feat= dataset.con_feat,
            optimization='plausibility',
            distance_metric= 'HEOM',
            num_normalization= 'minmax',
            justified_cf=True,
            auto_encoder= auto_encoder
        )

class NiceNoneWrapper(NiceWrapper):
    def __init__(self,dataset,model, **kwargs):
        self.explainer = NICE(
            X_train= dataset.X_train,
            predict_fn=model,
            y_train= dataset.y_train,
            cat_feat= dataset.cat_feat,
            num_feat= dataset.con_feat,
            optimization='none',
            distance_metric= 'HEOM',
            num_normalization= 'minmax',
            justified_cf=True
        )

class WitWrapper(NiceWrapper):
    def __init__(self,dataset,model, **kwargs):
        self.explainer = NICE(
            X_train= dataset.X_train,
            predict_fn=model,
            y_train= dataset.y_train,
            cat_feat= dataset.cat_feat,
            num_feat= dataset.con_feat,
            optimization='none',
            distance_metric= 'HEOM',
            num_normalization= 'std',
            justified_cf=False
        )

class CfProtoWrapper(CfWrapper):
    def __init__(self,dataset,model, **kwargs):
        self.model = model
        shape = self.model.pp.transform(dataset.X_explain)[0:1,:].shape
        ohe = False if dataset.cat_feat == [] else True
        predict_fn = lambda x: model.clf.predict_proba(x)
        rng_shape = (1,) + dataset.X_train.shape[1:]
        feature_range = ((np.ones(rng_shape) * -1).astype(np.float32),
                         (np.ones(rng_shape) * 1).astype(np.float32))

        cat_vars_ord = {}
        for i in range(len(dataset.cat_feat)):
            cat_vars_ord[i] = len(np.unique(dataset.X_train[:, i]))
        cat_vars_ohe = ord_to_ohe(dataset.X_train, cat_vars_ord)[1]

        self.explainer = CounterFactualProto(
            predict_fn,
            shape= shape ,
            cat_vars=cat_vars_ohe,
            ohe=ohe,
            feature_range= feature_range,
            **CFPROTO_PARAMS
        )
        self.explainer.fit(self.model.pp.transform(dataset.X_train), d_type= 'abdm')

    def explain(self,x):
        x_pp = self.model.pp.transform(x)
        start = time()
        explanation_pp = self.explainer.explain(x_pp)['data']['cf']
        run_time = time()-start
        if explanation_pp:
            explanation = self.model.pp.inverse_transform(explanation_pp['X'])
        else:
            explanation = np.empty(x.shape)
            explanation[:] = np.nan
        return explanation, run_time

class CbrWrapper(CfWrapper):
    def __init__(self,dataset,model, tolerance=0.2, **kwargs):
        self.explainer = CBR(dataset.X_train,dataset.y_train,model,dataset.cat_feat, dataset.con_feat,tolerance=tolerance)

    def explain(self,x):
        try:
            start = time()
            explanation= self.explainer.explain(x)
            run_time = time() - start
        except:
            run_time= np.nan
            explanation= np.tile(np.nan, x.shape)
        return explanation, run_time

class SedcWrapper(CfWrapper):
    def __init__(self,dataset,model, **kwargs):
        self.model = model
        self.default_values = np.empty(dataset.X_train.shape[1])
        if dataset.con_feat != []:
            self.default_values[dataset.con_feat] = np.nanmean(dataset.X_train[:, dataset.con_feat], axis=0)
        if dataset.cat_feat != []:
            self.default_values[dataset.cat_feat] = stats.mode(dataset.X_train[:, dataset.cat_feat], axis=0)[0][0]
        self.explainer={
            0:SEDCexplainer(score_f = (lambda x:self.model.predict_proba(x)[:,0]),default_values=self.default_values,prune=False),
            1:SEDCexplainer(score_f=(lambda x: self.model.predict_proba(x)[:, 1]),default_values=self.default_values,prune=False)
        }

    def explain(self,x):
        label = self.model.predict_proba(x).argmax(axis=1)[0]
        start = time()
        explanation=self.explainer[label].explain(x,thresholds=0.5,max_ite=100, stop_at_first= True)
        run_time = time() - start
        if explanation == [[]]:
            array_explanation = np.tile(np.nan,x.shape)
        else:
            array_explanation = x.copy()
            for i in explanation[0][0]:
                array_explanation[0,i]= self.default_values[i]
        return array_explanation, run_time






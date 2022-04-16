import pandas as pd
import numpy as np
from CBR.utils import HEOM

from math import ceil
from scipy.stats import rankdata
class CBR:
    def __init__(self,
        X_train,
        y_train,
        predict_fn,
        cat_feat,
        con_feat = 'auto',
        distance_metric='HEOM',
        explanation_length = 2,
        tolerance = 0.02,
        verbose = 0,):

        self.verbose = verbose
        self.eps = 0.00001,
        self.explanation_length = explanation_length
        self.distance_metric = distance_metric
        self.X_train = X_train.astype(np.float64)
        self.y_train = y_train.astype(np.float64)#todo clean up what's supposed to be internal variable
        self.cat_feat = cat_feat
        self.predict_fn = predict_fn
        self.con_feat = con_feat
        self.tolerance = tolerance
        if self.con_feat == 'auto':
            self.con_feat = [feat for feat in range(self.X_train.shape[1]) if feat not in self.cat_feat]

        if self.distance_metric == 'HEOM':
            self.con_range = self.X_train[:,self.con_feat].max(axis=0)-self.X_train[:,self.con_feat].min(axis=0)
            self.con_range[self.con_range<self.eps]= self.eps

        self.con_tolerance = self.con_range*self.tolerance

        self.X_train_class = np.argmax(self.predict_fn(self.X_train), axis=1)
        self.X_train_0 = self.X_train[self.X_train_class==0,:]
        self.X_train_1 = self.X_train[self.X_train_class == 1, :]
        self.XCS_0 = []
        self.XCS_1 = []
        min_diff = []
        for instance in self.X_train_0:
            con_diff = np.sum(abs(self.X_train_1[:, self.con_feat] - instance[self.con_feat]) > self.con_tolerance, axis=1)
            cat_diff = np.sum(abs(self.X_train_1[:, self.cat_feat] - instance[self.cat_feat]) > self.eps, axis=1)
            diff = cat_diff + con_diff
            diff[diff==0] = self.explanation_length+1
            min_diff.append(diff.min())
            diff_mask = (diff == np.min(diff))*(diff <=self.explanation_length)*(diff >0)
            if np.sum(diff_mask)>0:
                candidates = self.X_train_1[diff_mask,:]
                distance = HEOM(instance[np.newaxis,:], candidates, self.cat_feat, self.con_feat, self.con_range)
                self.XCS_0.append(instance)
                self.XCS_1.append(candidates[np.argmin(distance), :])

        for instance in self.X_train_1:
            min_diff.append(diff.min())
            con_diff = np.sum(abs(self.X_train_0[:, self.con_feat] - instance[self.con_feat]) > self.con_tolerance, axis=1)
            cat_diff = np.sum(abs(self.X_train_0[:, self.cat_feat] - instance[self.cat_feat]) > self.eps, axis=1)
            diff = cat_diff + con_diff
            diff[diff ==0] = self.explanation_length+1
            min_diff.append(diff.min())
            diff_mask = (diff == np.min(diff))&(diff <=self.explanation_length)
            if np.sum(diff_mask)>0:
                candidates = self.X_train_0[diff_mask,:]
                distance = HEOM(instance[np.newaxis,:], candidates, self.cat_feat, self.con_feat, self.con_range)
                self.XCS_1.append(instance)
                self.XCS_0.append(candidates[np.argmin(distance), :])
        self.XCS_0 = np.vstack(self.XCS_0)
        self.XCS_1 = np.vstack(self.XCS_1)

    def explain(self,X,target_class ='other'):#todo target class 'other'
        self.X = X.astype(np.float64)
        self.X_class = np.argmax(self.predict_fn(self.X), axis=1)[0]
        if self.X_class == 0:
            XCS_same = self.XCS_0
            XCS_diff = self.XCS_1
            X_train_diff = self.X_train_1
        else:
            XCS_same = self.XCS_1
            XCS_diff = self.XCS_0
            X_train_diff = self.X_train_0

        distance = HEOM(self.X, XCS_same, self.cat_feat, self.con_feat, self.con_range)
        XC_same = XCS_same[np.argmin(distance),:]
        XC_diff = XCS_diff[np.argmin(distance),:]
        diff_feat = list(np.array(self.con_feat)[abs(XC_same[self.con_feat] - XC_diff[self.con_feat]) > self.con_tolerance])
        diff_feat += list(np.array(self.cat_feat)[abs(XC_same[self.cat_feat] - XC_diff[self.cat_feat]) > self.eps])
        diff_feat.sort()
        self.X[0,diff_feat] = XC_diff[diff_feat].copy()

        if self.predict_fn(self.X).argmax()!=self.X_class:
            return self.X
        else:
            distance = HEOM(self.X,  X_train_diff, self.cat_feat, self.con_feat, self.con_range)
            distance = rankdata(distance,method='ordinal').astype(int)
            X_candidates = np.tile(self.X,(X_train_diff.shape[0],1))
            X_candidates[:,diff_feat]= X_train_diff[:,diff_feat].copy()
            valid = self.predict_fn(X_candidates).argmax(axis = 1)!=self.X_class
            if valid.sum()!=0:
                self.X= X_candidates[distance==distance[valid].min(),:]
                return self.X
        empty = np.empty(self.X.shape)
        empty[:] = np.nan
        if self.verbose:
            print('no CF explanation found')
        return empty
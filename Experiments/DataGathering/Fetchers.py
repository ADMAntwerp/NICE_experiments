from pmlb import fetch_data
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


class PmlbFetcher:
    def __init__(self, name, test_size=0.2, explain_n=200):
        self.folder_path = f'./Data/{name}'
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
            os.mkdir(self.folder_path+'/results')

        dataset = fetch_data(name)
        X = dataset.iloc[:,:-1]
        y = dataset.iloc[:,-1]
        y = y.values
        if name== 'diabetes':
            y=y-1
        con_feat, cat_feat = self._get_feature_types(X)
        feature_names = cat_feat+con_feat
        X = X[feature_names].copy()
        X = X.values
        cat_feat = list(range(len(cat_feat)))
        con_feat = list(range(len(cat_feat), len(cat_feat) + (len(con_feat))))

        feature_map = {}
        for feat in cat_feat:
            local_map = np.unique(X[:, feat])
            feature_map[feat] = local_map
            for i, v in enumerate(local_map):
                X[X[:, feat] == v, feat] = i

        if X.shape[0]*test_size <= explain_n:
            test_size=explain_n
            X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=test_size)
            X_explain = X_test.copy()
            y_explain = y_test.copy()
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size)
            _, X_explain, _, y_explain = train_test_split(X_test, y_test, stratify=y_test, test_size=explain_n)

        print(f'{name}  explain:{X_explain.shape[0]}  test:{X_test.shape[0]}')
        self.dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_explain': X_explain,
            'y_explain': y_explain,
            'feature_names': feature_names,
            'cat_feat': cat_feat,
            'con_feat': con_feat,
            'feature_map': feature_map
        }


    def _get_feature_types(self,X):
        feature_names = list(X.columns)
        X_na= X.dropna()
        con_feat = []
        cat_feat =[]
        for feature_name in feature_names:
            x = X_na[feature_name].copy()
            if  not all(float(i).is_integer() for i in x.unique()):
                con_feat.append(feature_name)
            elif x.nunique() > 10:
                con_feat.append(feature_name)
            else:
                cat_feat.append(feature_name)
        return con_feat, cat_feat


    def save(self):
        with open(f'{self.folder_path}/data.pkl', 'wb') as f:
            pickle.dump(self.dataset,f)
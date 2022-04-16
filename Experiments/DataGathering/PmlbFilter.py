from pmlb import fetch_data
from pmlb import classification_dataset_names
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

dic_result = {}
for dataset in tqdm(classification_dataset_names):
    a = fetch_data(dataset,return_X_y=True,dropna=True)
    dic_result[dataset]= [a[0].shape[0], a[0].shape[1], len(np.unique(a[1]))]

df_results = pd.DataFrame(dic_result).T
filter = df_results.loc[df_results.iloc[:,2]==2].sort_values(by=1,ascending=False)
filter = filter.loc[filter.iloc[:,0]>500]

datasets = list(filter.index)

with open('./Data/dataset_names.pkl', 'wb') as f:
    pickle.dump(datasets,f)

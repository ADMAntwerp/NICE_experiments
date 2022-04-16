import numpy as np
from abc import ABC,abstractmethod

class NumericDistance(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def measure(self):
        pass


class DistanceMetric(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def measure(self):
        pass

class l1StandardDistance(NumericDistance):
    def __init__(self,X_train:np.ndarray,num_feat:list,eps):
        self.num_feat = num_feat
        self.scale = X_train[:,num_feat].std(axis=0, dtype=np.float64)
        self.scale[self.scale < eps] = eps
    def measure(self,X1,X2):
        distance = X2[:,self.num_feat].copy()
        distance = abs(distance - X1[0, self.num_feat]) / self.scale
        distance = np.sum(distance,axis=1)
        return distance

class l1MinMaxDistance(NumericDistance):
    def __init__(self,X_train:np.ndarray,num_feat:list,eps):
        self.num_feat = num_feat
        self.scale = X_train[:, num_feat].max(axis=0) - X_train[:, num_feat].min(axis=0)
        self.scale[self.scale < eps] = eps
    def measure(self,X1,X2):
        distance = X2[:,self.num_feat].copy()
        distance = abs(distance - X1[0, self.num_feat]) / self.scale
        distance = np.sum(distance,axis=1)
        return distance

class l2StandardDistance(NumericDistance):
    def __init__(self,X_train:np.ndarray,num_feat:list,eps):
        self.num_feat = num_feat
        self.scale = X_train[:,num_feat].std(axis=0, dtype=np.float64)
        self.scale[self.scale < eps] = eps
    def measure(self,X1,X2):
        distance = X2[:,self.num_feat].copy()
        distance = ((distance - X1[0, self.num_feat]) / self.scale)**2
        distance = np.sum(distance,axis=1)
        return distance

class l2MinMaxDistance(NumericDistance):
    def __init__(self,X_train:np.ndarray,num_feat:list,eps):
        self.num_feat = num_feat
        self.scale = X_train[:, num_feat].max(axis=0) - X_train[:, num_feat].min(axis=0)
        self.scale[self.scale < eps] = eps
    def measure(self,X1,X2):
        distance = X2[:,self.num_feat].copy()
        distance = ((distance - X1[0, self.num_feat]) / self.scale)**2
        distance = np.sum(distance,axis=1)
        return distance


class HEOM(DistanceMetric):
    def __init__(self, data, numeric_distance:NumericDistance):
        self.data = data
        self.numeric_distance = numeric_distance(data.X_train,data.num_feat,data.eps)
    def measure(self,X1,X2):
        num_distance = self.numeric_distance.measure(X1,X2)
        cat_distance = np.sum(X2[:, self.data.cat_feat] != X1[0, self.data.cat_feat],axis=1)
        distance = num_distance + cat_distance
        return distance


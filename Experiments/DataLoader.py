import pickle


class TabularDataLoader:
    def __init__(self,path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.X_explain = data['X_explain']
        self.cat_feat = data['cat_feat']
        self.con_feat = data['con_feat']
        self.feature_names = data['feature_names']
        del data

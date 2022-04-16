import numpy as np
def HEOM(X1, X2, cat_feat, con_feat, con_range):
    '''

    :param X1:
    :param X2:
    :param cat_feat:
    :param con_feat:
    :param con_range: difference between max and minimum of each continous feature
    :return:
    '''
    distance = X2.copy()
    distance[:, con_feat]= abs(distance[:, con_feat] - X1[0, con_feat]) / con_range
    distance[:, cat_feat]= distance[:, cat_feat] != X1[0, cat_feat]
    distance = np.sum(distance, axis =1)
    return distance
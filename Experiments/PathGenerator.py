from typing import Optional


def path_generator(dataset:str, model:Optional[str]= None, cf: Optional[str]= None):
    paths = {
        'dataset': './Data/{}/data.pkl'.format(dataset),
        'pp': './Data/{}/pp.pkl'.format(dataset),
        'ae': './Data/{}/ae.pkl'.format(dataset),
        'ae_0': './Data/{}/ae_0.pkl'.format(dataset),
        'ae_1': './Data/{}/ae_1.pkl'.format(dataset),
        'model': './Data/{}/{}.pkl'.format(dataset, model),
        'results':'./Data/{}/results/{}_{}.pkl'.format(dataset, model, cf),
        'model_stats': './Data/{}/{}_stats.csv'.format(dataset,model)
    }
    return paths





import matplotlib.pyplot as plt
import pandas as pd

offset = 0.07
def draw(df,ylabel,xlabel,offset,special_offset,path):
    fig, ax = plt.subplots()
    ax.scatter(df.loc[xlabel,:],df.loc[ylabel,:])
    for i in list(df.columns):
        if i in list(special_offset.keys()):
            ax.annotate(i,(df.loc[xlabel,i]+special_offset[i][0], df.loc[ylabel,i]+special_offset[i][1]))
        else:
            ax.annotate(i,(df.loc[xlabel,i]+offset, df.loc[ylabel,i]+offset))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path)
    plt.show()
df = pd.read_csv('./Tables/images_RF.csv', index_col=0)
df = df


special_offset={
    'NICE (prox)':(0.10,-0.20),
    'NICE (spars)':(-0.275,0.13),
    'WIT':(0.1,0.1),
    'GeCo': (-0.35,-0.35),
    'NICE (none)': (-0.75,-0.30),
    'SEDC': (-0.65,-0.08)
}
draw(df,'Proximity','Sparsity',0.08,special_offset= special_offset,path='./tables/images/ProxSpars_RF.png')

special_offset={
    'NICE (spars)':(-0.27,-0.2),
    'NICE (none)':(0.3,-0.05),
    'NICE (plaus)':(-1.27,-0.05),
    'WIT':(-0.62,-0.1),
    'GeCo': (-0.25,0.1),
    'CBR': (0.15,-0.05),
    'DiCE': (-0.53,-0.05)
}

draw(df,'Plausibility','Sparsity',0.08,special_offset= special_offset,path='./tables/images/PlausSpars_RF.png')


special_offset={
    'NICE (none)':(-1.,0.1),
    'NICE (plaus)':(-1.25,-0.15),
    'NICE (spars)':(0.07,-0.12),
    'WIT': (0.1,-0.05),
    'GeCo': (-0.3,-0.2),
    'DiCE': (-0.55,-0.05),
    'CBR': (0.1,-0.05)
}
draw(df,'Plausibility','Proximity',0.08,special_offset= special_offset, path='./tables/images/PlausProx_RF.png')


df = pd.read_csv('./Tables/images_ANN.csv', index_col=0)
df = df


special_offset={
    'NICE (none)':(-1.25,-0.09),
    'NICE (spars)':(0.05,-0.15),
    'WIT': (0.1,-0.05),
    'GeCo': (-0.3,-0.2),
    'DiCE': (-0.55,-0.05),
    'CBR': (0.1,-0.05)
}
draw(df,'Plausibility','Proximity',0.08,special_offset= special_offset, path='./tables/images/PlausProx_ANN.png')

special_offset={
    'NICE (spars)':(0,-0.2),
    'NICE (none)':(0.3,-0.05),
    'WIT':(-0.65,-0.1),
    'GeCo': (-0.2,0.1),
    'CBR': (0.15,-0.05),
    'DiCE': (-0.50,-0.05)
}

draw(df,'Plausibility','Sparsity',0.08,special_offset= special_offset,path='./tables/images/PlausSpars_ANN.png')

special_offset={
    'NICE (prox)':(0.10,-0.20),
    'WIT':(0.1,0.1),
    'GeCo': (-0.35,-0.35),
    'NICE (none)': (0.05,-0.30)
}
draw(df,'Proximity','Sparsity',0.08,special_offset= special_offset,path='./tables/images/ProxSpars_ANN.png')

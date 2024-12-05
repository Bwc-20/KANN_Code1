import itertools
import networkx as nx
import numpy as np
import pandas as pd

from reactome import Reactome, ReactomeNetwork

reactome = Reactome()
names_df = reactome.pathway_names
hierarchy_df = reactome.hierarchy
genes_df = reactome.pathway_genes

print((names_df.head()))
print((hierarchy_df.head()))
print((genes_df.head()))

reactome_net = ReactomeNetwork()
print((reactome_net.info()))

print(('# of root nodes {} , # of terminal nodes {}'.format(len(reactome_net.get_roots()),
                                                           len(reactome_net.get_terminals()))))
print((nx.info(reactome_net.get_completed_tree(n_levels=5))))
print((nx.info(reactome_net.get_completed_network(n_levels=5))))
layers = reactome_net.get_layers(n_levels=3)
print(("层的个数", len(layers)))      ### 在这一共是有4层


def get_map_from_layer(layer_dict):
    '''
    :param layer_dict: dictionary of connections (e.g {'pathway1': ['g1', 'g2', 'g3']}   就是某个通路他对应的基因有哪些！
    :return: dataframe map of layer (index = genes, columns = pathways, , values = 1 if connected; 0 else)    返回的还是基因和通路之间的关系图
    '''
    pathways = list(layer_dict.keys())      ## 他的输入就是一个字典，通路对应着各个基因！
    genes = list(itertools.chain.from_iterable(list(layer_dict.values())))
    genes = list(np.unique(genes))
    df = pd.DataFrame(index=pathways, columns=genes)
    for k, v in list(layer_dict.items()):
        df.loc[k, v] = 1
    df = df.fillna(0)
    return df.T



## 下面是来构造各层通路网络的程序，一共四层，从最开始的基因开始，构建基因层与后面紧邻的那个通路层之间的关系网络，后续逐层往复
for i, layer in enumerate(layers[::-1]):
    mapp = get_map_from_layer(layer)
    if i == 0:
        genes = list(mapp.index)[0:5]
    filter_df = pd.DataFrame(index=genes)
    all = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')    ## 在这进行融合，关键是融合之后呢？？？
    genes = list(mapp.columns)
    print("此时是第几轮！", i)
    print("现在这个map是怎样的！", mapp)
    print("现在这个all又是怎样的！", all)
    print((all.shape))
print("测一下最终的这个all", all)

from os.path import join

import numpy as np
# from data.gmt_reader import GMT
import pandas as pd
from scipy import sparse as sp

from data.data_access import Data
# from data.pathways.pathway_loader import data_dir
from data.gmt_reader import GMT

print("目前所在的文件是pathway_connection.py, 这个文件被人调用没！")



def get_map(data, gene_dict, pathway_dict):     ### 子函数，为下面的生成基因-通路关系图做准备！
    genes = data['gene']
    pathways = data['group'].fillna('')

    n_genes = len(gene_dict)
    n_pathways = len(pathway_dict) + 1
    n = data.shape[0]
    row_index = np.zeros((n,))
    col_index = np.zeros((n,))

    for i, g in enumerate(genes):
        row_index[i] = gene_dict[g]      ### 现在是取行作为基因，列作为通路

    for i, p in enumerate(pathways):
        # print p, type(p)
        if p == '':
            col_index[i] = n_pathways - 1
        else:
            col_index[i] = pathway_dict[p]

    print(('当前是处于pathway_connection.py文件中， 当前输出的是构造的这个图中，基因的数目以及通路的数目：', n_genes, n_pathways))
    print((np.max(col_index)))
    mapp = sp.coo_matrix(([1] * n, (row_index, col_index)), shape=(n_genes, n_pathways))    ## 在这，sp.coo_matrix函数表示生成矩阵
    return mapp     ## 当前的这个函数应该是指生成一个基因与矩阵相关的矩阵图吧？？


def get_dict(listt):
    unique_list = np.unique(listt)
    output_dict = {}
    for i, gene in enumerate(unique_list):
        output_dict[gene] = i
    return output_dict


def get_connection_map(data_params):
    data = Data(**data_params)          ## 目前Data()这个类是指获取训练、测试和验证数据集的
    x, y, info, columns = data.get_data()
    x = pd.DataFrame(x.T, index=columns)

    # print x.head()
    # print x.shape
    # print x.index

    d = GMT()      ## 这个类就代表对GMT格式的文件进行加载处理！
    # pathways = d.load_data ('c4.all.v6.0.entrez.gmt')
    pathways = d.load_data('c4.all.v6.0.symbols.gmt')
    # pathways.to_csv('pathway.csv')

    # print pathways.head()
    # print pathways.shape

    n_genes = len(pathways['gene'].unique())
    n_pathways = len(pathways['group'].unique())
    print(('number of gene {}'.format(n_genes)))
    print(('number of pathways {}'.format(n_pathways)))
    density = pathways.shape[0] / (n_pathways * n_genes + 0.0)
    print(('density {}'.format(density)))

    all = x.merge(pathways, right_on='gene', left_index=True, how='left')

    n_genes = len(all['gene'].unique())
    n_pathways = len(all['group'].unique())
    print(('number of gene {}'.format(n_genes)))
    print(('number of pathways {}'.format(n_pathways)))
    density = all.shape[0] / (n_pathways * n_genes + 0.0)
    print(('density {}'.format(density)))

    # genes = all['gene']
    # pathways = all['group']
    # print all.shape
    gene_dict = get_dict(columns)        ## 根据下载好的，已有的那些基因和通路数据来生成相应的基因与通路字典列表，进而来构造相应的基因与通路关系图！
    pathway_dict = get_dict(pathways['group'])

    # print gene_dict
    # print pathway_dict

    mapp = get_map(all, gene_dict, pathway_dict)

    return mapp


# return: list of genes, list of pathways, list of input shapes, list of gene pathway memberships       这个函数主要是来返回基因-通路之间的关系矩阵的（就是所谓的掩码矩阵）
def get_connections(data_params):
    # data_params = {'type': 'prostate', 'params': {'data_type': ['gene_final_cancer', 'cnv_cancer']}}
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    print("目前所在的文件是pathway_connection.py，(x.shape, y.shape, info.shape, cols.shape)取值", (x.shape, y.shape, info.shape, cols.shape))
    # print cols

    x_df = pd.DataFrame(x, columns=cols)
    print("目前所在的文件是pathway_connection.py，现在的这个x_df的头部是谁！", (x_df.head()))

    genes = cols.get_level_values(0).unique()
    genes_list = []
    input_shapes = []
    for g in genes:
        g_df = x_df.loc[:, g].as_matrix()
        input_shapes.append(g_df.shape[1])
        genes_list.append(g_df)

    # get pathways
    d = GMT()

    pathways = d.load_data_dict(
        'c2.cp.kegg.v6.1.symbols.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    pathway_genes_map = []     ## 来构造通路基因关系矩阵
    for p in list(pathways.keys()):
        common_genes = set(genes).intersection(set(pathways[p]))
        indexes = [i for i, e in enumerate(genes) if e in common_genes]
        print((len(indexes)))
        pathway_genes_map.append(indexes)


    print("目前所在的文件是pathway_connection.py，最后所构造出来的通路-基因关系矩阵是谁！", pathway_genes_map)
    return genes, list(pathways.keys()), input_shapes, pathway_genes_map

    # data_params = {'type': 'prostate', 'params': {'data_type': 'gene_final'}}


# mapp = get_connection_map(data_params)
# # print mapp
# plt.imshow(mapp)
# plt.savefig('genes_pathways')


def get_input_map(cols):       ## 这个函数是得到了输入的那些基因属性与第一层基因他们之间的关系矩阵图  （可是这个位置不应该是全连接吗？？）
    index = cols
    col_index = list(index.labels[0])

    # print row_index, col_index
    n_genes = len(np.unique(col_index))
    n_inputs = len(col_index)

    row_index = list(range(n_inputs))
    n = len(row_index)
    mapp = sp.coo_matrix(([1.] * n, (row_index, col_index)), shape=(n_inputs, n_genes))
    return mapp.toarray()


# params:
# input_list: list of inputs under consideration (e.g. genes)
# filename : a gmt formated file e.g. pathway1 gene1 gene2 gene3     这个gmt文件也就详细说明了通路与基因之间的包含关系
#                                     pathway2 gene4 gene5 gene6
# genes_col: the start index of the gene columns
# shuffle_genes: {True, False}
# return mapp: dataframe with rows =genes and columns = pathways values = 1 or 0 based on the membership of certain gene in the corresponding pathway      这个返回的是基因通路关系图，他的行是基因，列是通路，构造关系矩阵（就是掩码矩阵），如果有关系，则单元格对应位置为1，否则则为0
def get_layer_map(input_list, filename='c2.cp.kegg.v6.1.symbols.gmt', genes_col=1, shuffle_genes=False):
    d = GMT()
    df = d.load_data(filename, genes_col)
    print(('目前所在的文件是pathway_connection.py， map # ones  before join {}'.format(df.shape[0])))

    df['value'] = 1
    mapp = pd.pivot_table(df, values='value', index='gene', columns='group', aggfunc=np.sum)
    mapp = mapp.fillna(0)
    # print mapp.head()
    print(('目前所在的文件是pathway_connection.py，第二位，map # ones  before join {}'.format(np.sum(mapp.as_matrix()))))
    cols_df = pd.DataFrame(index=input_list)
    mapp = cols_df.merge(mapp, right_index=True, left_index=True, how='left')
    mapp = mapp.fillna(0)
    mapp.to_csv(join(data_dir, filename + '.csv'))
    genes = mapp.index
    pathways = mapp.columns
    print(('目前所在的文件是pathway_connection.py，pathways', pathways))
    # print mapp.head()

    mapp = mapp.as_matrix()
    print(('目前所在的文件是pathway_connection.py，filename', filename))
    print(('map # ones  after join {}'.format(np.sum(mapp))))

    if shuffle_genes:
        # print mapp[0:10, 0:10]
        # print sum(mapp)
        # logging.info('shuffling the map')
        # mapp = mapp.T
        # np.random.shuffle(mapp)
        # mapp= mapp.T
        # print mapp[0:10, 0:10]
        # print sum(mapp)
        ones_ratio = np.sum(mapp) / np.prod(mapp.shape)       ## 在这np.prod函数用来计算所有元素的乘积

        mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])

    return mapp, genes, pathways


def get_gene_map(input_list, filename='c2.cp.kegg.v6.1.symbols.gmt', genes_col=1, shuffle_genes=False):
    d = GMT()
    # returns a pthway dataframe   返回与当前通路有关系的那些基因
    # e.g.  pathway1 gene1
    #       pathway1 gene2
    #       pathway1 gene3
    pathways = d.load_data(filename,
                           genes_col)  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('PathwayCommons9.kegg.hgnc.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('ReactomePathways.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('ReactomePathways_terminal.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('h.all.v6.1.symbols.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('c2.all.v6.1.symbols.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('c2.cp.reactome.v6.1.symbols.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('ReactomePathways_roots.gmt')  # KEGG pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('PathwayCommons9.All.hgnc.gmt')  # http://www.pathwaycommons.org/archives/PC2/v9/PathwayCommons9.All.hgnc.gmt.gz
    # pathways = d.load_data('c5.bp.v6.1.symbols.gmt')  # Go Biological pathways downloaded from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # pathways = d.load_data('c4.cm.v6.1.symbols.gmt')  # Cancer modules from http://software.broadinstitute.org/gsea/msigdb/collections.jsp
    # print 'pathways'
    # print pathways.head()
    print("目前所在的文件是pathway_connection.py，目前的这个文件名是谁！", filename)
    cols_df = pd.DataFrame(index=input_list)
    # print 'cols_df'
    # print cols_df.shape
    # print cols_df.head()
    # print 'pathways'
    # print pathways.shape
    # print pathways.head()
    print(('map # ones  before join {}'.format(pathways.shape[0])))

    # limit the rows to the input_lis only
    all = cols_df.merge(pathways, right_on='gene', left_index=True, how='left')
    # print 'joined'
    # print all.shape
    # print all.head()
    print(('UNK pathway', sum(pd.isnull(all['group']))))

    # ind = pd.isnull(all['group'])
    # print ind
    # print 'Known pathway', sum(~pd.isnull(all['group']))

    all = all.fillna('UNK')
    # print 'UNK genes', len(ind), sum(ind)
    # print all.loc[ind, :]
    # all = all.dropna()

    all = all.set_index(['gene', 'group'])
    # print all.head()
    index = all.index

    col_index = list(index.labels[1])
    row_index = list(index.labels[0])

    # print row_index, col_index
    n_pathways = len(np.unique(col_index))
    n_genes = len(np.unique(row_index))

    # row_index = range(n_inputs)
    n = len(row_index)
    # print 'pathways', [index.levels[1][i] for i in col_index]
    # print 'pathways',
    # for p in index.levels[1]:
    #     print p

    mapp = sp.coo_matrix(([1.] * n, (row_index, col_index)), shape=(n_genes, n_pathways))

    pathways = list(index.levels[1])
    genes = index.levels[0]
    mapp = mapp.toarray()

    print(('map # ones  after join {}'.format(np.sum(mapp))))

    if shuffle_genes:
        # print mapp[0:10, 0:10]
        # print sum(mapp)
        # logging.info('shuffling the map')
        # mapp = mapp.T
        # np.random.shuffle(mapp)
        # mapp= mapp.T
        # print mapp[0:10, 0:10]
        # print sum(mapp)

        ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
        mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])

    # print pathways

    # if 'UNK' in pathways:
    #     ind= list(pathways).index('UNK')
    #     mapp = np.delete(mapp, ind, 1)
    #     pathways.remove('UNK')

    map_df = pd.DataFrame(mapp, index=genes, columns=pathways)
    map_df.to_csv(join(data_dir, filename + '.csv'))
    return mapp, genes, pathways
    # return map_df

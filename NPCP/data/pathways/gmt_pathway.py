import logging
import numpy as np
import pandas as pd
from data.gmt_reader import GMT

## 从KEGG数据库中来获取所需要的那批通路数据    这个函数他所返回的就是一个二维表，行是基因，列是通路，中间具体的单元格就表示当前的这个基因与通路之间是否有关系，有关系就是1，没有关系就是0
### 构造通路与基因之间的关系矩阵
def get_KEGG_map(input_list, filename='c2.cp.kegg.v6.1.symbols.gmt', genes_col=1, shuffle_genes=False):
    '''
    :param input_list: list of inputs under consideration (e.g. genes)
    :param filename: a gmt formated file e.g. pathway1 gene1 gene2 gene3
#                                     pathway2 gene4 gene5 gene6
    :param genes_col: the start index of the gene columns
    :param shuffle_genes: {True, False}
    :return: dataframe with rows =genes and columns = pathways values = 1 or 0 based on the membership of certain gene in the corresponding pathway
    '''
    d = GMT()
    df = d.load_data(filename, genes_col)
    df['value'] = 1
    mapp = pd.pivot_table(df, values='value', index='gene', columns='group', aggfunc=np.sum)
    mapp = mapp.fillna(0)
    cols_df = pd.DataFrame(index=input_list)
    mapp = cols_df.merge(mapp, right_index=True, left_index=True, how='left')
    mapp = mapp.fillna(0)
    genes = mapp.index
    pathways = mapp.columns
    mapp = mapp.values
    print("上天护佑！当前是在data/pathways/gmt_pathway.py 文件，这个映射关系的具体数值为：", mapp)

    if shuffle_genes:
        logging.info('当前是data/pathways/gmt_pathway.py 文件   shuffling')
        ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
        logging.info('ones_ratio {}'.format(ones_ratio))
        mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
        logging.info('random map ones_ratio {}'.format(ones_ratio))
    return mapp, genes, pathways

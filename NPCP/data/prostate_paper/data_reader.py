import logging

import numpy as np
import pandas as pd
# from CeShi.OtherTest.ModelParamSave.ModelParam_Two import GetOriGeneData

from config_path import *

data_path = DATA_PATH
processed_path = join(PROSTATE_DATA_PATH, 'processed')

# use this one
gene_final_no_silent_no_intron = 'P1000_final_analysis_set_cross__no_silent_no_introns_not_from_the_paper.csv'
cnv_filename = 'P1000_data_CNA_paper.csv'
response_filename = 'response_paper.csv'
gene_important_mutations_only = 'P1000_final_analysis_set_cross_important_only.csv'
gene_important_mutations_only_plus_hotspots = 'P1000_final_analysis_set_cross_important_only_plus_hotspots.csv'
gene_hotspots = 'P1000_final_analysis_set_cross_hotspots.csv'
gene_truncating_mutations_only = 'P1000_final_analysis_set_cross_truncating_only.csv'
gene_expression = 'P1000_adjusted_TPM.csv'
fusions_filename = 'p1000_onco_ets_fusions.csv'
cnv_burden_filename = 'P1000_data_CNA_burden.csv'
fusions_genes_filename = 'fusion_genes.csv'

cached_data = {}


def load_data(filename, selected_genes=None):
    filename = join(processed_path, filename)
    logging.info('loading data from %s,' % filename)
    if filename in cached_data:
        logging.info('loading from memory cached_data')
        data = cached_data[filename]
    else:
        data = pd.read_csv(filename, index_col=0)
        cached_data[filename] = data
    logging.info("当前所导入的数据的形状为：(%d, %d)" % data.shape)

    if 'response' in cached_data:
        logging.info('目前处于data_reader.py文件，loading from memory cached_data')
        labels = cached_data['response']
    else:
        labels = get_response()
        cached_data['response'] = labels

    # remove all zeros columns (note the column may be added again later if another feature type belongs to the same gene has non-zero entries      请注意，如果属于同一基因的另一种要素类型具有非零条目，则稍后可能会再次添加该列).
    # zero_cols = data.sum(axis=0) == 0
    # data = data.loc[:, ~zero_cols]

    # join with the labels
    all = data.join(labels, how='inner')
    all = all[~all['response'].isnull()]         ### 选择所有response列值不为null的行  在这主要是担心它所对应的那个类别标签为空！

    response = all['response']           ### 现在就获取了当前这批数据的标签
    samples = all.index

    del all['response']
    x = all
    genes = all.columns

    print("现在这批数据的标签是谁！All是谁！", labels)


    if not selected_genes is None:
        # genes = GetOriGeneData()
        # intersect = set.intersection(genes, selected_genes)               ### 现在来求交集，求两个集合中都包含的元素！
        intersect = set.intersection(set(genes), selected_genes)        ### 现在来求交集，求两个集合中都包含的元素！
        # intersect = GetOriGeneData()
        print("现在的这个intersect是谁！", len(intersect), len(selected_genes), len(set(genes)), intersect)
        # print("现在变成集合的基因是怎样的！", set(genes))
        if len(intersect) < len(selected_genes):
            # raise Exception('wrong gene')
            logging.warning('目前处于data_reader.py文件，目前来读取数据  some genes dont exist in the original data set')       ### 此时说明所选择的那些基因有一些并不在原始的那个基因数据集中，因为在求交集后小于了原始的长度
        x = x.loc[:, intersect]       ### 把各个样本中属于交集的那些基因给选择出来！
        # print("重置之前的这个基因情况是怎样的！", genes)
        genes = intersect
        # print("重置之前的这个基因情况是怎样的！", genes)
        # print("重置之前的这个基因情况是怎样的！", x)
    logging.info('目前处于data/prostate_paper/data_reader.py文件，loaded data %d samples, %d variables, %d responses ' % (x.shape[0], x.shape[1], response.shape[0]))
    logging.info("目前处于data/prostate_paper/data_reader.py文件， 目前所读取的基因长度为：%d " % len(genes))         ### 目前这个是对原始读出来的那些基因进行进一步的滤除选择，去掉那些突变为空的基因！
    return x, response, samples, genes


def load_TMB(filename=gene_final_no_silent_no_intron):
    x, response, samples, genes = load_data(filename)
    x = np.sum(x, axis=1)
    x = np.array(x)
    x = np.log(1. + x)
    n = x.shape[0]
    response = response.values.reshape((n, 1))
    samples = np.array(samples)
    cols = np.array(['TMB'])
    return x, response, samples, cols


def load_CNV_burden(filename=gene_final_no_silent_no_intron):
    x, response, samples, genes = load_data(filename)
    x = np.sum(x, axis=1)
    x = np.array(x)
    x = np.log(1. + x)
    n = x.shape[0]
    response = response.values.reshape((n, 1))
    samples = np.array(samples)
    cols = np.array(['TMB'])
    return x, response, samples, cols


def load_data_type(data_type='gene', cnv_levels=5, cnv_filter_single_event=True, mut_binary=False, selected_genes=None):
    logging.info('loading {}'.format(data_type))
    if data_type == 'TMB':
        x, response, info, genes = load_TMB(gene_important_mutations_only)
    if data_type == 'mut_no_silent_no_intron':
        x, response, info, genes = load_data(gene_final_no_silent_no_intron, selected_genes)
        if mut_binary:
            logging.info('mut_binary = True')
            x[x > 1.] = 1.

    if data_type == 'mut_important':
        x, response, info, genes = load_data(gene_important_mutations_only, selected_genes)
        if mut_binary:
            logging.info('mut_binary = True')
            x[x > 1.] = 1.

    if data_type == 'mut_important_plus_hotspots':
        x, response, info, genes = load_data(gene_important_mutations_only_plus_hotspots, selected_genes)

    if data_type == 'mut_hotspots':
        x, response, info, genes = load_data(gene_hotspots, selected_genes)

    if data_type == 'truncating_mut':
        x, response, info, genes = load_data(gene_truncating_mutations_only, selected_genes)
        if mut_binary:
            logging.info('mut_binary = True')
            x[x > 1.] = 1.

    if data_type == 'gene_final_no_silent':
        x, response, info, genes = load_data(gene_final_no_silent, selected_genes)
    if data_type == 'cnv':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        if cnv_levels == 3:
            logging.info('cnv_levels = 3')
            # remove single amplification and single delteion, they are usually noisey
            if cnv_levels == 3:
                if cnv_filter_single_event:
                    x[x == -1.] = 0.0
                    x[x == -2.] = 1.0
                    x[x == 1.] = 0.0
                    x[x == 2.] = 1.0
                else:
                    x[x < 0.] = -1.
                    x[x > 0.] = 1.

    if data_type == 'cnv_del':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x >= 0.0] = 0.
        if cnv_levels == 3:
            if cnv_filter_single_event:
                x[x == -1.] = 0.0
                x[x == -2.] = 1.0
            else:
                x[x < 0.0] = 1.0
        else:  # cnv == 5 , use everything
            x[x == -1.] = 0.5
            x[x == -2.] = 1.0

    if data_type == 'cnv_amp':           ### 目前走的是这个位置！
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        print("腓特烈大帝！！现在这个genes是怎样的！", genes)
        x[x <= 0.0] = 0.
        if cnv_levels == 3:
            if cnv_filter_single_event:
                x[x == 1.0] = 0.0
                x[x == 2.0] = 1.0
            else:
                x[x > 0.0] = 1.0
        else:  # cnv == 5 , use everything
            x[x == 1.] = 0.5
            x[x == 2.] = 1.0

    if data_type == 'cnv_single_del':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == -1.] = 1.0
        x[x != -1.] = 0.0
    if data_type == 'cnv_single_amp':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == 1.] = 1.0
        x[x != 1.] = 0.0
    if data_type == 'cnv_high_amp':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == 2.] = 1.0
        x[x != 2.] = 0.0
    if data_type == 'cnv_deep_del':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == -2.] = 1.0
        x[x != -2.] = 0.0

    if data_type == 'gene_expression':
        x, response, info, genes = load_data(gene_expression, selected_genes)

    if data_type == 'fusions':
        x, response, info, genes = load_data(fusions_filename, None)

    if data_type == 'cnv_burden':
        x, response, info, genes = load_data(cnv_burden_filename, None)
        # x.loc[:, :] = 0.

    if data_type == 'fusion_genes':
        x, response, info, genes = load_data(fusions_genes_filename, selected_genes)
        # x.loc[:,:]=0.

    return x, response, info, genes


def get_response():
    logging.info('loading response from %s' % response_filename)
    labels = pd.read_csv(join(processed_path, response_filename))
    labels = labels.set_index('id')
    return labels


# complete_features: make sure all the data_types have the same set of features_processing (genes)
def combine(x_list, y_list, rows_list, cols_list, data_type_list, combine_type, use_coding_genes_only=False):
    cols_list_set = [set(list(c)) for c in cols_list]         ### 现在这个就是那个列标题就是那一堆的基因！

    if combine_type == 'intersection':
        cols = set.intersection(*cols_list_set)
    else:
        cols = set.union(*cols_list_set)

    if use_coding_genes_only:
        f = join(data_path, 'genes/HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt')
        coding_genes_df = pd.read_csv(f, sep='\t', header=None)
        coding_genes_df.columns = ['chr', 'start', 'end', 'name']
        coding_genes = set(coding_genes_df['name'].unique())
        cols = cols.intersection(coding_genes)

    # the unique (super) set of genes
    all_cols = list(cols)

    all_cols_df = pd.DataFrame(index=all_cols)

    df_list = []
    for x, y, r, c in zip(x_list, y_list, rows_list, cols_list):
        df = pd.DataFrame(x, columns=c, index=r)
        df = df.T.join(all_cols_df, how='right')
        df = df.T
        df = df.fillna(0)
        df_list.append(df)

    ### 将三类数据按行（样本）进行合并！
    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1, )               ### 将一个包含多个数据集的列表df_list进行拼接，并按照指定的data_type_list作为多级索引，在水平方向（列）进行合并。通过设置join='inner'参数，只保留具有相同索引的行。最后返回合并后的新数据集。   返回具有相同行标题的那些行！

    # put genes on the first level and then the data type
    all_data = all_data.swaplevel(i=0, j=1, axis=1)        ## 这句代码表示交换all_data数据集的多级索引的第一个和第二个级别的位置。该操作会在数据集的列级别上进行交换，通过设置axis=1参数来指定操作在列上进行。最后返回交换后的新数据集all_data。

    # order the columns based on genes
    order = all_data.columns.levels[0]
    all_data = all_data.reindex(columns=order, level=0)

    # print("目前是在data/prostate_paper/data_reader.py 文件中，现在的这个all_data情况是怎样的！", all_data)
    # ### 下面这个是来进行测试！将这个all_data数据保存到一定的csv文件中！看一下这个数据情况
    # all_data.to_csv("../CeShi/OtherTest/Other/GeneInput.csv")
    # print("现在这个数据保存成功了！！")


    x = all_data.values

    reordering_df = pd.DataFrame(index=all_data.index)
    y = reordering_df.join(y, how='left')

    y = y.values
    cols = all_data.columns
    rows = all_data.index
    logging.info(
        'After combining, loaded data %d samples, %d variables, %d responses ' % (x.shape[0], x.shape[1], y.shape[0]))
    print("目前是在 data/prostate_paper/data_reader.py 文件中， 现在这个也就是相当于将导入进来的这三项数据进行融合！看看融合之后的数据情况是怎样的！", len(x), len(x[0]), x)
    # print("那么现在提取出来的这个值的情况是怎样的！", cols, rows)

    return x, y, rows, cols


def split_cnv(x_df):
    genes = x_df.columns.levels[0]
    x_df.rename(columns={'cnv': 'CNA_amplification'}, inplace=True)
    for g in genes:
        x_df[g, 'CNA_deletion'] = x_df[g, 'CNA_amplification'].replace({-1.0: 0.5, -2.0: 1.0})
        x_df[g, 'CNA_amplification'] = x_df[g, 'CNA_amplification'].replace({1.0: 0.5, 2.0: 1.0})
    x_df = x_df.reindex(columns=genes, level=0)
    return x_df


class ProstateDataPaper():

    def __init__(self, data_type='mut', account_for_data_type=None, cnv_levels=5,
                 cnv_filter_single_event=True, mut_binary=False,
                 selected_genes=None, combine_type='intersection',
                 use_coding_genes_only=False, drop_AR=False,
                 balanced_data=False, cnv_split=False,
                 shuffle=False, selected_samples=None, training_split=0):

        self.training_split = training_split
        if not selected_genes is None:
            if type(selected_genes) == list:
                # list of genes
                selected_genes = selected_genes
            else:
                # file that will be used to load list of genes
                selected_genes_file = join(data_path, 'genes')
                selected_genes_file = join(selected_genes_file, selected_genes)
                print(("目前是在data/prostate_paper/data_reader.py文件，  现在测试一下目前所选的文件是谁！", selected_genes_file))       ### 在这所读取的基因大概约为14658各左右！
                df = pd.read_csv(selected_genes_file, header=0)
                selected_genes = list(df['genes'])

        if type(data_type) == list:
            x_list = []
            y_list = []
            rows_list = []
            cols_list = []

            for t in data_type:
                x, y, rows, cols = load_data_type(t, cnv_levels, cnv_filter_single_event, mut_binary, selected_genes)
                x_list.append(x), y_list.append(y), rows_list.append(rows), cols_list.append(cols)
            x, y, rows, cols = combine(x_list, y_list, rows_list, cols_list, data_type, combine_type,
                                       use_coding_genes_only)
            x = pd.DataFrame(x, columns=cols)
            # print("现在处理之后的这个数据情况是谁！！", x)
            # print("现在位于data/prostate_paper/data_reader.py文件中，看看这个位置走进来了吗！！444", cols)            ### 现在走的是这个位置，现在的这个cols是一个n行，两列的数据表，第一列是基因，第二列是突变、CNV扩增、CNV缺失

        else:
            x, y, rows, cols = load_data_type(data_type, cnv_levels, cnv_filter_single_event, mut_binary,
                                              selected_genes)                  ### 这块没走进来！
            print("看看这个位置走进来了吗！！555", cols)

        if drop_AR:

            print("测试一下看看目前这个数据走进来没！", drop_AR)        ### 目前！这里是没走进来！！
            data_types = x.columns.levels[1].unique()
            ind = True
            if 'cnv' in data_types:
                ind = x[('AR', 'cnv')] <= 0.
            elif 'cnv_amp' in data_types:
                ind = x[('AR', 'cnv_amp')] <= 0.

            if 'mut_important' in data_types:
                ind2 = (x[('AR', 'mut_important')] < 1.)
                ind = ind & ind2
            x = x.loc[ind,]
            y = y[ind]
            rows = rows[ind]

        if cnv_split:
            x = split_cnv(x)
            print("这里CNV分割之后的这个数据情况是怎样的！", x)               ## 现在这里没走进来！

        if type(x) == pd.DataFrame:
            x = x.values

        if balanced_data:             ### 这块没走进来！
            pos_ind = np.where(y == 1.)[0]
            neg_ind = np.where(y == 0.)[0]

            n_pos = pos_ind.shape[0]
            n_neg = neg_ind.shape[0]
            n = min(n_pos, n_neg)

            pos_ind = np.random.choice(pos_ind, size=n, replace=False)
            neg_ind = np.random.choice(neg_ind, size=n, replace=False)

            ind = np.sort(np.concatenate([pos_ind, neg_ind]))

            y = y[ind]
            x = x[ind,]
            rows = rows[ind]
            print("现在这个ind以及x数据是怎样的！", ind, x)

        if shuffle:             ### 这块没走进来！
            n = x.shape[0]
            ind = np.arange(n)
            np.random.shuffle(ind)
            x = x[ind, :]
            y = y[ind, :]
            rows = rows[ind]
            print("看看这个位置走进来了吗！！333 这一位置被赋予重要意义！", cols, rows, ind)

        if account_for_data_type is not None:          ### 这块没走进来！
            x_genomics = pd.DataFrame(x, columns=cols, index=rows)
            y_genomics = pd.DataFrame(y, index=rows, columns=['response'])
            x_list = []
            y_list = []
            rows_list = []
            cols_list = []
            for t in account_for_data_type:
                x_, y_, rows_, cols_ = load_data_type(t, cnv_levels, cnv_filter_single_event, mut_binary,
                                                      selected_genes)
                x_df = pd.DataFrame(x_, columns=cols_, index=rows_)
                x_list.append(x_df), y_list.append(y_), rows_list.append(rows_), cols_list.append(cols_)

            x_account_for = pd.concat(x_list, keys=account_for_data_type, join='inner', axis=1)
            x_all = pd.concat([x_genomics, x_account_for], keys=['genomics', 'account_for'], join='inner', axis=1)

            common_samples = set(rows).intersection(x_all.index)
            x_all = x_all.loc[common_samples, :]
            y = y_genomics.loc[common_samples, :]

            y = y['response'].values
            x = x_all.values
            cols = x_all.columns
            rows = x_all.index
            print("看看这个位置走进来了吗！！222", cols)

        if selected_samples is not None:          ### 这块没走进来！
            selected_samples_file = join(processed_path, selected_samples)
            df = pd.read_csv(selected_samples_file, header=0)
            selected_samples_list = list(df['Tumor_Sample_Barcode'])

            x = pd.DataFrame(x, columns=cols, index=rows)
            y = pd.DataFrame(y, index=rows, columns=['response'])

            x = x.loc[selected_samples_list, :]
            y = y.loc[selected_samples_list, :]
            rows = x.index
            cols = x.columns
            y = y['response'].values
            x = x.values
            print("看看这个位置走进来了吗！！111", cols)

        print("测试一下当前这个函数哪些会走到哪些走不到！", selected_samples is not None, account_for_data_type is not None, shuffle, balanced_data, cnv_split, type(x) == pd.DataFrame)              ### 目前这些是都走不到
        self.x = x
        self.y = y
        self.info = rows          ### 现在他就代表着样本编号
        self.columns = cols       ### 它代表着基因名字

    def get_data(self):
        return self.x, self.y, self.info, self.columns

    def get_train_validate_test(self):
        info = self.info
        x = self.x
        y = self.y
        columns = self.columns
        splits_path = join(PROSTATE_DATA_PATH, 'splits')

        training_file = 'training_set_{}.csv'.format(self.training_split)
        print("目前是在data/prostate_paper/data_reader.py文件中，当前的这个training_split是谁，以及最终的文件是谁", self.training_split, training_file)      ## training_set_0.csv
        ## 现在下面这块就是来从原始数据文件中读取数据的(数据并没有均衡！)
        training_set = pd.read_csv(join(splits_path, training_file))
        validation_set = pd.read_csv(join(splits_path, 'validation_set.csv'))
        testing_set = pd.read_csv(join(splits_path, 'test_set.csv'))

        print("现在传进来之前这个x数据是谁！", len(x), len(x[0]), len(columns))

        # ## 现在下面进行一下测试，读取数据的读那些平衡的测试数据
        # print("目前是在data/prostate_paper/data_reader.py文件中，注意！注意！目前所读的数据跟原来的数据不太一样，他是正负样本均衡！")
        # splits_path_Ceshi = join(splits_path, 'BalanceData_Ceshi')
        # training_set = pd.read_csv(join(splits_path_Ceshi, training_file))
        # validation_set = pd.read_csv(join(splits_path_Ceshi, 'validation_set.csv'))
        # testing_set = pd.read_csv(join(splits_path_Ceshi, 'test_set.csv'))



        # ## 现在下面进行一下测试，读取数据的读那些平衡的测试数据
        # print("目前是在data/prostate_paper/data_reader.py文件中，注意！注意！目前所读的数据跟原来的数据不太一样，他是正样本比负样本为2:1！")           ## PosNeg2B1Data_Ceshi
        # splits_path_Ceshi = join(splits_path, 'PosNeg2B1Data_Ceshi')
        # training_set = pd.read_csv(join(splits_path_Ceshi, training_file))
        # validation_set = pd.read_csv(join(splits_path_Ceshi, 'validation_set.csv'))
        # testing_set = pd.read_csv(join(splits_path_Ceshi, 'test_set.csv'))



        info_train = list(set(info).intersection(training_set.id))           ### 目前所构造的数据中已有的样本与给定的训练测试验证数据集求交集！
        info_validate = list(set(info).intersection(validation_set.id))
        info_test = list(set(info).intersection(testing_set.id))

        ind_train = info.isin(info_train)
        ind_validate = info.isin(info_validate)
        ind_test = info.isin(info_test)

        x_train = x[ind_train]
        x_test = x[ind_test]
        x_validate = x[ind_validate]

        # print("测试一下看这几个数据的情况！", info_train, ind_train)

        y_train = y[ind_train]
        y_test = y[ind_test]
        y_validate = y[ind_validate]

        info_train = info[ind_train]
        info_test = info[ind_test]
        info_validate = info[ind_validate]

        print("现在的这个文件是data/prostate_paper/data_reader.py， 现在的这个x_train是谁！", len(x_train), len(x_train[0]))

        return x_train, x_validate, x_test, y_train, y_validate, y_test, info_train.copy(), info_validate, info_test.copy(), columns

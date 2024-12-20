import pandas as pd
from data.data_access import Data

selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes_and_memebr_of_reactome.csv'           ##    这个文件中一共6054个基因，只有一列就是基因的名字！！从这个文件中来选择基因！ 这个算是从Reactome数据集中来选择对应的基因了！
selected_samples = 'samples_with_fusion_data.csv'
data_params = {'id': 'ALL', 'type': 'prostate_paper',
               'params': {
                   'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
                   'account_for_data_type': ['TMB'],
                   'drop_AR': False,
                   'cnv_levels': 3,
                   'mut_binary': False,
                   'balanced_data': False,
                   'combine_type': 'union',  # intersection
                   'use_coding_genes_only': True,
                   'selected_genes': selected_genes,
                   'selected_samples': None,
                   'training_split': 0,
               }
               }

data_adapter = Data(**data_params)
x, y, info, columns = data_adapter.get_data()

print((x.shape, y.shape, len(columns), len(info)))

x_train, x_test, y_train, y_test, info_train, info_test, columns = data_adapter.get_train_test()
x_train_df = pd.DataFrame(x_train, columns=columns, index=info_train)

print("现在是data_access_test.py文件 测试一下目前的columns.levels", (columns.levels))
# print("目前这个col是个什么玩意！", columns)
print("现在是data_access_test.py文件 测试一下目前的训练数据和测试数据的形状", (x_train.shape, x_test.shape, y_train.shape, y_test.shape))
print("现在是data_access_test.py文件 测试一下目前的训练数据求和的情况", (x_train.sum().sum()))

x, y, info, columns = data_adapter.get_data()
x_df = pd.DataFrame(x, columns=columns, index=info)
print("现在是data_access_test.py文件 测试一下目前转化为DF格式的X数据", (x_df.shape))
print((x_df.sum().sum()))


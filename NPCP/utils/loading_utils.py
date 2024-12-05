import os
from os.path import join
import numpy as np
import yaml
from data.data_access import Data
from model import model_factory

### 此文件主要是用来加载模型的参数、训练测试数据以及具体模型信息等数据
class DataModelLoader():
    def __init__(self, params_file):
        self.dir_path = os.path.dirname(os.path.realpath(params_file))
        model_parmas, data_parmas = self.load_parmas(params_file)
        data_reader = Data(**data_parmas)
        self.model = None
        x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data_reader.get_train_validate_test()

        self.x_train = x_train
        self.x_test = np.concatenate([x_validate_, x_test_], axis=0)

        self.y_train = y_train
        self.y_test = np.concatenate([y_validate_, y_test_], axis=0)

        self.info_train = info_train
        self.info_test = list(info_validate_) + list(info_test_)
        self.columns = cols

    def get_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test, self.info_train, self.info_test, self.columns

    def get_model(self, model_name='P-net_params.yml'):
        # if self.model is None:
        self.model = self.load_model(self.dir_path, model_name)
        return self.model

    def load_model(self, model_dir_, model_name):
        # 1 - load architecture
        params_filename = join(model_dir_, model_name + '_params.yml')
        stream = open(params_filename, 'r')
        params = yaml.load(stream, Loader=yaml.FullLoader)
        # print params
        # fs_model = model_factory.get_model(params['model_params'][0])
        print("现在导入进来的这个参数是谁！", params)
        # fs_model = model_factory.get_model(params['model_params'])
        fs_model = model_factory.get_model(params['models'])                  ### 现在这个位置为测试部分！
        # 2 -compile model and load weights (link weights)
        weights_file = join(model_dir_, 'fs\{}_0.h5'.format(model_name))
        model = fs_model.load_model(weights_file)
        return model

    def load_parmas(self, params_filename):
        print("现在，到这的这个文件是谁！", params_filename)
        stream = open(params_filename, 'r')
        params = yaml.load(stream, Loader=yaml.UnsafeLoader)
        print("现在导入进来的这个参数情况是怎样的！", params)
        # model_parmas = params['model_params']
        print("测试一下参数的这一项！", params['models'])
        model_parmas = params['models']
        # model_parmas = params['models']['model_params']

        # data_parmas = params['data_params']
        data_parmas = params['data']
        # data_parmas = params['models']['model_params']['data_params']
        return model_parmas, data_parmas







# {'data':
#      {'id': 'ALL',
#       'params':
#           {'balanced_data': False, 'cnv_levels': 3, 'combine_type': 'union', 'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
#            'drop_AR': False, 'mut_binary': True, 'selected_genes': 'tcga_prostate_expressed_genes_and_cancer_genes.csv', 'training_split': 0, 'use_coding_genes_only': True
#            }, 'type': 'prostate_paper'
#       },
#  'models':
#      {'id': 'P-net_ALL',
#       'params': {'WeightOutputLayers': [0, 1], 'build_fn': <function build_pnet2 at 0x000001EDAF81F950>, 'feature_importance': 'deepexplain_grad*input',
#                  'fitting_params': {'batch_size': 50, 'class_weight': 'auto', 'debug': False, 'early_stop': False, 'epoch': 300, 'lr': 0.001, 'max_f1': True,
#                                     'monitor': 'val_o6_f1', 'n_outputs': 6, 'prediction_output': 'average', 'reduce_lr': False,
#                                     'reduce_lr_after_nepochs': {'drop': 0.25, 'epochs_drop': 50}, 'samples_per_epoch': 10, 'save_gradient': False, 'save_name': 'pnet',
#                                     'select_best_model': False, 'shuffle': True, 'verbose': 2
#                                     },
#                  'gradients': None, 'gradients_Flag': False,
#                  'model_params':
#                       {'activation': 'tanh', 'add_unk_genes': False, 'attention': False,
#                        'data_params': {'id': 'ALL',
#                                        'params': {'balanced_data': False, 'cnv_levels': 3, 'combine_type': 'union', 'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
#                                                   'drop_AR': False, 'mut_binary': True, 'selected_genes': 'tcga_prostate_expressed_genes_and_cancer_genes.csv', 'training_split': 0,
#                                                   'use_coding_genes_only': True
#                                                   }, 'type': 'prostate_paper'
#                                        },
#                        'dropout': [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 'dropout_testing': False, 'kernel_initializer': 'lecun_uniform', 'loss_weights': [2, 7, 20, 54, 148, 400],
#                        'n_hidden_layers': 5, 'optimizer': 'Adam', 'shuffle_genes': False, 'use_bias': True, 'w_reg': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
#                        'w_reg_outcomes': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]}},
#         'type': 'nn'},
#   'pipeline': {'params': {'n_splits': 5, 'save_train': True}, 'type': 'crossvalidation'},
#   'pre': {'type': None},
#   'scores': '{"accuracy":{"0":0.8646616541,"1":0.7669172932,"2":0.7894736842,"3":0.8571428571,"4":0.8409090909},"precision":{"0":0.8139534884,"1":0.6346153846,"2":0.7179487179,"3":0.7954545455,"4":0.7222222222},"auc":{"0":0.9366161616,"1":0.8752525253,"2":0.8742424242,"3":0.8911616162,"4":0.9060025543},"f1":{"0":0.7954545455,"1":0.6804123711,"2":0.6666666667,"3":0.7865168539,"4":0.7878787879},"aupr":{"0":0.9149588433,"1":0.8068632807,"2":0.769676049,"3":0.7699931727,"4":0.7887652113},"recall":{"0":0.7777777778,"1":0.7333333333,"2":0.6222222222,"3":0.7777777778,"4":0.8666666667}}', 'scores_mean': '{"accuracy":0.8238209159,"precision":0.7368388717,"auc":0.8966550563,"f1":0.743385845,"aupr":0.8100513114,"recall":0.7555555556}', 'scores_std': '{"accuracy":0.0432668406,"precision":0.0714188123,"auc":0.0258594178,"f1":0.0640362128,"aupr":0.0606293414,"recall":0.0888888889}'}











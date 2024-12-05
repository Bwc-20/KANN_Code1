from model.builders.prostate_models import build_pnet2
from model.builders.prostate_models import build_dense

task = 'classification_binary'
selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'
data_base = {'id': 'ALL', 'type': 'prostate_paper',
             'params': {
                 'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
                 'drop_AR': False,
                 'cnv_levels': 3,
                 'mut_binary': True,
                 'balanced_data': False,
                 'combine_type': 'union',  # intersection
                 'use_coding_genes_only': True,
                 'selected_genes': selected_genes,
                 'training_split': 0,
             }
             }
data = [data_base]

n_hidden_layers = 5     ### 设计的中间的隐藏层
base_dropout = 0.5
wregs = [0.001] * 7
loss_weights = [2, 7, 20, 54, 148, 400]       ### 因为除了输入层之外，剩下的六层每一层都是要得到一个预测结果的，最后将这六个结果进行综合来看  所以每层结果都要乘以相应的权重再进行求和，越是后面几层，他的权重就要越大！       中间调参记录：[2, 7, 20, 54, 148, 400]
wreg_outcomes = [0.01] * 6                     ### 现在在这形成[0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
pre = {'type': None}

class_weight = {0: 1, 1: 1}


## 现在下面的这个是不加注意力机制的这个模型的参数
nn_pathway = {
    'type': 'nn',
    'id': 'P-net',
    'params':
        {
            'build_fn': build_pnet2,          ### 此参数格外重要，指明模型具体构建的手段
            'model_params': {             ### 与模型相关的一些基础的参数  （在最开始构建模型时需要说明的）
                'use_bias': True,
                'w_reg': wregs,
                'w_reg_outcomes': wreg_outcomes,
                'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),           ### 越是后面几层他的dropout率是要越大的！
                'loss_weights': loss_weights,
                'optimizer': 'Adam',
                'activation': 'tanh',
                'data_params': data_base,
                'add_unk_genes': False,
                'shuffle_genes': False,
                'kernel_initializer': 'lecun_uniform',               ### 这个kernel_initializer是 keras网络层的初始化层      初始化定义了设置 Keras 各层权重随机初始值的方法
                'n_hidden_layers': n_hidden_layers,
                'attention': False,                                  ### 在这决定是否要对当前的这个模型增加注意力机制！   原来这个位置是 False
                'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference

            }, 'fitting_params': dict(samples_per_epoch=10,
                                      select_best_model=False,
                                      monitor='val_o6_f1',
                                      verbose=2,
                                      epoch=80,       ### 原先这个位置是300
                                      shuffle=True,
                                      batch_size=50,
                                      save_name='pnet',
                                      debug=False,
                                      save_gradient=False,
                                      # class_weight='auto',
                                      class_weight=class_weight,
                                      n_outputs=n_hidden_layers + 1,
                                      prediction_output='average',
                                      early_stop=False,
                                      reduce_lr=False,
                                      reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
                                      lr=0.001,                  ### 原来这个位置是0.001   进行参数鲁棒性时，这里的取值分别为0.01-0.05
                                      max_f1=True
                                      ),
            'WeightOutputLayers': [0, 1],           ## 这个参数是指明某层神经网络的输出结果要不要根据当前梯度的大小来对他们进行加权处理！    梯度值大的话就对神经元的输出值进行强化；梯度值小的话就对神经元的输出值进行弱化；如果梯度为负的话就直接对这个输出值进行清零！       现在这个变量的取值为None 或者是[0,1,2....]  0表示基因层，之后就分别代表各个通路层
            'gradients': None,
            'gradients_Flag': False,           ### 这两项是新加的参数，看看能不能把新算出来的梯度值给传进来！
            'feature_importance': 'deepexplain_grad*input'         ### 在这说明了可解释方法选用哪种！
        },
}
features = {}
models = [nn_pathway]
# models = []



### 下面这块为全连接神经网络的构建代码！
nn_pathway_dense = {
    'type': 'nn',
    'id': 'dense',
    'params':
        {
            'build_fn': build_dense,
            'model_params': {
                'w_reg': 0.01,
                'n_weights': 71009,
                'optimizer': 'Adam',
                'activation': 'selu',
                'data_params': data_base,
            }, 'fitting_params': dict(samples_per_epoch=10,
                                      select_best_model=False,
                                      monitor='val_f1',
                                      verbose=2,
                                      epoch=80,                      ###  原先这个位置是300
                                      shuffle=True,
                                      batch_size=50,
                                      save_name='dense',
                                      debug=False,
                                      save_gradient=False,
                                      class_weight='auto',
                                      n_outputs=1,
                                      prediction_output='average',
                                      early_stop=False,
                                      reduce_lr=False,
                                      reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
                                      ),
            # 'feature_importance': 'deepexplain_grad*input'
        },
}

# models.append(nn_pathway_dense)                    ### 现在这个加的是全连接神经网络！





# ### 下面这几个模型是只处理其中的某一个通路层（加知识迁移，加梯度加权！）
# models = []
# ## 现在下面的这个是只有第一层通路曾进行处理的！
# PNet_Path1 = {
#     'type': 'nn',
#     'id': 'P-net-Path1',
#     'params':
#         {
#             'build_fn': build_pnet2,          ### 此参数格外重要，指明模型具体构建的手段
#             'model_params': {             ### 与模型相关的一些基础的参数  （在最开始构建模型时需要说明的）
#                 'use_bias': True,
#                 'w_reg': wregs,
#                 'w_reg_outcomes': wreg_outcomes,
#                 'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),           ### 越是后面几层他的dropout率是要越大的！
#                 'loss_weights': loss_weights,
#                 'optimizer': 'Adam',
#                 'activation': 'tanh',
#                 'data_params': data_base,
#                 'add_unk_genes': False,
#                 'shuffle_genes': False,
#                 'kernel_initializer': 'lecun_uniform',               ### 这个kernel_initializer是 keras网络层的初始化层      初始化定义了设置 Keras 各层权重随机初始值的方法
#                 'n_hidden_layers': n_hidden_layers,
#                 'attention': True,                                  ### 在这决定是否要对当前的这个模型增加注意力机制！   原来这个位置是 False
#                 'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference
#
#             }, 'fitting_params': dict(samples_per_epoch=10,
#                                       select_best_model=False,
#                                       monitor='val_o6_f1',
#                                       verbose=2,
#                                       epoch=300,       
#                                       shuffle=True,
#                                       batch_size=50,
#                                       save_name='pnet',
#                                       debug=False,
#                                       save_gradient=False,
#                                       class_weight='auto',
#                                       n_outputs=n_hidden_layers + 1,
#                                       prediction_output='average',
#                                       early_stop=False,
#                                       reduce_lr=False,
#                                       reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
#                                       lr=0.001,
#                                       max_f1=True
#                                       ),
#             'gradients': 0,
#             'gradients_Flag': False,           ### 这两项是新加的参数，看看能不能把新算出来的梯度值给传进来！
#             'feature_importance': 'deepexplain_grad*input'         ### 在这说明了可解释方法选用哪种！
#         },
# }
# models.append(PNet_Path1)
#
# ## 现在下面的这个是只有第二层通路曾进行处理的！
# PNet_Path2 = {
#     'type': 'nn',
#     'id': 'P-net-Path2',
#     'params':
#         {
#             'build_fn': build_pnet2,          ### 此参数格外重要，指明模型具体构建的手段
#             'model_params': {             ### 与模型相关的一些基础的参数  （在最开始构建模型时需要说明的）
#                 'use_bias': True,
#                 'w_reg': wregs,
#                 'w_reg_outcomes': wreg_outcomes,
#                 'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),           ### 越是后面几层他的dropout率是要越大的！
#                 'loss_weights': loss_weights,
#                 'optimizer': 'Adam',
#                 'activation': 'tanh',
#                 'data_params': data_base,
#                 'add_unk_genes': False,
#                 'shuffle_genes': False,
#                 'kernel_initializer': 'lecun_uniform',               ### 这个kernel_initializer是 keras网络层的初始化层      初始化定义了设置 Keras 各层权重随机初始值的方法
#                 'n_hidden_layers': n_hidden_layers,
#                 'attention': True,                                  ### 在这决定是否要对当前的这个模型增加注意力机制！   原来这个位置是 False
#                 'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference
#
#             }, 'fitting_params': dict(samples_per_epoch=10,
#                                       select_best_model=False,
#                                       monitor='val_o6_f1',
#                                       verbose=2,
#                                       epoch=300,       ### 原先这个位置是300
#                                       shuffle=True,
#                                       batch_size=50,
#                                       save_name='pnet',
#                                       debug=False,
#                                       save_gradient=False,
#                                       class_weight='auto',
#                                       n_outputs=n_hidden_layers + 1,
#                                       prediction_output='average',
#                                       early_stop=False,
#                                       reduce_lr=False,
#                                       reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
#                                       lr=0.001,
#                                       max_f1=True
#                                       ),
#             'gradients': 1,
#             'gradients_Flag': False,           ### 这两项是新加的参数，看看能不能把新算出来的梯度值给传进来！
#             'feature_importance': 'deepexplain_grad*input'         ### 在这说明了可解释方法选用哪种！
#         },
# }
# models.append(PNet_Path2)
#
# ## 现在下面的这个是只有第一层通路曾进行处理的！
# PNet_Path3 = {
#     'type': 'nn',
#     'id': 'P-net-Path3',
#     'params':
#         {
#             'build_fn': build_pnet2,          ### 此参数格外重要，指明模型具体构建的手段
#             'model_params': {             ### 与模型相关的一些基础的参数  （在最开始构建模型时需要说明的）
#                 'use_bias': True,
#                 'w_reg': wregs,
#                 'w_reg_outcomes': wreg_outcomes,
#                 'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),           ### 越是后面几层他的dropout率是要越大的！
#                 'loss_weights': loss_weights,
#                 'optimizer': 'Adam',
#                 'activation': 'tanh',
#                 'data_params': data_base,
#                 'add_unk_genes': False,
#                 'shuffle_genes': False,
#                 'kernel_initializer': 'lecun_uniform',               ### 这个kernel_initializer是 keras网络层的初始化层      初始化定义了设置 Keras 各层权重随机初始值的方法
#                 'n_hidden_layers': n_hidden_layers,
#                 'attention': True,                                  ### 在这决定是否要对当前的这个模型增加注意力机制！   原来这个位置是 False
#                 'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference
#
#             }, 'fitting_params': dict(samples_per_epoch=10,
#                                       select_best_model=False,
#                                       monitor='val_o6_f1',
#                                       verbose=2,
#                                       epoch=300,       ### 原先这个位置是300
#                                       shuffle=True,
#                                       batch_size=50,
#                                       save_name='pnet',
#                                       debug=False,
#                                       save_gradient=False,
#                                       class_weight='auto',
#                                       n_outputs=n_hidden_layers + 1,
#                                       prediction_output='average',
#                                       early_stop=False,
#                                       reduce_lr=False,
#                                       reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
#                                       lr=0.001,
#                                       max_f1=True
#                                       ),
#             'gradients': 2,
#             'gradients_Flag': False,           ### 这两项是新加的参数，看看能不能把新算出来的梯度值给传进来！
#             'feature_importance': 'deepexplain_grad*input'         ### 在这说明了可解释方法选用哪种！
#         },
# }
# models.append(PNet_Path3)
#
# ## 现在下面的这个是只有第一层通路曾进行处理的！
# PNet_Path4 = {
#     'type': 'nn',
#     'id': 'P-net-Path4',
#     'params':
#         {
#             'build_fn': build_pnet2,          ### 此参数格外重要，指明模型具体构建的手段
#             'model_params': {             ### 与模型相关的一些基础的参数  （在最开始构建模型时需要说明的）
#                 'use_bias': True,
#                 'w_reg': wregs,
#                 'w_reg_outcomes': wreg_outcomes,
#                 'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),           ### 越是后面几层他的dropout率是要越大的！
#                 'loss_weights': loss_weights,
#                 'optimizer': 'Adam',
#                 'activation': 'tanh',
#                 'data_params': data_base,
#                 'add_unk_genes': False,
#                 'shuffle_genes': False,
#                 'kernel_initializer': 'lecun_uniform',               ### 这个kernel_initializer是 keras网络层的初始化层      初始化定义了设置 Keras 各层权重随机初始值的方法
#                 'n_hidden_layers': n_hidden_layers,
#                 'attention': True,                                  ### 在这决定是否要对当前的这个模型增加注意力机制！   原来这个位置是 False
#                 'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference
#
#             }, 'fitting_params': dict(samples_per_epoch=10,
#                                       select_best_model=False,
#                                       monitor='val_o6_f1',
#                                       verbose=2,
#                                       epoch=300,       ### 原先这个位置是300
#                                       shuffle=True,
#                                       batch_size=50,
#                                       save_name='pnet',
#                                       debug=False,
#                                       save_gradient=False,
#                                       class_weight='auto',
#                                       n_outputs=n_hidden_layers + 1,
#                                       prediction_output='average',
#                                       early_stop=False,
#                                       reduce_lr=False,
#                                       reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
#                                       lr=0.001,
#                                       max_f1=True
#                                       ),
#             'gradients': 3,
#             'gradients_Flag': False,           ### 这两项是新加的参数，看看能不能把新算出来的梯度值给传进来！
#             'feature_importance': 'deepexplain_grad*input'         ### 在这说明了可解释方法选用哪种！
#         },
# }
# models.append(PNet_Path4)
#
# ## 现在下面的这个是只有第一层通路曾进行处理的！
# PNet_Path5 = {
#     'type': 'nn',
#     'id': 'P-net-Path5',
#     'params':
#         {
#             'build_fn': build_pnet2,          ### 此参数格外重要，指明模型具体构建的手段
#             'model_params': {             ### 与模型相关的一些基础的参数  （在最开始构建模型时需要说明的）
#                 'use_bias': True,
#                 'w_reg': wregs,
#                 'w_reg_outcomes': wreg_outcomes,
#                 'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),           ### 越是后面几层他的dropout率是要越大的！
#                 'loss_weights': loss_weights,
#                 'optimizer': 'Adam',
#                 'activation': 'tanh',
#                 'data_params': data_base,
#                 'add_unk_genes': False,
#                 'shuffle_genes': False,
#                 'kernel_initializer': 'lecun_uniform',               ### 这个kernel_initializer是 keras网络层的初始化层      初始化定义了设置 Keras 各层权重随机初始值的方法
#                 'n_hidden_layers': n_hidden_layers,
#                 'attention': True,                                  ### 在这决定是否要对当前的这个模型增加注意力机制！   原来这个位置是 False
#                 'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference
#
#             }, 'fitting_params': dict(samples_per_epoch=10,
#                                       select_best_model=False,
#                                       monitor='val_o6_f1',
#                                       verbose=2,
#                                       epoch=300,       ### 原先这个位置是300
#                                       shuffle=True,
#                                       batch_size=50,
#                                       save_name='pnet',
#                                       debug=False,
#                                       save_gradient=False,
#                                       class_weight='auto',
#                                       n_outputs=n_hidden_layers + 1,
#                                       prediction_output='average',
#                                       early_stop=False,
#                                       reduce_lr=False,
#                                       reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
#                                       lr=0.001,
#                                       max_f1=True
#                                       ),
#             'gradients': 4,
#             'gradients_Flag': False,           ### 这两项是新加的参数，看看能不能把新算出来的梯度值给传进来！
#             'feature_importance': 'deepexplain_grad*input'         ### 在这说明了可解释方法选用哪种！
#         },
# }
# models.append(PNet_Path5)





### 下面的这个是加注意力机制的P-net网络


PNet_Attention = {
    'type': 'nn',
    'id': 'P-net-Attention',
    'params':
        {
            'build_fn': build_pnet2,          ### 此参数格外重要，指明模型具体构建的手段
            'model_params': {             ### 与模型相关的一些基础的参数  （在最开始构建模型时需要说明的）
                'use_bias': True,
                'w_reg': wregs,
                'w_reg_outcomes': wreg_outcomes,
                'dropout': [base_dropout] + [0.1] * (n_hidden_layers + 1),           ### 越是后面几层他的dropout率是要越大的！
                'loss_weights': loss_weights,
                'optimizer': 'Adam',
                'activation': 'tanh',
                'data_params': data_base,
                'add_unk_genes': False,
                'shuffle_genes': False,
                'kernel_initializer': 'lecun_uniform',               ### 这个kernel_initializer是 keras网络层的初始化层      初始化定义了设置 Keras 各层权重随机初始值的方法
                'n_hidden_layers': n_hidden_layers,
                'attention': True,                                  ### 在这决定是否要对当前的这个模型增加注意力机制！   原来这个位置是 False
                'dropout_testing': False  # keep dropout in testing phase, useful for bayesian inference

            }, 'fitting_params': dict(samples_per_epoch=10,
                                      select_best_model=False,
                                      monitor='val_o6_f1',
                                      verbose=2,
                                      epoch=300,       ### 原先这个位置是300
                                      shuffle=True,
                                      batch_size=50,
                                      save_name='pnet',
                                      debug=False,
                                      save_gradient=False,
                                      class_weight='auto',
                                      n_outputs=n_hidden_layers + 1,
                                      prediction_output='average',
                                      early_stop=False,
                                      reduce_lr=False,
                                      reduce_lr_after_nepochs=dict(drop=0.25, epochs_drop=50),
                                      lr=0.001,
                                      max_f1=True
                                      ),
            'feature_importance': 'deepexplain_grad*input'         ### 在这说明了可解释方法选用哪种！
        },
}
# models.append(PNet_Attention)

### 下面的这个是逻辑回归算法的一些参数
# class_weight = {0: 0.75, 1: 1.5}
logistic = {'type': 'sgd', 'id': 'Logistic Regression',
            'params': {'loss': 'log', 'penalty': 'l2', 'alpha': 0.01, 'class_weight': class_weight}}                   
##SGDClassifier是一系列采用了梯度下降来求解参数的算法的集合，例如（SVM, logistic regression)等； 而sklearn中，LogisticRegression的实现方法是基于“liblinear”, “newton-cg”, “lbfgs” and “sag”这些库来实现的，当数据集特别大的时候，推荐使用SGDClassifier中的逻辑回归
        ### 所以说眼下的这个SGDClassifier() 其实是一个算法的集合！   它里面包含了SVM以及逻辑回归，而且这两类算法都是使用SGD算法来进行实现的！    而且里面的这个损失函数是 log  这个就代表是逻辑回归！     对数损失函数，表示逻辑回归模型
models.append(logistic)      ### 此时这个模型models中是包含两个算法的参数   p-net以及逻辑回归





#### 原先下面的这几个对标算法是没有的， 现在是新添加的几项 其他的对标算法  包括：RBF Support Vector Machine、Linear Support Vector Machine、random_forest、adaboost、decision_tree
OtherAlg = [

    {
        'type': 'svc',
        'id': 'RBF Support Vector Machine ',
        'params': {'kernel': 'rbf', 'C': 100, 'gamma': 0.001, 'probability': True, 'class_weight': class_weight}
    },

    {
        'type': 'svc', 'id':
        'Linear Support Vector Machine ',
        'params': {'kernel': 'linear', 'C': 0.1, 'probability': True, 'class_weight': class_weight}
    },

    {
        'type': 'random_forest',
        'id': 'Random Forest',
        'params': {'max_depth': None, 'n_estimators': 50, 'bootstrap': False, 'class_weight': class_weight}
    },

    {
        'type': 'adaboost',
        'id': 'Adaptive Boosting',
        'params': {'learning_rate': 0.1, 'n_estimators': 50}
    },

    {
        'type': 'decision_tree',
        'id': 'Decision Tree',
        'params': {'min_samples_split': 10, 'max_depth': 10}
    },

]

models = models + OtherAlg









# pipeline = {'type':  'one_split', 'params': { 'save_train' : True, 'eval_dataset': 'test'}}
pipeline = {'type': 'crossvalidation', 'params': {'n_splits': 5, 'save_train': True}}             ### 在这也说明进行几折交叉运算

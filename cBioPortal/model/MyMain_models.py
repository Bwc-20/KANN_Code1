import logging

import numpy as np

# from keras import Input
# from keras.engine import Model
# from keras.layers import Dense, Dropout, Lambda, Concatenate
# from keras.regularizers import l2
### 下面是来对导进来的这个包进行测试！
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras import Input
from keras.engine import Model
from keras.layers import Dense, Dropout, Lambda, Concatenate
from keras.regularizers import l2

import tensorflow as tf
from tensorflow.keras import backend as K

from data.data_access import Data
from data.pathways.gmt_pathway import get_KEGG_map
from model.builders.builders_utils import get_pnet
from model.layers_custom import f1, Diagonal, SparseTF
from model.model_utils import print_model, get_layers

from keras.layers import Dense, Dropout, Activation, BatchNormalization, multiply, Layer



### 当前文件用来构建含有生物学意义的神经网络模型（现在这个网络的节点以及边都被赋予相应的生物学意义）



# assumes the first node connected to the first n nodes and so on    假设第一个节点与前n个节点相连，以此类推
## 直接根据builders_utils文件中的get_net()函数来进行构建最后返回的就是  构建完成的网络以及对应的特征的名字
def build_pnet(optimizer, w_reg, add_unk_genes=True, sparse=True, dropout=0.5, use_bias=False, activation='tanh',
               loss='binary_crossentropy', data_params=None, n_hidden_layers=1, direction='root_to_leaf',
               batch_normal=False, kernel_initializer='glorot_uniform', shuffle_genes=False, reg_outcomes=False):
    print("现在是在prostate_models.py文件中， 数据参数情况！", data_params)
    print('现在是在prostate_models.py文件中， n_hidden_layers', n_hidden_layers)
    data = Data(**data_params)
    x, y, info, cols = data.get_data()        ### 再次的来读取训练数据！
    print(x.shape)
    print(y.shape)
    print(info.shape)
    print("现在是在prostate_models.py文件中，现在的这个cols.shape为：", cols.shape)
    # features = cols.tolist()
    features = cols          ### 这个输入数据是作为特征
    if loss == 'binary_crossentropy':     ### 根据损失函数来确定最后的激活函数
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('现在是在prostate_models.py文件中，x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))          ## 在这，这个logging表示记录日志

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))
    feature_names = []
    feature_names.append(features)

    n_features = x.shape[1]

    if hasattr(cols, 'levels'):     ## 此函数用于判断对象是否包含对应的属性！
        genes = cols.levels[0]
    else:
        genes = cols

    # n_genes = len(genes)
    # genes = list(genes)
    # layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg), use_bias=False, name='h0')
    # layer1 = SpraseLayer(n_genes, input_shape=(n_features,), activation=activation,  use_bias=False,name='h0')
    # layer1 = Dense(n_genes, input_shape=(n_features,), activation=activation, name='h0')
    ins = Input(shape=(n_features,), dtype='float32', name='inputs')       ## 构造输入的数据格式

    outcome, decision_outcomes, feature_n = get_pnet(ins,
                                                     features,
                                                     genes,
                                                     n_hidden_layers,
                                                     direction,
                                                     activation,
                                                     activation_decision,
                                                     w_reg,
                                                     w_reg_outcomes,
                                                     dropout,
                                                     sparse,
                                                     add_unk_genes,
                                                     batch_normal,
                                                     use_bias=use_bias,
                                                     kernel_initializer=kernel_initializer,
                                                     shuffle_genes=shuffle_genes,
                                                     attention=attention,
                                                     dropout_testing=dropout_testing,
                                                     non_neg=non_neg
                                                     # reg_outcomes=reg_outcomes,
                                                     # adaptive_reg =adaptive_reg,
                                                     # adaptive_dropout=adaptive_dropout
                                                     )
    # outcome= outcome[0:-2]
    # decision_outcomes= decision_outcomes[0:-2]
    # feature_n= feature_n[0:-2]


    print("加油！白文超！当前调用get_pnet所获取的输出结果outcome, decision_outcomes, feature_n为：", outcome, decision_outcomes, feature_n)
    feature_names.extend(feature_n)

    print('现在是prostate_models.py文件中，   Compiling...')

    model = Model(input=[ins], output=decision_outcomes)     ## 利用Keras来构造模型   指定输入和输出进而来构造当前的这个网络模型！

    # n_outputs = n_hidden_layers + 2
    n_outputs = len(decision_outcomes)        ### 最后的预测结果是要根据每一层的输出结果进行组合得到的！
    loss_weights = list(range(1, n_outputs + 1))
    # loss_weights = [l*l for l in loss_weights]
    loss_weights = [np.exp(l) for l in loss_weights]
    # loss_weights = [l*np.exp(l) for l in loss_weights]
    # loss_weights=1
    print('loss_weights', loss_weights)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, metrics=[f1], loss_weights=loss_weights)      ### 使用complier函数来配置训练模型   主要是进行网络模型的配置，还没进行训练呢！
    # loss=['binary_crossentropy']*(n_hidden_layers +2))
    logging.info('done compiling')

    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    return model, feature_names


# assumes the first node connected to the first n nodes and so on   假设第一个节点与前n个节点相连，以此类推
WeightOutputLayers = [0, 1]
def build_pnet2(optimizer, w_reg, w_reg_outcomes, add_unk_genes=True, sparse=True, loss_weights=1.0, dropout=0.5,
                use_bias=False, activation='tanh', loss='binary_crossentropy', data_params=None, n_hidden_layers=1,
                direction='root_to_leaf', batch_normal=False, kernel_initializer='glorot_uniform', shuffle_genes=False,
                attention=False, dropout_testing=False, non_neg=False, repeated_outcomes=True, sparse_first_layer=True, WeightOutputLayers=WeightOutputLayers, gradients=None, gradients_Flag=False):
    print("现在是在prostate_models.py文件中的build_pnet2函数， 数据参数情况22！", data_params)
    print('n_hidden_layers', n_hidden_layers)
    data = Data(**data_params)
    x, y, info, cols = data.get_data()          ### 再次获取训练所用的数据情况！
    # print("现在是在prostate_models.py文件中,现在读取的这个训练数据是谁！", len(x), type(x), x)
    print(x.shape)
    print(y.shape)
    print(info.shape)       ### 这个是样本的名字（行标题）
    print(cols.shape)       ### 这个是基因的名字（列标题）
    features = cols
    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    n_features = x.shape[1]

    if hasattr(cols, 'levels'):
        genes = cols.levels[0]
    else:
        genes = cols

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')       ### 指明当前第一层输入层

    outcome, decision_outcomes, feature_n = get_pnet(ins,
                                                     features=features,
                                                     genes=genes,
                                                     n_hidden_layers=n_hidden_layers,
                                                     direction=direction,
                                                     activation=activation,
                                                     activation_decision=activation_decision,
                                                     w_reg=w_reg,
                                                     w_reg_outcomes=w_reg_outcomes,
                                                     dropout=dropout,
                                                     sparse=sparse,
                                                     add_unk_genes=add_unk_genes,
                                                     batch_normal=batch_normal,
                                                     sparse_first_layer=sparse_first_layer,
                                                     use_bias=use_bias,
                                                     kernel_initializer=kernel_initializer,
                                                     shuffle_genes=shuffle_genes,
                                                     attention=attention,
                                                     dropout_testing=dropout_testing,
                                                     non_neg=non_neg,
                                                     WeightOutputLayers=WeightOutputLayers,
                                                     # gradients=gradients,             ### 这两项是后续所新加的两项值，是来进行测试的！
                                                     # gradients_Flag=gradients_Flag

                                                     )            ### 目前这个函数是根据中间隐藏层彼此之间的连接情况来构建中间的各个隐藏层！

    feature_names = feature_n
    feature_names['inputs'] = cols

    print('目前在prostate_models.py文件中的build_pnet2函数， Compiling...')

    if repeated_outcomes:
        outcome = decision_outcomes
    else:
        print("测试一下这个决策输出集看看目前是不是走的这里！")
        outcome = decision_outcomes[-1]

    print("目前的这个决策输出集以及最终输出分别是多少！", decision_outcomes, outcome)
    model = Model(input=[ins], output=outcome)




    ### $$$$$$$$$$$$$$$$$ 现在下面的这个来对指定网络层的输出结果进行梯度加权！！
    ###  自定义层   这个是又自定义的一层网络，用于对网络的输出结果继续进行加权！
    class WeightedOutput(Layer):
        def __init__(self, **kwargs):
            super(WeightedOutput, self).__init__(**kwargs)

        def build(self, input_shape):
            self.kernel = self.add_weight(name='kernel_BWC',
                                          shape=(input_shape[-1],),
                                          initializer='ones',
                                          trainable=True)
            super(WeightedOutput, self).build(input_shape)

        def call(self, inputs):
            # gradients = tf.gradients(self.loss, self.kernel)[0]
            # gradients /= (K.sqrt(K.mean(K.square(gradients))) + K.epsilon())  ### 将梯度值进行处理，进行归一化操作！
            # grad_values_normalise = K.relu(gradients)
            # weighted_output = tf.multiply(inputs, grad_values_normalise)
            # return weighted_output
            return inputs



    gradWeight = False
    if gradWeight:
        ### 下面为测试部分， 在这可以尝试着修改某层网络中的相应的参数值
        # AllName = ['h0', 'h1', 'h2', 'h3', 'h4']  ### 在这，他存放所有要处理的层的名字   ['h1']
        AllName = ['h1']        ### 在这，他存放所有要处理的层的名字   ['h1']
        for layer_name in AllName:  ### 每一层的来进行处理
            # 获取指定层的梯度
            # layer_name = 'h1'     ### 先试试第一层通路层
            ### 另一种的梯度计算方法
            layer = model.get_layer(layer_name)
            layer_output = model.get_layer(layer_name).output      ### 指定网络层的输出
            # 创建中间层之前的部分模型
            partial_model = Model(inputs=model.inputs, outputs=layer_output)       ### 这个是前半部分的网络
            # 使用自定义网络层处理中间层的输出
            custom_output = WeightedOutput()(partial_model.output)
            # 创建中间层之后的部分模型
            After_model = Model(inputs=custom_output, outputs=model.output)
            # 创建最终模型，连接中间层之前和之后的部分
            new_model = Model(inputs=model.inputs, outputs=After_model.output)
            # 打印新模型的结构
            new_model.summary()
            print("现在这个新模型是构建好了！！")

            ### 赋值新模型！
            model = new_model
            model.summary()


            # ### 求出对应的梯度值
            # layer_output = model.get_layer(layer_name).output
            # gradients = K.gradients(self.model.total_loss, layer_output)[0]  ### 求出对应的梯度值
            # ### 将梯度值进行进一步的处理！
            # gradients /= (K.sqrt(K.mean(K.square(gradients))) + K.epsilon())  ### 将梯度值进行处理，进行归一化操作！
            # grad_values_normalise = K.relu(gradients)
            # ### 将处理后的梯度值进行赋值
            # layer.attentionWeights = grad_values_normalise  ### 每隔10个epoch便来将当前的





    if type(outcome) == list:
        n_outputs = len(outcome)
    else:
        n_outputs = 1

    if type(loss_weights) == list:
        loss_weights = loss_weights
    else:
        loss_weights = [loss_weights] * n_outputs              ###   因为这个损失权重除了首个的输入层  之后六层各层输出结果都是要参与损失计算的

    print('目前在prostate_models.py文件中， loss_weights', loss_weights)

    ### 下面在这进行自定义个一个损失函数，并在这个损失函数之后加上对应的正则化项，使模型的输出结果偏向对应的GSEA分数值！
    def custom_loss_with_regularization(y_true, y_pred):
        cross_entropy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        # regularization_loss = tf.reduce_sum(model.losses)  # 添加模型中定义的正则化损失项
        # LossList = []
        # for i in range(6):
        #     LossList.append(cross_entropy_loss)
        # # total_loss = cross_entropy_loss + regularization_loss
        # # return total_loss
        # LossList = np.array(LossList)
        LossList = [cross_entropy_loss] * 6
        return LossList

    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, metrics=[f1], loss_weights=loss_weights)       ### 根据优化器以及训练时的相关损失参数来进行拟合训练

    # model.compile(optimizer=optimizer,
    #               loss=custom_loss_with_regularization, metrics=[f1], loss_weights=loss_weights)       ### 根据优化器以及训练时的相关损失参数来进行拟合训练


    logging.info('目前在prostate_models.py文件中， done compiling')

    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    return model, feature_names



### 使用下面这个函数来将模型中的各层网络给串起来！
def apply_models(models, inputs):
    output = inputs
    for m in models:
        output = m(output)

    return output


## 根据给定的输入信息以及网络等信息来构造实际的各层神经网络，之后再将各层网络进行连接
def get_clinical_netowrk(ins, n_features, n_hids, activation):
    layers = []
    ## 根据隐藏层信息构建具体的机器学习神经网络，并向其中添加Dropout等信息
    for i, n in enumerate(n_hids):
        if i == 0:
            layer = Dense(n, input_shape=(n_features,), activation=activation, W_regularizer=l2(0.001),
                          name='h_clinical' + str(i))       ### 第一层网络为基因层 要指定他的输入数据的形状，之后各层都不需要指定！
        else:
            layer = Dense(n, activation=activation, W_regularizer=l2(0.001), name='h_clinical' + str(i))        ###在此，W_regularizer为权重衰减率  相当于一个正则化项！

        layers.append(layer)
        drop = 0.5
        layers.append(Dropout(drop, name='droput_clinical_{}'.format(i)))

    merged = apply_models(layers, ins)
    output_layer = Dense(1, activation='sigmoid', name='clinical_out')     ### 最后一层单独来给，作为输出层的，他只有一个神经元  就只输出一个一维的单独数据，作为当前的预测概率
    outs = output_layer(merged)

    return outs


def build_pnet2_account_for(optimizer, w_reg, w_reg_outcomes, add_unk_genes=True, sparse=True, loss_weights=1.0,
                            dropout=0.5,
                            use_bias=False, activation='tanh', loss='binary_crossentropy', data_params=None,
                            n_hidden_layers=1,
                            direction='root_to_leaf', batch_normal=False, kernel_initializer='glorot_uniform',
                            shuffle_genes=False,
                            attention=False, dropout_testing=False, non_neg=False, repeated_outcomes=True,
                            sparse_first_layer=True):
    print("目前所在文件是：prostate_models.py，现在来求数据参数", data_params)

    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    assert len(
        cols.levels) == 3, "expect to have pandas dataframe with 3 levels [{'clinicla, 'genomics'}, genes, features] "

    import pandas as pd
    x_df = pd.DataFrame(x, columns=cols, index=info)
    genomics_label = list(x_df.columns.levels[0]).index('genomics')
    genomics_ind = x_df.columns.labels[0] == genomics_label
    genomics = x_df['genomics']
    features_genomics = genomics.columns.remove_unused_levels()

    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('目前在prostate_models.py文件中，x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    n_features = x_df.shape[1]
    n_features_genomics = len(features_genomics)

    if hasattr(features_genomics, 'levels'):
        genes = features_genomics.levels[0]
    else:
        genes = features_genomics

    print("n_features", n_features, "n_features_genomics", n_features_genomics)
    print("genes", len(genes), genes)

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')

    ins_genomics = Lambda(lambda x: x[:, 0:n_features_genomics])(ins)
    ins_clinical = Lambda(lambda x: x[:, n_features_genomics:n_features])(ins)

    clinical_outs = get_clinical_netowrk(ins_clinical, n_features, n_hids=[50, 1], activation=activation)

    outcome, decision_outcomes, feature_n = get_pnet(ins_genomics,
                                                     features=features_genomics,
                                                     genes=genes,
                                                     n_hidden_layers=n_hidden_layers,
                                                     direction=direction,
                                                     activation=activation,
                                                     activation_decision=activation_decision,
                                                     w_reg=w_reg,
                                                     w_reg_outcomes=w_reg_outcomes,
                                                     dropout=dropout,
                                                     sparse=sparse,
                                                     add_unk_genes=add_unk_genes,
                                                     batch_normal=batch_normal,
                                                     sparse_first_layer=sparse_first_layer,
                                                     use_bias=use_bias,
                                                     kernel_initializer=kernel_initializer,
                                                     shuffle_genes=shuffle_genes,
                                                     attention=attention,
                                                     dropout_testing=dropout_testing,
                                                     non_neg=non_neg

                                                     )

    feature_names = feature_n
    feature_names['inputs'] = x_df.columns

    print('目前在prostate_models.py文件中，Compiling...')

    if repeated_outcomes:
        outcome = decision_outcomes
    else:
        outcome = decision_outcomes[-1]

    outcome_list = outcome + [clinical_outs]

    combined_outcome = Concatenate(axis=-1, name='combine')(outcome_list)
    output_layer = Dense(1, activation='sigmoid', name='combined_outcome')
    combined_outcome = output_layer(combined_outcome)
    outcome = outcome_list + [combined_outcome]
    model = Model(input=[ins], output=outcome)

    if type(outcome) == list:
        n_outputs = len(outcome)
    else:
        n_outputs = 1

    if type(loss_weights) == list:
        loss_weights = loss_weights
    else:
        loss_weights = [loss_weights] * n_outputs

    print('目前在prostate_models.py文件中，loss_weights', loss_weights)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * n_outputs, metrics=[f1], loss_weights=loss_weights)
    logging.info('done compiling')

    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    print("加油！专心点！当前build_pnet2_account_for函数的feature_names输出结果是多少！", feature_names)
    return model, feature_names




## 构建这个模型的输入和输出层的网络
def build_dense(optimizer, n_weights, w_reg, activation='tanh', loss='binary_crossentropy', data_params=None):
    print(data_params)

    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    print(x.shape)
    print(y.shape)
    print(info.shape)
    print(cols.shape)
    # features = cols.tolist()
    features = cols
    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('目前在prostate_models.py文件中，x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))

    logging.info('目前在prostate_models.py文件中，x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))
    feature_names = []
    feature_names.append(features)

    n_features = x.shape[1]

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')            ### 初始化网络输入层的tensor
    n = np.ceil(float(n_weights) / float(n_features))
    print("现在当前这一层神经元的数目是多少个！", n, float(n_features), float(n_weights))
    layer1 = Dense(units=int(n), activation=activation, W_regularizer=l2(w_reg), name='h0')       ### 这个算是对应的那个基因层

    # layer1 = Dense(8255, activation=activation, W_regularizer=l2(w_reg), name='h0')  ### 这个算是对应的那个基因层
    outcome = layer1(ins)
    # ### 下面的这个是自己手动进行添加的各层网络！
    # outcome = Dropout(0.5, name='dropout_0')(outcome)
    # outcome = Dense(1387, activation=activation_decision, name='h1')(outcome)            ### 现在这个为第一个通路层网络！
    # outcome = Dropout(0.6, name='dropout_1')(outcome)
    # outcome = Dense(1066, activation=activation_decision, name='h2')(outcome)  ### 现在这个为第一个通路层网络！
    # outcome = Dropout(0.7, name='dropout_2')(outcome)
    # outcome = Dense(447, activation=activation_decision, name='h3')(outcome)  ### 现在这个为第一个通路层网络！
    # outcome = Dropout(0.8, name='dropout_3')(outcome)
    # outcome = Dense(147, activation=activation_decision, name='h4')(outcome)  ### 现在这个为第一个通路层网络！
    # outcome = Dropout(0.9, name='dropout_4')(outcome)
    # outcome = Dense(26, activation=activation_decision, name='h5')(outcome)  ### 现在这个为第一个通路层网络！


    outcome = Dense(1, activation=activation_decision, name='output')(outcome)                  ### 现在这个为输出层网络！
    model = Model(input=[ins], output=outcome)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=[f1])
    logging.info('done compiling')

    print_model(model)
    print(get_layers(model))
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    return model, feature_names


def build_pnet_KEGG(optimizer, w_reg, dropout=0.5, activation='tanh', use_bias=False,
                    kernel_initializer='glorot_uniform', data_params=None, arch=''):
    print(data_params)
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    print("目前是在prostate_models.py文件中，现在来看一下这个x，y以及其他变量他们的形状大小", x.shape)
    print(y.shape)
    print(info.shape)
    print(cols.shape)

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols.shape))
    feature_names = {}
    feature_names['inputs'] = cols
    # feature_names.append(cols)

    n_features = x.shape[1]
    if hasattr(cols, 'levels'):       ## hasattr() 函数用于判断对象是否包含对应的属性。
        genes = cols.levels[0]
    else:
        genes = cols

    feature_names['h0'] = genes
    # feature_names.append(genes)
    decision_outcomes = []
    n_genes = len(genes)
    genes = list(genes)

    ### 当前下面的这个是第一层网络，可以认为是输入层（就是基因层），他对接前面一系列突变等表达特征   第一参数表示神经元节点的个数  他的输入就是突变、CNV等信息
    ### 下面设计的这个Diagonal类继承自 keras库中的Layer类  直接根据输入的参数来构造一层神经网络
    layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg),
                      use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer)

    ins = Input(shape=(n_features,), dtype='float32', name='inputs')
    layer1_output = layer1(ins)

    decision0 = Dense(1, activation='sigmoid', name='o0'.format(0))(ins)
    decision_outcomes.append(decision0)       ### 当前的这个第一层的输出结果他是要再另外再加一个分支，加上sigmoid函数也来得到一个预测结果

    decision1 = Dense(1, activation='sigmoid', name='o{}'.format(1))(layer1_output)
    decision_outcomes.append(decision1)

    mapp, genes, pathways = get_KEGG_map(genes, arch)             ### 现在这个函数就是来从KEGG库中来导入相应的数据（就是所用的基因、通路以及这些基因和通路之间的对照关系表）
    print("拿捏再胡思乱想了！当前这个基因、通路和对照关系的长度与形状：", len(genes), len(pathways), mapp.shape)
    print("贞观世民，当前的这个基因情况！", genes)
    print("加油哇！少年英雄！当前这个通路情况是！", pathways)
    print("再坚持一下！基因和通路的关系表情况是！", mapp)

    n_genes, n_pathways = mapp.shape
    logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))

    hidden_layer = SparseTF(n_pathways, mapp, activation=activation, W_regularizer=l2(w_reg),
                            name='h1', kernel_initializer=kernel_initializer,
                            use_bias=use_bias)          ### 此时得到的这个隐藏层也是经过掩码矩阵处理之后的结果！

    # hidden_layer = Dense(n_pathways, activation=activation, W_regularizer=L1L2_with_map(mapp, w_reg, w_reg),
    #                      kernel_constraint=ConnectionConstaints(mapp), use_bias=False,
    #                      name='h1')

    layer2_output = hidden_layer(layer1_output)
    decision2 = Dense(1, activation='sigmoid', name='o2')(layer2_output)
    decision_outcomes.append(decision2)

    feature_names['h1'] = pathways
    # feature_names.append(pathways)
    print('目前在prostate_models.py文件中，Compiling...')

    model = Model(input=[ins], output=decision_outcomes)      ## 现在他的输出就是各层网络各自的概率输出结果

    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'] * 3, metrics=[f1])
    # loss=['binary_crossentropy']*(n_hidden_layers +2))
    logging.info('done compiling')

    print_model(model)
    print("目前在prostate_models.py文件中，现在来获取所构建的这个模型中各层网络的具体信息！\n", get_layers(model))
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    return model, feature_names

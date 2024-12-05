import sys

import numpy as np
from keras import backend as K
from keras.engine import InputLayer
from keras.layers import Dropout, BatchNormalization
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tensorflow as tf
# tf.global_variables_initializer()
import warnings
warnings.filterwarnings("ignore")



from model.model_utils import get_layers

# if __name__ == '__main__':
#     print("此时进行相互导入！！")
#     from model.model_utils import get_layers


# import sys
# sys.path.append('../')
# from model_utils import get_layers



### 根据传进来的模型以及输入数据直接获取该数据的预测分数
def predict(model, X, loss=None):
    prediction_scores = model.predict(X)

    prediction_scores = np.mean(np.array(prediction_scores), axis=0)
    if loss == 'hinge':
        prediction = np.where(prediction_scores >= 0.0, 1., 0.)
    else:
        prediction = np.where(prediction_scores >= 0.5, 1., 0.)

    return prediction


# def get_gradient_layer(model, X, y, layer):
#
#     # print 'layer', layer
#     grad = model.optimizer.get_gradients(model.total_loss, layer)
#     gradients = layer *  grad# gradient tensors
#     # gradients =  grad# gradient tensors
#     # gradients = layer * model.optimizer.get_gradients(model.output[0,0], layer) # gradient tensors
#     # gradients = model.optimizer.get_gradients(model.output[0,0], layer) # gradient tensors
#     # gradients =  model.optimizer.get_gradients(model.total_loss, layer) # gradient tensors
#
#     #special case of repeated outputs (e.g. output for each hidden layer)
#     if type(y) == list:
#         n = len(y)
#     else:
#         n = 1
#
#     # print model.inputs[0]._keras_shape, model.targets[0]._keras_shape
#     # print 'model.targets', model.targets[0:n]
#     # print 'model.inputs[0]', model.inputs[0]
#     input_tensors = [model.inputs[0],  # input data
#                      # model.sample_weights[0],  # how much to weight each sample by
#                      # model.targets[0:n],  # labels
#                      # model.targets[0],  # labels
#                      # K.learning_phase(),  # train or test mode
#                      ]
#
#     for i in range(n):
#         input_tensors.append(model.sample_weights[i])
#
#     for i in range(n):
#         input_tensors.append(model.targets[i])
#
#     input_tensors.append(K.learning_phase())
#     gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)
#
#
#     get_gradients = K.function(inputs=input_tensors, outputs=[gradients])
#
#
#     # https: // github.com / fchollet / keras / issues / 2226
#     # print 'y_train', y_train.shape
#
#     nb_sample = X.shape[0]
#
#     # if type(y ) ==list:
#     #     y= [yy.reshape((nb_sample, 1)) for yy in y]
#     #     sample_weights = [np.ones(nb_sample) for i in range(n)]
#     # else:
#     #     y = y.reshape((nb_sample, 1))
#     #     sample_weights = np.ones(nb_sample)
#
#     inputs = [X,  # X
#               # sample_weights,  # sample weights
#               # y,  # y
#               # 0  # learning phase in TEST mode
#               ]
#
#     for i in range(n):
#         inputs.append(np.ones(nb_sample))
#
#     if n>1 :
#         for i in range(n):
#             inputs.append(y[i].reshape((nb_sample, 1)))
#     else:
#         inputs.append(y.reshape(nb_sample, 1))
#
#     inputs.append(0)# learning phase in TEST mode
#     # print(X.shape)
#     # print (y.shape)
#
#     # inputs = [X,  # X
#     #           sample_weights,  # sample weights
#     #           y,  # y
#     #           0  # learning phase in TEST mode
#     #           ]
#     # print weights
#     gradients = get_gradients(inputs)[0]
#
#     return gradients


def get_gradient_layer(model, X, y, layer, normalize=True):
    grad = model.optimizer.get_gradients(model.total_loss, layer)     ### 获取指定层的梯度情况！
    gradients = layer * grad  # gradient tensors

    # special case of repeated outputs (e.g. output for each hidden layer)
    if type(y) == list:
        n = len(y)
    else:
        n = 1
    input_tensors = [model.inputs[0],  # input data
                     # model.sample_weights[0],  # how much to weight each sample by
                     # model.targets[0:n],  # labels
                     # model.targets[0],  # labels
                     # K.learning_phase(),  # train or test mode
                     ]

    # how much to weight each sample by
    for i in range(n):
        input_tensors.append(model.sample_weights[i])
    # labels
    for i in range(n):
        input_tensors.append(model.targets[i])
    # train or test mode
    input_tensors.append(K.learning_phase())
    # normalize
    if normalize:
        gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)

    get_gradients = K.function(inputs=input_tensors, outputs=[gradients])
    # https: // github.com / fchollet / keras / issues / 2226
    # print 'y_train', y_train.shape

    nb_sample = X.shape[0]

    inputs = [X,  # X
              # sample_weights,  # sample weights
              # y,  # y
              # 0  # learning phase in TEST mode
              ]

    for i in range(n):
        inputs.append(np.ones(nb_sample))

    if n > 1:
        for i in range(n):
            inputs.append(y[i].reshape((nb_sample, 1)))
    else:
        inputs.append(y.reshape(nb_sample, 1))

    inputs.append(0)  # learning phase in TEST mode
    gradients = get_gradients(inputs)[0][0]
    return gradients


def get_shap_scores_layer(model, X, layer_name, output_index=-1, method_name='deepexplainer'):
    # local_smoothing ?
    # ranked_outputs
    def map2layer(model, x, layer_name):
        fetch = model.get_layer(layer_name).output     ## 获取模型每一层的输出
        feed_dict = dict(list(zip([model.layers[0].input], [x.copy()])))           ### 现在，在这这个x就是传进来的X，表示模型的原始输入数据   现在等于是让模型最原始的输入数据与给定的训练数据一一对应起来！然后通过 dict 函数将其转换为字典形式。
        return K.get_session().run(fetch, feed_dict)            ### fetch就是当前这层网路的输出， feed_dict为当前整个网络模型的输入！  输入整个模型的输入数据，来返回当前这个网络层的输出！

    import shap
    if type(output_index) == str:
        y = model.get_layer(output_index).output
    else:
        y = model.outputs[output_index]

    x = model.get_layer(layer_name).output

    ### 下面这块是专门进行测试的
    thisLayerOutput = map2layer(model, X.copy(), layer_name)  ### 获得当前这层网络的输出数据
    print("当前的这个网络的输出结果的数据类型是怎样的！", type(thisLayerOutput))
    print("当前这组数据的平均值、最小以及最大值分别是多少！", np.mean(thisLayerOutput), np.max(thisLayerOutput), np.min(thisLayerOutput))
    if method_name == 'deepexplainer':
        print("现在是测试部分！", map2layer(model, X.copy(), layer_name).shape, map2layer(model, X.copy(), layer_name))                ### 现在这个map2layer(model, X.copy(), layer_name)就表示当前这个给定网络层的输出！

        # 初始化变量
        sess = tf.keras.backend.get_session()
        sess.run(tf.global_variables_initializer())

        print("现在下面是测试部分测一下，当前的这些x, y的形状，为啥会出错呢？！", x.shape, y.shape, x, y, X, layer_name)             ### 在这X是输入的训练数据
        explainer = shap.DeepExplainer((x, y), map2layer(model, X.copy(), layer_name))                      ### 应该是这句代码发生了错误！  里面这两组参数都是当前这个网络层的输出
        shap_values, indexes = explainer.shap_values(map2layer(model, X, layer_name), ranked_outputs=2)
    elif method_name == 'gradientexplainer':
        explainer = shap.GradientExplainer((x, y), map2layer(model, X.copy(), layer_name), local_smoothing=2)
        shap_values, indexes = explainer.shap_values(map2layer(model, X, layer_name), ranked_outputs=2)
    else:
        raise ('unsppuorted method')

    print("测试部分，目前是在coef_weights_utils.py文件中，测试下目前这个shap_values的形状情况！", type(shap_values), shap_values)
    print((shap_values[0].shape))
    return shap_values[0]


# model, X_train, y_train, target, detailed=detailed, method_name=method
def get_shap_scores(model, X_train, y_train, target=-1, method_name='deepexplainer', detailed=False):
    gradients_list = []
    gradients_list_sample_level = []
    i = 0
    for l in get_layers(model):
        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )
            if target is None:
                output = i
            else:
                output = target
            print('layer # {}, layer name {},  output name {}'.format(i, l.name, output))
            i += 1
            # gradients = get_deep_explain_score_layer(model, X_train, l.name, output, method_name= method_name )
            gradients = get_shap_scores_layer(model, X_train, l.name, output, method_name=method_name)       ### 获取每一层的shap分数
            # getting average score
            if gradients.ndim > 1:
                # feature_weights = np.sum(np.abs(gradients), axis=-2)
                feature_weights = np.sum(gradients, axis=-2)
            else:
                feature_weights = np.abs(gradients)
            gradients_list.append(feature_weights)
            gradients_list_sample_level.append(gradients)
    if detailed:
        return gradients_list, gradients_list_sample_level
    else:
        return gradients_list
    pass


### 下面是根据输入的训练数据以及模型，利用deeplift算法来求特征重要性！
def get_deep_explain_scores(model, X_train, y_train, target=-1, method_name='grad*input', detailed=False, **kwargs):
    # gradients_list = []
    # gradients_list_sample_level = []

    gradients_list = {}
    gradients_list_sample_level = {}

    i = 0
    for l in get_layers(model):
        if type(l) in [Sequential, Dropout, BatchNormalization]:      ### 在特定的网络层上是不用计算相应的重要性分数的！
            continue
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )

            if target is None:
                output = i
            else:
                output = target

            print('layer # {}, layer name {},  output name {}'.format(i, l.name, output))
            i += 1
            gradients = get_deep_explain_score_layer(model, X_train, l.name, output, method_name=method_name)      ## 这个来求各个层的贡献分数
            if gradients.ndim > 1:
                # feature_weights = np.sum(np.abs(gradients), axis=-2)
                # feature_weights = np.sum(gradients, axis=-2)
                print('目前是在coef_weights_utils.py文件中, gradients.shape', gradients.shape)        ### 他现在的形状为（传进来的样本的个数, 当前这一层中神经元的数目）
                # feature_weights = np.abs(np.sum(gradients, axis=-2))
                feature_weights = np.sum(gradients, axis=-2)        ### 每个样本在该位置的重要性分数进行求和来作为当前这个位置的重要性分数！
                # feature_weights = np.mean(gradients, axis=-2)
                print('目前是在coef_weights_utils.py文件中, feature_weights.shape', feature_weights.shape)             ### 现在所计算出来的这个feature_weights 就是当前层各个神经元节点他们重要性分数！  他的形状就是(当前这一层中神经元的数目, )
                print('目前是在coef_weights_utils.py文件中, feature_weights min max', min(feature_weights), max(feature_weights))       ### 现在所计算出来的这个feature_weights 就是当前层各个神经元节点他们重要性分数！
            else:
                # feature_weights = np.abs(gradients)
                feature_weights = gradients
                # feature_weights = np.mean(gradients)
            # gradients_list.append(feature_weights)
            # gradients_list_sample_level.append(gradients)
            gradients_list[l.name] = feature_weights
            gradients_list_sample_level[l.name] = gradients
    if detailed:
        return gradients_list, gradients_list_sample_level
    else:
        return gradients_list
    pass


### 计算具体的某一层中各个神经元节点的贡献分数     在这用的是deep_explain算法，只是在进行具体计算的时候这个deep_explain算法内部使用的是偏导数
def get_deep_explain_score_layer(model, X, layer_name, output_index=-1, method_name='grad*input'):
    scores = None
    import keras
    from deepexplain.tensorflow_ import DeepExplain
    import tensorflow as tf
    ww = model.get_weights()
    with tf.Session() as sess:
        try:
            with DeepExplain(session=sess) as de:  # <-- init DeepExplain context
                # Need to reconstruct the graph in DeepExplain context, using the same weights.    需要在DeepExplain上下文中使用相同的权重来重建图
                # model= nn_model.model
                print("目前是coef_weights_utils.py文件中，get_deep_explain_score_layer函数下， 当前这个网络层的名字是：", layer_name, model)

                model = keras.models.clone_model(model)
                model.set_weights(ww)       ### 目前在这里我再复制构建一个网络，省的将原来的网络破坏！


                # if layer_name=='inputs':
                #     layer_outcomes= X
                # else:
                #     layer_outcomes = nn_model.get_layer_output(layer_name, X)[0]

                x = model.get_layer(layer_name).output
                # x = model.inputs[0]
                if type(output_index) == str:
                    y = model.get_layer(output_index).output
                else:
                    y = model.outputs[output_index]

                # y = model.get_layer('o6').output
                # x = model.inputs[0]
                print("现在是在coef_weights_utils.py文件下，当前这一层的名字是：", layer_name)
                print('model.inputs', model.inputs)
                print('model y', y)
                print('model x', x)
                ### 下面的这个函数是关键，他是直接调用 deepexplain 函数来求解一下各层网络中各个神经元节点的重要性分数
                attributions = de.explain(method_name, y, x, model.inputs[0], X)        ## 现在这个属性就是对每一层他的输入数据的解释结果！具体的分数情况  对每一层都来进行一波这样的处理，输入层就是指的原始的输入数据27687，后续各个隐藏层就是各隐藏层内部的节点！    在这实例化DeepExplain这个类
                print('现在是在coef_weights_utils.py文件下，现在来求一下属性的形状，attributions', attributions.shape)
                print('现在是在coef_weights_utils.py文件下，现在来求一下属性的具体情况：', np.max(attributions[0]), np.min(attributions[0]), np.average(attributions[0]), attributions)
                scores = attributions
                return scores
        except:
            sess.close()
            print(("Unexpected error:", sys.exc_info()[0]))
            raise


def get_skf_weights(model, X, y, importance_type):
    from features_processing.feature_selection import FeatureSelectionModel
    layers = get_layers(model)
    inp = model.input
    layer_weights = []
    for i, l in enumerate(layers):

        if type(l) == InputLayer:
            layer_out = X
        elif l.name.startswith('h'):
            out = l.output
            print("当前是处于网络的第几层以及这个网络的输入以及输出情况是！", i, l, out)
            func = K.function([inp] + [K.learning_phase()], [out])
            layer_out = func([X, 0.])[0]
        else:
            continue

        if type(y) == list:
            y = y[0]

        # layer_out = StandardScaler().fit_transform(layer_out)
        p = {'type': importance_type, 'params': {}}
        fs_model = FeatureSelectionModel(p)
        fs_model = fs_model.fit(layer_out, y.ravel())
        fs_coef = fs_model.get_coef()
        fs_coef[fs_coef == np.inf] = 0
        layer_weights.append(fs_coef)
    return layer_weights


def get_gradient_weights(model, X, y, signed=False, detailed=False, normalize=True):
    gradients_list = []
    gradients_list_sample_level = []
    for l in get_layers(model):
        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )
            w = l.get_output_at(0)
            gradients = get_gradient_layer(model, X, y, w, normalize)       ## 获取每一层的梯度情况
            print("目前这个是在coef_weights_utils.py文件中，这个梯度信息是！", gradients)
            if gradients.ndim > 1:
                if signed:
                    feature_weights = np.sum(gradients, axis=-2)
                else:
                    feature_weights = np.sum(np.abs(gradients), axis=-2)

            else:
                feature_weights = np.abs(gradients)
            gradients_list.append(feature_weights)
            gradients_list_sample_level.append(gradients)
    if detailed:
        return gradients_list, gradients_list_sample_level
    else:
        return gradients_list


def get_gradient_weights_with_repeated_output(model, X, y):
    gradients_list = []
    # print 'trainable weights',model.trainable_weights
    # print 'layers', get_layers (model)

    for l in get_layers(model):

        if type(l) in [Sequential, Dropout, BatchNormalization]:
            continue

        # print 'get the gradient of layer {}'.format(l.name)
        if l.name.startswith('o') and not l.name.startswith('o_'):
            print(l.name)
            print(l.weights)
            weights = l.get_weights()[0]
            # weights = l.get_weights()
            # print 'weights shape {}'.format(weights.shape)
            gradients_list.append(weights.ravel())

    return gradients_list


# get weights of each layer based on training a linear model that predicts the outcome (y) given the layer output          在训练线性模型的基础上获得每一层的权重，该模型可以预测给定层输出的结果（y）
def get_weights_linear_model(model, X, y):
    weights = None
    layer_weights = []
    layers = get_layers(model)
    inp = model.input
    for i, l in enumerate(layers):
        if type(l) in [Sequential, Dropout]:
            continue
        print(type(l))
        if type(l) == InputLayer:
            layer_out = X
        else:
            out = l.output
            print(i, l, out)
            func = K.function([inp] + [K.learning_phase()], [out])
            layer_out = func([X, 0.])[0]
        # print layer_out.shape
        # layer_outs.append(layer_out)
        linear_model = LogisticRegression(penalty='l1', solver='liblinear')
        # linear_model = LinearRegression( )
        # layer_out = StandardScaler().fit_transform(layer_out)
        if type(y) == list:
            y = y[0]
        linear_model.fit(layer_out, y.ravel())
        # print 'layer coef  shape ', linear_model.coef_.T.ravel().shape
        layer_weights.append(linear_model.coef_.T.ravel())
    return layer_weights


# def get_weights_gradient_outcome(model, x_train, y_train):
#     if type(y_train) == list:
#         n = len(y_train)
#     else:
#         n = 1
#     nb_sample = x_train.shape[0]
#     sample_weights = np.ones(nb_sample)
#     print model.output
#     # output = model.output[-1]
#     # model = nn_model.model
#     input_tensors = model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
#     # input_tensors = model.inputs + model.targets + [K.learning_phase()]
#     layers = get_layers(model)
#     gradients_list= []
#     i=0
#     for l in layers:
#         if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just a convention )
#             print i
#             output = model.output[i]
#             i+=1
#             print i, l.name, output.name, output, l.get_output_at(0)
#             # gradients = K.gradients(K.mean(output), l.get_output_at(0))
#             gradients = K.gradients(output, l.get_output_at(0))
#             # w= l.get_output_at(0)
#             # gradients = [w*g for g in K.gradients(output, w)]
#             get_gradients = K.function(inputs=input_tensors, outputs=gradients)
#             inputs = [x_train] + [sample_weights] * n + y_train  + [0]
#             gradients = get_gradients(inputs)
#             print 'gradients',len(gradients), gradients[0].shape
#             g= np.sum(np.abs(gradients[0]), axis = 0)
#             g= np.sum(gradients[0], axis = 0)
#             g= np.abs(g)
#             print 'gradients', gradients[0].shape
#             gradients_list.append(g)
#
#     return gradients_list
#

def get_gradeint(model, x, y, x_train, y_train, multiply_by_input=False):
    n_outcomes = 1
    if type(y_train) == list:
        n_outcomes = len(y_train)
    n_sample = x_train.shape[0]
    sample_weights = np.ones(n_sample)
    input_tensors = model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
    if multiply_by_input:
        gradients = [x * g for g in K.gradients(y, x)]
    else:
        gradients = K.gradients(y, x)
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)

    ### 下面是进行新修改的部分
    tempData = ([sample_weights] * n_outcomes)[0]
    tempData = tempData.reshape((y_train.shape[0], y_train.shape[1]))
    inputs = [[x_train] + tempData + y_train + [0]]

    # print("此时重塑后的这个数据形式是怎样的！", tempData.shape, tempData)
    # print("测试一下，看看此时还会不会报错了！", [x_train] + tempData + y_train + [0])
    # print("测试一下两项合并之后的情况", [x_train] + y_train)
    # print("测试一下，此时各个变量的输出规模形状", x_train.shape, x_train[0].shape, len(([sample_weights] * n_outcomes)[0]), y_train.shape)
    # print("测试一下两项合并之后的情况", ([x_train] + y_train).shape)
    # print("具体的各个变量的输出结果是怎样的！！[x_train]", [x_train])
    # print("具体的各个变量的输出结果是怎样的！！[sample_weights] * n_outcomes", [sample_weights] * n_outcomes)
    # print("具体的各个变量的输出结果是怎样的！！y_train", y_train)


    # inputs = [x_train] + [sample_weights] * n_outcomes + y_train + [0]    ## 原先这个位置是留的，但是因为数据的形状不一致，不能进行相加，因此用上面1那几句进行修改
    gradients = get_gradients(inputs)
    return gradients


def get_weights_gradient_outcome(model, x_train, y_train, detailed=False, target=-1, multiply_by_input=False,
                                 signed=True):
    print(model.output)
    layers = get_layers(model)
    gradients_list = []
    gradients_list_sample_level = []
    i = 0
    for l in layers:
        if l.name.startswith('h') or l.name.startswith('inputs'):  # hidden layers (this is just an ad hoc convention )

            if target is None:
                output = model.output[i]
            else:
                if type(target) == str:
                    output = model.get_layer(target).output
                else:
                    output = model.outputs[target]

            print('layer # {}, layer name {},  output name {}'.format(i, l.name, output.name))
            i += 1
            print(i, l.name, output.name, output, l.get_output_at(0))
            gradients = get_gradeint(model, l.output, output, x_train, y_train, multiply_by_input=multiply_by_input)           ### 这个是根据整个模型的输出以及当前层的输出来求当前层的梯度情况！

            print('目前是在coef_weights_utils.py文件中, gradients', len(gradients), gradients[0].shape)
            if signed:
                g = np.sum(gradients[0], axis=0)
            else:
                g = np.sum(np.abs(gradients[0]), axis=0)
            # g = np.abs(g)
            gradients_list_sample_level.append(gradients[0])
            print('gradients', gradients[0].shape)
            gradients_list.append(g)

    if detailed:
        return gradients_list, gradients_list_sample_level

    return gradients_list


# def get_gradient_weights(model, X, y):
#     gradients_list = []
#     print 'trainable weights',model.trainable_weights
#     print 'layers', model.layers
#
#     # for l in get_layers(model):
#     # for l in [model.inputs[0] ]+ model.trainable_weights:
#     # c = get_gradient_layer(model, X, y, model.inputs[0])
#     # gradients_list.append(np.mean(c, axis=0))
#     for l in  model.trainable_weights:
#         # print l
#         # l = l.trainable_weights
#         # layer = model.inputs[0]
#         # print  ,
#
#         # if type(l) == InputLayer:
#         #     w = model.inputs[0]
#         # # elif type(l)==Sequential:
#         # #     continue
#         # elif hasattr(l, 'kernel') and type(l) != SpraseLayer:
#         #     w= l.output
#         # else: continue
#
#         if 'kernel' in str(l):
#
#             gradients = get_gradient_layer(model, X, y, l)
#             if gradients.ndim >1:
#                 feature_weights = np.mean(gradients, axis=1)
#             else:
#                 feature_weights = gradients
#             # feature_weights= gradients
#             print 'layer {} grdaient shape {}', l, feature_weights.shape
#             gradients_list.append(feature_weights)
#     return gradients_list


### 通过扰动输入数据，观察一下输出结果的变化情况！输出结果准确性的变化就是各个节点的重要性分数
def get_permutation_weights(model, X, y):
    scores = []
    prediction_scores = predict(model, X)
    # print y
    # print prediction_scores
    print("目前是在coef_weights_utils.py文件中，测试部分，看看输入的训练数据以及标签情况！", X.shape, y.shape)
    print("目前是在coef_weights_utils.py文件中，测试部分，看看各个变量的维度情况！", y[0].shape)
    print("目前是在coef_weights_utils.py文件中，测试部分，看看各个变量的维度情况！prediction_scores", prediction_scores.shape)
    baseline_acc = accuracy_score(y, prediction_scores)     ### 根据原始数据的输入结果获取模型的预测结果并计算模型的准确性分数
    rnd = np.random.random((X.shape[0],))       ## 让输入数据发生随机性的变化
    x_original = X.copy()
    for i in range(X.shape[1]):
        # if (i%100)==0:
        # print("当前是coef_weights_utils.py文件中的get_permutation_weights，目前的这个输入数据的维度情况！", i)
        # x = X.copy()
        x_vector = x_original[:, i]
        # np.random.shuffle(x[:, i])
        x_original[:, i] = rnd
        acc = accuracy_score(y, predict(model, x_original))
        x_original[:, i] = x_vector
        scores.append((baseline_acc - acc) / baseline_acc)        ### 计算每个元素的重要性分数
    return np.array(scores)


def get_deconstruction_weights(model):
    for layer in model.layers:
        # print layer.name
        weights = layer.get_weights()  # list of numpy arrays
        # for w in weights:
        #     print w.shape
    pass
















### 选择一种可解释性方法来计算输入数据（主要是输入的训练数据）的重要性
def get_coef_importance(model, X_train, y_train, target, feature_importance, detailed=True, **kwargs):
    print("目前是在coef_weights_utils.py文件中，现在是来求可解释性算法的，现在是在可解释性算法外围，计算一下此时传进来的一些超参数：", len(X_train), len(X_train[0]), feature_importance)          ### 在这这个就是指输入进来的训练数据的维度（在这一般就是(727, 27687)，这个特征重要性选的是deepexplain_grad*input）
    print("目前的这个目标的信息是谁！", target)                ### 在这，这个target取值为-1

    ### 下面这块可以自主的来选择相应的可解释方法   正常情况下这是什么也不选的！   正常情况下这的这个取值是：feature_importance = 'deepexplain_grad*input'
    # # feature_importance = 'permutation'     ### deepexplainer  shap

    # feature_importance = 'shap_deepexplainer'                       ### 至于可解释性方法要不要用SHAP，主要是来调整这句代码！
    feature_importance = 'DeepLIFT'                       ### 至于可解释性方法要不要用SHAP，主要是来调整这句代码！
    # feature_importance = 'LIME'  ### 至于可解释性方法要不要用SHAP，主要是来调整这句代码！

    # feature_importance = 'shap_gradientexplainer'                   ### 第二种SHAP的可解释性方法（原来代码中用的并不是这个，而是上面那个！）

    # feature_importance = 'gradient_outcome_signed'         ## 这个是需要选用的！
    # feature_importance = 'loss_gradient'     ## 暂时还不行
    # feature_importance = 'skf'        ## 这个暂时也不行
    # feature_importance = 'linear'     ### 这个到最后也不行
    # feature_importance == 'one_to_one'



    if feature_importance.startswith('skf'):
        coef_ = get_skf_weights(model, X_train, y_train, feature_importance)
        # pass
    elif feature_importance == 'loss_gradient':
        coef_ = get_gradient_weights(model, X_train, y_train, signed=False, detailed=detailed,
                                     normalize=True)  # use total loss
    elif feature_importance == 'loss_gradient_signed':
        coef_ = get_gradient_weights(model, X_train, y_train, signed=True, detailed=detailed,
                                     normalize=True)  # use total loss
    elif feature_importance == 'gradient_outcome':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=False, signed=False)
    elif feature_importance == 'gradient_outcome_signed':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target=target, detailed=detailed,
                                             multiply_by_input=False, signed=True)
    elif feature_importance == 'gradient_outcome*input':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=True, signed=False)
    elif feature_importance == 'gradient_outcome*input_signed':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=True, signed=True)

    elif feature_importance.startswith('deepexplain'):
        method = feature_importance.split('_')[1]             ### 现在所求得的这个method是grad*input
        coef_ = get_deep_explain_scores(model, X_train, y_train, target, method_name=method, detailed=detailed,
                                        **kwargs)         ### 现在这个coef_就是所计算出来的各层神经网络中各个神经元节点的贡献分数
        # print("天行健，君子以自强不息！目前是在coef_weights_utils.py文件中，当前所计算出来的系数是谁：", coef_)
        ## 现在来统计一下计算得到的各层网络他们重要性的一些统计情况     现在所得到的这个coef_是一个字典的形式，key值就是各个网络层的名字，他的Value就是各个网络层他们内部各个节点的重要性分数         Similarity
        # coef_Infor = {}
        # for key in coef_:
        #     coef_Infor[key] = [len(coef_[key]), np.max(coef_[key]), np.min(coef_[key]), np.average(coef_[key])]
        # print("目前是在coef_weights_utils.py文件中，所计算出来的各层神经网络他们的重要性分数的统计情况为：", coef_Infor)

    elif feature_importance == 'DeepLIFT':
        method = 'deeplift'
        coef_ = get_deep_explain_scores(model, X_train, y_train, target, method_name=method, detailed=detailed,
                                        **kwargs)         ### 现在这个coef_就是所计算出来的各层神经网络中各个神经元节点的贡献分数
    elif feature_importance == 'LIME':
        method = 'intgrad'
        coef_ = get_deep_explain_scores(model, X_train, y_train, target, method_name=method, detailed=detailed,
                                        **kwargs)         ### 现在这个coef_就是所计算出来的各层神经网络中各个神经元节点的贡献分数

    elif feature_importance.startswith('shap'):
        method = feature_importance.split('_')[1]
        print("目前是在coef_weights_utils.py文件中, 目前是已经选用了SHAP可解释性方法！")
        coef_ = get_shap_scores(model, X_train, y_train, target, method_name=method, detailed=detailed)


    elif feature_importance == 'gradient_with_repeated_outputs':
        coef_ = get_gradient_weights_with_repeated_output(model, X_train, y_train, target)
    elif feature_importance == 'permutation':     ### 通过随机扰动输入的数据，观察一下对最终准确性结果的变化情况！
        coef_ = get_permutation_weights(model, X_train, y_train)
    elif feature_importance == 'linear':
        coef_ = get_weights_linear_model(model, X_train, y_train)
    elif feature_importance == 'one_to_one':          ### 这个就是直接以当前这个神经网络中这个神经元所对应的那个权重，直接拿那个权重来作为对应的重要性分数！
        weights = model.layers[1].get_weights()
        switch_layer_weights = weights[0]
        coef_ = np.abs(switch_layer_weights)
    else:
        coef_ = None
    return coef_
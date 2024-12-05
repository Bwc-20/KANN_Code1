import datetime
import logging
import math
import os
import json

import numpy as np
import pandas as pd
import csv
import random


### 下面对导入的keras版本进行修改！
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from model.callbacks_custom import GradientCheckpoint, FixedEarlyStopping

### 下面这个是导入回调函数，根据这些回调函数来求出各层网络中各个神经元的梯度值，并利用这些梯度值来将对应的神经元的输出值进行处理！
from model.callbacks_custom import GradientModifier, adjust_output_by_gradient, GradientProcessingCallback, GradientProcessingCallback_Four, HyperParam, GradientProcessingCallback_Five
from keras.callbacks import LambdaCallback


# from model.model_utils import get_layers, plot_history, get_coef_importance
import keras
from keras.models import Model as NewModel
from model.model_utils import get_layers, plot_history
from model.coef_weights_utils import get_coef_importance        ### 上下这三行 是进行修改后的，因为原来coef_weights_utils这个模块中是没有get_coef_importance这个函数，此函数主要是在model_utils.py文件中的，但是这样存在相互调用问题，因此将这个函数写到coef_weights_utils这个文件中


from utils.logs import DebugFolder
from config_path import *
base_path = BASE_PATH                   ### 现在的这个base_path是D:\DailyCode\P-Net\CodeModel\(Father)pnet_prostate_paper-published_to_zenodo\pnet_prostate_paper-published_to_zenodo\
from CeShi.OtherTest.ModelParamSave.ModelParam_Two import LayerEleGSEA, LayerEleRelationship

from os import makedirs
from os.path import join, exists

import tensorflow as tf






### 此文件，设立的Model类，其前提是这个网络模型已经构建好了，接下来这个模型类包括的功能主要是：训练拟合模型；根据输入的数据来获取该数据的预测结果；获取各层网络的输出结果   （在构建好原始模型之后，后续的一些训练以及预测等操作！）






### 构建网络模型的原函数
class Model(BaseEstimator):
    def __init__(self, build_fn, **sk_params):
        params = sk_params
        params['build_fn'] = build_fn
        self.set_params(params)


    ### 设置当前模型的一些核心参数
    def set_params(self, sk_params):     ### 设置这个模型训练时的一些基本的参数
        self.params = sk_params
        self.build_fn = sk_params['build_fn']
        self.sk_params = sk_params
        self.batch_size = sk_params['fitting_params']['batch_size']
        self.model_params = sk_params['model_params']
        self.nb_epoch = sk_params['fitting_params']['epoch']
        self.verbose = sk_params['fitting_params']['verbose']
        self.select_best_model = sk_params['fitting_params']['select_best_model']
        if 'save_gradient' in sk_params['fitting_params']:
            self.save_gradient = sk_params['fitting_params']['save_gradient']
        else:
            self.save_gradient = False

        if 'prediction_output' in sk_params['fitting_params']:
            self.prediction_output = sk_params['fitting_params']['prediction_output']
        else:
            self.prediction_output = 'average'

        if 'x_to_list' in sk_params['fitting_params']:
            self.x_to_list = sk_params['fitting_params']['x_to_list']
        else:
            self.x_to_list = False

        if 'period' in sk_params['fitting_params']:
            self.period = sk_params['fitting_params']['period']
        else:
            self.period = 10

        if 'max_f1' in sk_params['fitting_params']:
            self.max_f1 = sk_params['fitting_params']['max_f1']
        else:
            self.max_f1 = False

        if 'debug' in sk_params['fitting_params']:
            self.debug = sk_params['fitting_params']['debug']
        else:
            self.debug = False

        if 'feature_importance' in sk_params:
            self.feature_importance = sk_params['feature_importance']

        if 'loss' in sk_params['model_params']:
            self.loss = sk_params['model_params']['loss']
        else:
            self.loss = 'binary_crossentropy'          ### 损失函数默认为二项交叉熵

        if 'reduce_lr' in sk_params['fitting_params']:
            self.reduce_lr = sk_params['fitting_params']['reduce_lr']
        else:
            self.reduce_lr = False        ### 是否调整学习率（主要就是越往后，降低学习率）

        if 'lr' in sk_params['fitting_params']:
            self.lr = sk_params['fitting_params']['lr']
        else:
            self.lr = 0.001

        if 'reduce_lr_after_nepochs' in sk_params['fitting_params']:
            self.reduce_lr_after_nepochs = True
            self.reduce_lr_drop = sk_params['fitting_params']['reduce_lr_after_nepochs']['drop']
            self.reduce_lr_epochs_drop = sk_params['fitting_params']['reduce_lr_after_nepochs']['epochs_drop']
        else:
            self.reduce_lr_after_nepochs = False

        pid = os.getpid()
        timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}-{0:%S}'.format(datetime.datetime.now())
        self.debug_folder = DebugFolder().get_debug_folder()
        self.save_filename = os.path.join(self.debug_folder,
                                          sk_params['fitting_params']['save_name'] + str(pid) + timeStamp)
        self.shuffle = sk_params['fitting_params']['shuffle']
        self.monitor = sk_params['fitting_params']['monitor']
        self.early_stop = sk_params['fitting_params']['early_stop']

        self.duplicate_samples = False
        self.class_weight = None
        if 'duplicate_samples' in sk_params:
            self.duplicate_samples = sk_params['duplicate_samples']

        if 'n_outputs' in sk_params['fitting_params']:
            self.n_outputs = sk_params['fitting_params']['n_outputs']
        else:
            self.n_outputs = 1

        if 'class_weight' in sk_params['fitting_params']:
            self.class_weight = sk_params['fitting_params']['class_weight']
            logging.info('class_weight {}'.format(self.class_weight))


        ### 下边这两项关于梯度的参数是新加的！
        if 'gradients' in sk_params:
            self.gradients = sk_params['gradients']
        if 'gradients_Flag' in sk_params:
            self.gradients_Flag = sk_params['gradients_Flag']

        ### 下面的这个参数也是新加的  这个是来决定神经网络在测试阶段是否要进行消融！
        self.XiaoRongFlag = False





    def get_params(self, deep=False):
        return self.params

    ## 在这里构造回调函数（回调函数（callback）是在调用fit 时传入模型的一个对象（即实现特定方法的类实例），它在训练过程中的不同时间点都会被模型调用。它可以访问关于模型状态与性能的所有可用数据，还可以采取行动：中断训练、保存模型、加载一组不同的权重或改变模型的状态）
    ## 可以使用回调函数来观察训练过程中网络内部的状态和统计信息。
    ## 自己重写了一个回调函数，用以返回运行到当前的轮数时，这个模型的一些基本参数和信息的情况！
    def get_callbacks(self, X_train, y_train):
        callbacks = []
        print("上天保佑！现在传进来的这个训练数据是谁！", X_train)
        ### 下面开始读取当前模型的一些核心参数与信息，并将其保存至回调函数数组中
        # modify_callback = GradientModifier(layer_name='h0', X_train=X_train)
        # modify_callback = adjust_output_by_gradient(self.model)
        print("现在传进来的这个数据类型是怎样的！", type(X_train), type(y_train), type(X_train[0]), type(y_train[0]))
        print("现在这个模型的参数情况是怎样的！！", self.params)
        # modify_callback = GradientProcessingCallback(layer_name='h0', X_train=X_train, y_train=y_train)                 ## GradientProcessingCallback_Four
        # modify_callback = GradientProcessingCallback_Four(Model=self, layer_name='h0', gradients_Flag='gradients_Flag')
        # modify_callback = HyperParam(Model=self, layer_name='h0')
        modify_callback = GradientProcessingCallback_Five(layer_name='h1')




        # layer_name = 'h0'
        # def modify_layer_output(x):
        #     layer_output = x
        #     this_Layer = self.model.get_layer(layer_name).get_output_at(0)
        #     gradients = self.model.optimizer.get_gradients(self.model.total_loss, this_Layer)[0]
        #     gradients /= (K.sqrt(K.mean(K.square(gradients))) + K.epsilon())
        #     grad_values_normalise = K.relu(gradients)
        #     processed_outputs = layer_output * grad_values_normalise
        #     return processed_outputs
        #
        # # 创建LambdaCallback回调函数
        # modify_callback = LambdaCallback(
        #     on_batch_end=lambda batch, logs:
        #     K.function([self.model.input], [self.model.get_layer(layer_name).output])([self.model.input])[0].assign(
        #         modify_layer_output(self.model.get_layer(layer_name).output))
        # )



        # callbacks.append(modify_callback)
        logging.info("现在根据梯度信息对网络的输出值进行了注意力强化 ")
        # print("那么现在这个参数情况是怎样的！", self.params['gradients_Flag'])


        if self.reduce_lr:
            reduce_lr = ReduceLROnPlateau(monitor=self.monitor, factor=0.5,
                                          patience=2, min_lr=0.000001, verbose=1, mode='auto')
            logging.info("adding a reduce lr on Plateau callback%s " % reduce_lr)
            callbacks.append(reduce_lr)

        if self.select_best_model:
            saving_callback = ModelCheckpoint(self.save_filename, monitor=self.monitor, verbose=1, save_best_only=True,
                                              mode='max')
            logging.info("adding a saving_callback%s " % saving_callback)
            callbacks.append(saving_callback)

        if self.save_gradient:
            saving_gradient = GradientCheckpoint(self.save_filename, self.feature_importance, X_train, y_train,
                                                 self.nb_epoch,
                                                 self.feature_names, period=self.period)
            logging.info("adding a saving_callback%s " % saving_gradient)
            callbacks.append(saving_gradient)


        if self.early_stop:
            # early_stop = EarlyStopping(monitor=self.monitor, min_delta=0.01, patience=20, verbose=1, mode='min', baseline=0.6, restore_best_weights=False)
            early_stop = FixedEarlyStopping(monitors=[self.monitor], min_deltas=[0.0], patience=10, verbose=1,
                                            modes=['max'], baselines=[0.0])
            callbacks.append(early_stop)

        if self.reduce_lr_after_nepochs:
            # learning rate schedule     学习率的运行规则
            def step_decay(epoch, init_lr, drop, epochs_drop):        ## 这个是指学习率随着运行步骤的增加而不断地衰减
                initial_lrate = init_lr
                lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
                return lrate      ## 这个是得到经过一定的epoch之后所得到的衰减后的学习率

            from functools import partial
            step_decay_part = partial(step_decay, init_lr=self.lr, drop=self.reduce_lr_drop,
                                      epochs_drop=self.reduce_lr_epochs_drop)
            lr_callback = LearningRateScheduler(step_decay_part, verbose=1)
            callbacks.append(lr_callback)
        return callbacks       ### 返回出当前这个模型在当前的这个迭代轮数情况下他的一些基本参数和信息的情况！

    ## 从训练数据集中进行切割，划分出验证集
    def get_validation_set(self, X_train, y_train, test_size=0.2):       ### 获取验证集
        X_train1, X_validatioin, y_train_debug, y_validation_debug = train_test_split(X_train, y_train,
                                                                                      test_size=test_size,
                                                                                      stratify=y_train,
                                                                                      random_state=422342)
        training_data = [X_train1, y_train_debug]
        validation_data = [X_validatioin, y_validation_debug]
        print("划分之后的这个y标签的样子是什么样的！", len(y_train_debug), len(y_validation_debug))
        print("目前是在model/nn.py文件中，是从当前这个文件中划分训练集和验证集的吗？这个训练集和验证集的长度分别是！", len(training_data), len(validation_data))
        return training_data, validation_data


    ## 在模型训练预测中阈值也是一个超参数，（预测分数大于这个阈值归为一类，小于这个阈值则归为另一类），因此如何来选择一个合适的阈值，自己便构造下述这个函数来进行逐个的尝试
    ## 思想：随机产生一堆阈值，之后逐个阈值的进行尝试，获得在当前阈值的情况下，此时这个预测结果的他的实际得分，每个阈值都要算一下他此时的得分，最后来进行比较，进而来选出对预测结果而言，最终评价指标分数最高的那个阈值，即为最佳阈值
    def get_th(self, y_validate, pred_scores):
        thresholds = np.arange(0.1, 0.9, 0.01)      ### 返回一个等差数组（关于阈值的数组），数组的起点为0.1，终点为0.9，中间相差0.01
        print("目前为：nn.py 文件，当前随机生成的这个阈值是谁！真实的目标值又是谁！", thresholds, len(thresholds), len(y_validate))            ### 目前的这个y_validate就是一个数组，这个数组中的元素就是0、1      现在的这个y_validate就是这批数据的真实标签！
        scores = []
        for th in thresholds:
            y_pred = pred_scores > th        ### 此时这个 y_pred他的取值就是True或者False，就看当前的这个预测分数是否是大于这个阈值！  主要是做到二分类，就是看当前的预测概率是否超过这一目标！
            f1 = metrics.f1_score(y_validate, y_pred)        ## 计算F1值，输入的参数分别是真实的目标值以及分类器返回的估计目标
            precision = metrics.precision_score(y_validate, y_pred)
            recall = metrics.recall_score(y_validate, y_pred)
            accuracy = accuracy_score(y_validate, y_pred)
            score = {}
            score['accuracy'] = accuracy
            score['precision'] = precision
            score['f1'] = f1
            score['recall'] = recall
            score['th'] = th
            scores.append(score)      ### 获得在当前阈值的情况下，此时这个预测结果的他的实际得分，每个阈值都要算一下他此时的得分，最后来进行比较，进而来选出对预测结果而言，最终评价指标分数最高的那个阈值
        ret = pd.DataFrame(scores)
        print("目前为：nn.py 文件，这个整合后的预测结果分数为：\n", ret)           ### 现在ret为80组得分数据，每组数据是对应着其中一个阈值的情况下此时的计算得分，然后来看一下哪个得分最高进而来求出此时的那个阈值（还是来找寻最佳阈值的！）
        best = ret[ret.f1 == max(ret.f1)]
        th = best.th.values[0]
        print("现在是在model/nn.py文件当中，当前所求出的这个最佳阈值是谁！", th)
        return th     ### 此时获取了最佳的阈值


    ## 根据训练数据进行拟合模型，并将每一轮训练得到的结果进行保存，并优化整体模型（同时根据每一个epoch的预测结果来找寻最优的阈值等参数），返回是训练后的模型
    def fit(self, X_train, y_train, X_val=None, y_val=None, fold=None):   ## 该函数表示进入训练阶段，开始根据训练数据进行拟合

        ret = self.build_fn(**self.model_params)
        if type(ret) == tuple:
            self.model, self.feature_names = ret
        else:
            self.model = ret
        logging.info('目前为：nn.py 文件，start fitting')

        callbacks = self.get_callbacks(X_train, y_train)     ## 获取模型当前的信息
        print("目前是在model/nn.py文件当中，当前这个模型的中间层的输出结果是怎样的！", callbacks)

        if self.class_weight == 'auto':
            classes = np.unique(y_train)
            class_weights = class_weight.compute_class_weight('balanced', classes, y_train.ravel())
            class_weights = dict(list(zip(classes, class_weights)))
        else:
            class_weights = self.class_weight

        logging.info('目前为：nn.py 文件，class_weight {}'.format(class_weights))

        # speical case of survival 生存的特殊情况
        if y_train.dtype.fields is not None:
            y_train = y_train['time']



        print("目前是在model/nn.py文件中，原来传进来之前这个训练数据是多少！", len(X_train), X_train)
        print("目前为：nn.py 文件，测试一下这个self.debug，它关系着验证集能不能划分！", self.debug)
        # self.debug = True    ### 现在强行令这个self.debug为True，让他来划分一下这个验证集
        if self.debug:
            # train on 80 and validate on 20, report validation and training performance over epochs
            logging.info('目前为：nn.py 文件，dividing training data into train and validation with split 80 to 20')
            training_data, validation_data = self.get_validation_set(X_train, y_train, test_size=0.2)          ### 将原来的训练数据进行拆分，构造训练集与验证集     因为原来就是训练集和测试集合在一块的   因此说原来是1011总的数据样本，他拆成了909个（训练+验证）和102测试；现在在这里，他再次拆成727个训练与182个验证！
            X_train, y_train = training_data
            X_val, y_val = validation_data
            # print("现在这个训练与验证数据的标签是怎样的！", len(y_train), len(y_val), len(validation_data[1]), y_val)

        # print("目前是在model/nn.py文件中，现在是训练数据分割之前的样子！！", len(training_data), len(training_data[0]), len(training_data[1]), training_data)
        # print("目前是在model/nn.py文件中，现在经过处理之后的这个训练数据又是怎样的！", len(X_train), X_train)

        if self.n_outputs > 1:
            y_train = [y_train] * self.n_outputs
            y_val = [y_val] * self.n_outputs

        if not X_val is None:
            print("现在为model/nn.py文件中，此时的验证集不为空")
            validation_data = [X_val, y_val]
            # print("现在在传入之前，这个验证集的数据情况是怎样的！", len(training_data), len(validation_data), len(validation_data[0]), len(validation_data[1]), type(validation_data), validation_data)
        else:
            print("现在为model/nn.py文件中，此时的验证集为空！！！好奇怪！！")
            validation_data = []

        ## 下面为重头戏，开始拟合模型进行训练

        # ### 另一种的梯度计算方法
        # layer_name = 'h0'
        # this_Layer = self.model.get_layer(layer_name).get_output_at(0)
        # gradients = self.model.optimizer.get_gradients(self.model.total_loss, this_Layer)[0]  ### 获取指定层的梯度情况！
        # print("youyoucangtain!!!现在来测试一下，目前的这个梯度情况是谁！", self.gradients_Flag)
        # print("苍天保佑！目前算出来的这个梯度值的情况是怎样的！", self.model.total_loss, gradients)
        # self.gradients = gradients           ### 现在是把新算的梯度值给加进去！
        # self.gradients_Flag = True           ### 表明现在要使用这个梯度值进行计算了！！


        # print("看看能不能读到对应的梯度参数", self.model.get_config())
        # print("看看能不能读到对应的梯度参数", self.model.model_params['params']['gradients'], self.model.model_params['params']['gradients_Flag'])

        layer_names = [layer.name for layer in self.model.layers]
        print("目前这个神经网络模型中，各层神经网络的名字是怎样的！", layer_names)
        print("目前为：nn.py 文件，下面开始进行拟合！")             ### 就是在这根据传进来的那些模型参数，进行了300轮的模型的拟合！
        history = self.model.fit(X_train, y_train, validation_data=validation_data, epochs=self.nb_epoch,
                                 batch_size=self.batch_size,
                                 verbose=self.verbose, callbacks=callbacks,
                                 shuffle=self.shuffle, class_weight=class_weights)

        print("目前为：nn.py 文件，目前模型拟合结束")

        '''
        saving history
        '''
        plot_history(history.history, self.save_filename + '_validation')
        hist_df = pd.DataFrame(history.history)
        ### 下面这个是将每一轮迭代所得到的模型的训练结果写入到指定的这个文件中
        print("现在的这个文件名是谁！", self.save_filename)
        savefile_path = self.save_filename + '_train_history.csv'
        hist_df.to_csv(savefile_path)           ### 将数据保存到指定的文件路径下！
        print("目前这个文件已经保存成功了！")
        # with open(self.save_filename + '_train_history.csv', mode='w') as f:
        #     hist_df.to_csv(f)
        #     print("现在是在nn.py 文件，目前模型拟合结束，同时训练好的这个模型也都保存好了！")


        # if not X_val is None:
        pred_validate_score = self.get_prediction_score(X_train)    ## 根据当前训练的模型来获取一下对当前的这个训练数据的一个预测分数
        if self.n_outputs > 1:
            y_train = y_train[0]

        if self.max_f1:
            self.th = self.get_th(y_train, pred_validate_score)
            logging.info('目前为：nn.py 文件，prediction threshold {}'.format(self.th))



        ### 现在这些可解释性的方法计算老是会出错！因此这个位置暂时先注释掉！先全力提升知识融合的精度！
        IFKnowledge = True
        if hasattr(self, 'feature_importance') and IFKnowledge:       ## hasattr() 函数用于判断对象是否包含对应的属性    在这来计算可解释性，求各个节点的重要性分数
            print("目前为：nn.py 文件，来求各个节点的重要性分数")
            self.coef_ = self.get_coef_importance(X_train, y_train, target=-1,
                                                  feature_importance=self.feature_importance)


            ### $$$$$$$$$$$ 现在在得到当前这一折的可解释性分数之后，就直接对这一折的可解释结果进行保存！
            coef = self.get_named_coef()               ### 直接调用现成的函数获取各层的可解释结果！
            file_name = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-SHAP/'                                ### D:\DailyCode\P-Net\CodeModel\(Father)pnet_prostate_paper-published_to_zenodo\pnet_prostate_paper-published_to_zenodo\_logs\p1000\pnet\crossvalidation_average_reg_10_tanh\fs\coef.csv
            file_name = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-Grad/'             ### 现在尝试一下使用梯度的方法
            file_name = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-DeepLIFT-Kno/'            ### 现在尝试一下使用DeepLIFT的方法
            file_name = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-DeepLIFT/'  ### 现在尝试一下使用DeepLIFT的方法
            OriPath = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs/OriSave/'
            if not os.path.exists(OriPath):
                os.mkdir(OriPath)
            if not os.path.exists(file_name):
                os.mkdir(file_name)                     ### 判断一下这个文件夹是否存在，如果不存在的话，则自动进行创建
                print("现在开始创建这个文件了吗？？？")
            for c in list(coef.keys()):
                if type(coef[c]) == pd.DataFrame:
                    # print("现在的这个数据情况是谁！", type(coef[c]), coef[c])
                    # coef[c].to_csv(file_name + 'fold' + str(fold) + '_layer' + str(c) + '.csv')       ### 现在是对当前这一层的可解释结果进行保存！
                    # print("当前的这个文件名是谁！", file_name)
                    ### 现在其实可以考虑着一步到位，直接将得到的这个可解释结果进行一下排序！
                    coef_Dict = coef[c].to_dict(orient='split')
                    eleNames = coef_Dict['index']
                    eleData = coef_Dict['data']
                    FinalAllData = {}
                    for j in range(len(eleNames)):
                        nowName = eleNames[j]
                        if type(nowName) != str:         ### 此时就是表示输入数据，就类似于 ('STK32B', 'mut_important') 这种！  现在要把这个给合并了
                            nowName = nowName[0] + '_' + nowName[1]
                        nowdata = eleData[j][0]        ### 获取具体的数据内容！
                        FinalAllData[nowName] = nowdata
                    ### 接下来对得到的可解释性结果进行排序，按照解释分数取值从大到校排序
                    SortAllData = sorted(FinalAllData.items(), key=lambda x: x[1], reverse=True)       ### 现在就是按照value取值从大到校进行排序！
                    ### 注意啊！！现在这个排序后的数据SortAllData  他并不是一个字典，而是一个列表，列表中的每个元素是一个元组！


                    ### 下面这三点是来想办法向csv文件中写入数据
                    # 1. 创建文件对象
                    if not os.path.exists(file_name):
                        os.mkdir(file_name)
                    FinallFileName = file_name + 'coef_nn_fold_' + str(fold) + '_layer' + str(c) + '.csv'                  ### coef_nn_fold_0_layerh1.csv
                    print("目前最终的这个文件名是谁！", FinallFileName)
                    f = open(FinallFileName, 'w', newline='', encoding='utf-8')  ### 这个就是最终要写入的这个文件
                    # 2. 基于文件对象构建 csv写入对象
                    csv_writer = csv.writer(f)
                    # 3. 构建列表头     注！！输入层的数据他是需要多一列基因表达数据类型的！
                    RowHeader = ["element", "coef"]
                    csv_writer.writerow(RowHeader)         ## 将这个标题写到对应的文件中！
                    ## 遍历当前层中的各个元素以及对应的数据
                    for ele in SortAllData:
                        csv_writer.writerow(ele)  ## 将当前这一行写入！
                print("目前这一层的数据已经写入成功了！这一层网络层的名字叫做！", c)




        return self         ### 注现在的这个 self.coef_就是代表各层神经元节点对最终预测结果的重要性分数


    ## 下面是重写了get_coef_importance函数    现在来实现可解释性，来计算各个节点的重要性！
    def get_coef_importance(self, X_train, y_train, target=-1, feature_importance='deepexplain_grad*input'):

        coef_ = get_coef_importance(self.model, X_train, y_train, target, feature_importance, detailed=False)    ### 在这这个get_coef_importance是外调的
        print("现在是nn.py文件，目前所计算出来的关于训练数据的重要性情况是：", len(X_train), coef_)
        return coef_






    ### 下面这个是AAAI论文里的算法，主要是来求各个神经元在各个输出之间相似度的！相似度高的排名靠前！
    def get_CorrAll(self, xtest):

        # 计算相似度字典
        def get_similarity_dict(layer_outputs):
            similarity_dict = {}
            for i, layer_output in enumerate(layer_outputs):
                layer_similarity = {}
                for j in range(layer_output.shape[1]):
                    output_j = K.variable(layer_output[:, j])        ### 单就取所有测试数据中的某一列！
                    for k in range(j + 1, layer_output.shape[1]):
                        output_k = K.variable(layer_output[:, k])
                        similarity = K.eval(K.dot(output_j, output_k) / (
                                    K.sqrt(K.sum(K.square(output_j))) * K.sqrt(K.sum(K.square(output_k)))))
                        layer_similarity[(j, k)] = similarity
                similarity_dict[i] = layer_similarity
            return similarity_dict

        # 获取排名结果
        def get_ranking(similarity_dict):
            layer_ranking = {}
            for i, layer_similarity in similarity_dict.items():
                similarity_list = [(k[0], v) for k, v in layer_similarity.items()]
                similarity_list.sort(key=lambda x: x[1], reverse=True)
                ranking_list = [item[0] for item in similarity_list]
                layer_ranking[i] = ranking_list
            return layer_ranking


        ### 因为现在单个神经元的输出结果是单个数据，并不是一个向量，因此说在这不能来求向量的相似度，而是手动实现一下相似度求法，（假设有10个测试数据，那么它就会有10个当前神经元的输出结果，然后求一下这10个结果的方差！根据方差结果来进行排名！）
        def get_NeuralSimalrRanking(Result):
            Var_Result = {}      ### 现在这个字典中存储当前层中每个神经元在不同测试数据中输出结果的方差情况！   key是神经元的编号（0-...），value就是具体的结果值
            for j in range(Result.shape[1]):             ### 现在他的每一列就代表一个神经元在不同测试输入数据情况下的输出结果值！
                this_NeuralResult = Result[:, j]
                # print("目前想要来尝试一下这个神经元的输出结果是怎样的！", type(this_NeuralResult), this_NeuralResult)
                Var_Result[j] = np.var(this_NeuralResult)
            return Var_Result     ### 当前层各个神经元的方差情况值！





        ### 下面这个是来计算单层神经网络内部各个神经元节点他们的相似度排名！
        def GetOneLayerRank_Old(layerName, xtest):
            inputs, outputs = self.model.input, self.model.get_layer(layerName).output
            activation_model = NewModel(inputs=inputs,
                                        outputs=outputs)           ### 现在新构建一个模型，以原本的测试数据为输入来得到各层的输出结果！
            activations = activation_model.predict(xtest)         ### 根据测试数据来获取当前这一层的输出结果！
            # 计算同一神经元在不同测试数据下输出结果的相似性，并根据相似性的高低对各神经元进行排名
            print("现在当前这层网络的输出·结果是怎样的！", len(activations), len(activations[0]), activations)
            corrcoeffs = []
            # 计算各神经元在不同测试数据下输出的相关系数
            for i in range(len(activations)):
                corrcoeffs.append(np.corrcoef(activations[i].T))
            # 对相关系数进行平均，得到同一神经元在不同测试数据下输出结果的相似性
            mean_corrcoeffs = [np.mean(corrcoeff) for corrcoeff in corrcoeffs]
            rankings = np.argsort(mean_corrcoeffs)[::-1]
            return rankings            ### 将当前这一层的个个神经元给返回出来！



        ### 下面这个是来计算单层神经网络内部各个神经元节点他们的相似度排名！
        def GetOneLayerRank(layerName, xtest):
            inputs, outputs = self.model.input, self.model.get_layer(layerName).output
            activation_model = NewModel(inputs=inputs,
                                        outputs=outputs)           ### 现在新构建一个模型，以原本的测试数据为输入来得到各层的输出结果！
            activations = activation_model.predict(xtest)         ### 根据测试数据来获取当前这一层的输出结果！
            # 现在的这个activations是一个二维数组，他的维度是：测试数据数目*当前层神经元数目   （相当于每个神经元最终只输出了一个单个数据）
            # 计算同一神经元在不同测试数据下输出结果的相似性，并根据相似性的高低对各神经元进行排名
            print("现在当前这层网络的输出·结果是怎样的！", len(activations), len(activations[0]))                       ### 现在对于这个而言，他的行数就是测试样本的数目，他的列数就是当前这一层中神经元的数目！
            rankings = get_NeuralSimalrRanking(activations)           ### 现在读取当前层各个神经元的不同输出结果的方差情况（未排序）
            rankings = sorted(rankings.items(), key=lambda x: x[1])         ### 对当前的这个字典按照value值从小到大排序！！   （因为在这是要求他的方差是要越小越好的！）
            return rankings            ### 将当前这一层的个个神经元给返回出来！




        layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']          ### 定义具体的各个通路名！
        ### 现在一层层的过，求每一层网络中各个神经元的相似性排名！
        print("现在的这个特征的名字是谁！", type(self.feature_names), type(self.feature_names['h0']), np.array(self.feature_names['h0']))                           ### 现在这个self.feature_names 就是各层网络中各个元素的名字！
        LayerEleIDtoName = {}                 ### 现在这个字典来存放各层网络中各个神经元的元素他们名字与对应的位置编号；其中，这个key是这个元素在当前层中的位置编号（0、1、2、3）,编号是从0开始编的！！
        AllRank = {}
        for layer in layerNames:
            NowFeatureNames = np.array(self.feature_names[layer])
            NowLayerEleIDtoName = {}
            for i in range(len(NowFeatureNames)):
                NowLayerEleIDtoName[i] = NowFeatureNames[i]
            NowLayerRank = GetOneLayerRank(layer, xtest)             ### 现在得到的是一个列表，里面是一个个的元组，元组的第一项是这个元素的位置编号，元组第二项是这个元素的不同测试数据时输出激活值的方差数据
            ThislayerRank = {}
            for ele in NowLayerRank:
                EleName = NowLayerEleIDtoName[ele[0]]            ### 根据当前提供的元素编号来获取这个元素的名字！
                ThislayerRank[EleName] = ele[1]              ### 将这个元素的方差传递进去！
            AllRank[layer] = ThislayerRank                    ### 现在这个存放所有网络的排名！
        # print("最终所有网络的排名是怎样的！", AllRank)
        return AllRank





    ### 下面这个是老师新提出的想法，主要是来求各个神经元在不同的测试数据中，他的激活情况（在这假设他的输出值大于0就代表激活，否则就是不激活），计算一下在这些输入的测试样本的情况下，一个神经元的激活次数是多少，即算一下他的激活频率，激活频率高的，就认为这个神经元重要！
    def get_ActivationsAll(self, xtest, logictActParams=None):

        ### 因为现在单个神经元的输出结果是单个数据，并不是一个向量，现在传进来的这个Result是一个二维数组，他的行是表示一个个的样本，他的列是表示对于该样本情况下各个神经元的输出结果值！现在就是要看某个神经元所代表的这一列他的输出结果正值的情况是有多少个，正值出现的次数作为他的频率，频率越高就代表这个神经元越重要！
        def get_NeuralActivationsNumRanking(Result):
            Var_Result = {}      ### 现在这个字典中存储当前层中每个神经元在不同测试数据中输出结果的正值出现的次数！   key是神经元的编号（0-...），value就是具体的结果值
            for j in range(Result.shape[1]):             ### 现在他的每一列就代表一个神经元在不同测试输入数据情况下的输出结果值！
                this_NeuralResult = Result[:, j]         ### 现在这个输出结果他是一个数组，它是代表了一个具体的神经元在不同的测试样本中的输出结果！
                ### 下面这部分代码是来计算神经元被激活的次数
                # PosNum = 0                               ### 这个就表示当前神经元在各个测试样本中正值出现的次数！
                # for thisI in range(len(this_NeuralResult)):
                #     if this_NeuralResult[thisI] > 0:
                #         PosNum = PosNum + 1

                ### 下面这部分代码是来计算神经元整体被激活的程度！
                PosNum = 0                               ### 这个就表示当前神经元在各个测试样本中正值出现的次数！
                for thisI in range(len(this_NeuralResult)):
                    PosNum = PosNum + this_NeuralResult[thisI]

                Var_Result[j] = PosNum
            return Var_Result     ### 当前层各个神经元的激活频率情况值！


        ### 现在是要拿各个神经元的输出激活值来拟合一个逻辑回归，之后那这个逻辑回归的权重参数作为各个神经元的重要性分数！
        ### 因为现在单个神经元的输出结果是单个数据，并不是一个向量，现在传进来的这个Result是一个二维数组，他的行是表示一个个的样本，他的列是表示对于该样本情况下各个神经元的输出结果值！现在就是要看某个神经元所代表的这一列他的输出结果正值的情况是有多少个，正值出现的次数作为他的频率，频率越高就代表这个神经元越重要！
        def get_NeuralActivateLogictRanking(Result, logictActParams):
            Var_Result = {}  ### 现在这个字典中存储当前层中每个神经元在不同测试数据中输出结果的正值出现的次数！   key是神经元的编号（0-...），value就是具体的结果值
            y = logictActParams['Y']        ### 现在是来得到当前各个样本所对应的那个标签是谁！
            X = Result     ### 现在对于这个Result他的每一列就代表一个神经元在不同测试输入数据情况下的输出结果值！  每一行就代表一个样本   他现在是一个二维数组
            # 创建逻辑回归模型
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
            # 训练逻辑回归模型
            model.fit(X, y)
            # 获取训练后的权重（参数）w
            weights = model.coef_            ### 那么现在这个参数也是按照对应的顺序排好的
            weights = weights[0]
            for i in range(len(weights)):
                # Var_Result[i] = weights[i]
                Var_Result[i] = abs(weights[i])             ### 在这考虑一下这个权重加绝对值的情况，即哪怕是负权重，那他队最终的输出影响也是极大的！


            # Activations = get_NeuralActivationsNumRanking(Result)           ### 现在这个得到所有样本的激活分数的求和，他现在得到的仍然是一个字典形式！
            ### 接下来就要先对这个可解释的结果先进行一下归一化都归一化到0~1之间的范围之后再进行操作
            def normalize_dict_values(input_dict):
                # 提取原字典的值构成一个二维数组
                X = [[value] for value in input_dict.values()]
                # 使用MinMaxScaler对值进行归一化处理
                scaler = MinMaxScaler()  ### 现在这个 feature_range=(-1, 1)参数决定了最终的归一化范围，不写的话就是（0,1）
                normalized_values = scaler.fit_transform(X).flatten()
                # 构建归一化后的字典，键值对关系与原字典保持一致
                normalized_dict = {}
                for index, (key, value) in enumerate(input_dict.items()):
                    normalized_dict[key] = normalized_values[index]
                return normalized_dict

            # ### 将这两个字典中的元素都归一化到同样的范围，之后，开始拿激活分数值来修正当前的可解释结果！
            # Var_Result = normalize_dict_values(Var_Result)
            # Activations = normalize_dict_values(Activations)
            # for ele in Var_Result:
            #     if ele in Activations and abs(Activations[ele]-Var_Result[ele])>0.5:
            #         Var_Result[ele] = 0.3 * Activations[ele] + 0.7 * Var_Result[ele]

            return Var_Result  ### 当前层各个神经元的激活频率情况值！





        ### 现在是要拿各个神经元的输出激活值来拟合一个很简单的神经网络，之后拿这个神经网络中各神经元对应的权重参数作为原来大模型中各个神经元的重要性分数！
        ### 因为现在单个神经元的输出结果是单个数据，并不是一个向量，现在传进来的这个Result是一个二维数组，他的行是表示一个个的样本，他的列是表示对于该样本情况下各个神经元的输出结果值！现在就是要看某个神经元所代表的这一列他的输出结果正值的情况是有多少个，正值出现的次数作为他的频率，频率越高就代表这个神经元越重要！
        def get_NeuralActivateNNRankingTwo(Result, logictActParams):
            Var_Result = {}  ### 现在这个字典中存储当前层中每个神经元在不同测试数据中输出结果的正值出现的次数！   key是神经元的编号（0-...），value就是具体的结果值
            Alllabels = logictActParams['Y']        ### 现在是来得到当前各个样本所对应的那个标签是谁！
            Alldata = Result     ### 现在对于这个Result他的每一列就代表一个神经元在不同测试输入数据情况下的输出结果值！  每一行就代表一个样本   他现在是一个二维数组
            # 打乱二维数组的行顺序
            # # np.random.shuffle(Alldata)                ### 将里面的数据随机打乱，从而随即来划分训练集与测试集！！
            # data = Alldata[0:int(0.7*len(Alldata))]
            # testX = Alldata[int(0.7*len(Alldata)):len(Alldata)]
            # labels, testY = Alllabels[0:int(0.7*len(Alllabels))], Alllabels[int(0.7*len(Alllabels)):len(Alllabels)]
            #
            # data = Alldata[int(0.3*len(Alldata)):len(Alldata)]
            # testX = Alldata[0:int(0.3*len(Alldata))]
            # labels, testY = Alllabels[int(0.3*len(Alldata)):len(Alldata)], Alllabels[0:int(0.3*len(Alldata))]

            def split_data(X, Y, train_ratio):
                # 将X和Y合并成一个数据集
                data = list(zip(X, Y))
                # 设置不同的随机种子
                import time
                random.seed(time.time())
                # 将数据集随机打乱
                random.shuffle(data)
                # 计算训练集的长度
                train_len = int(len(data) * train_ratio)
                # 划分训练集和测试集
                train_set = data[:train_len]
                test_set = data[train_len:]
                # 解压缩训练集和测试集
                X_train, Y_train = zip(*train_set)
                X_test, Y_test = zip(*test_set)
                return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

            train_ratio = 0.7
            data, labels, testX, testY = split_data(Alldata, Alllabels, train_ratio)


            # valX, valY = Alldata[0:int(0.2*len(Alldata))], labels[0:int(0.2*len(labels))]             ### 获取对应的验证数据！！
            input_dim = len(data[0])          ### 现在就代表输入的这个样本数据他的对应的维度，其实就是这一层中神经元的数目！

            print("测试一下输入数据的类型！", type(data), type(labels), type(testX), type(testY))

            # 创建模型
            from keras.models import Sequential
            from keras.layers import Dense
            import matplotlib.pyplot as plt
            model = Sequential()
            model.add(Dense(10, activation='relu', input_shape=(input_dim,)))
            model.add(Dense(1, activation='sigmoid'))
            # 编译模型
            model.compile(optimizer='adam', loss='binary_crossentropy')

            # 训练模型
            history = model.fit(data, labels, epochs=100, batch_size=32)
            # 进行模型测试
            # valPredict = model.predict(valX)             ### 这个是对验证数据的预测结果！！
            # threshold = self.get_th(valY, valPredict)       ### 根据验证数据来获取最佳阈值！    他的输入参数有两个，一个是验证数据标签，一个是输入参数！
            threshold = 0.5
            predictions = model.predict(testX)        ### 现在根据测试样本来进行预测获取对应的预测结果！
            binary_predictions = np.where(predictions > threshold, 1, 0)           ### 根据获得的阈值来对预测结果进行二值化处理！
            # 计算Accuracy、Precision、Recall、AUPRC、AUC和F1值
            from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score, \
                roc_auc_score, f1_score
            accuracy = accuracy_score(testY, binary_predictions)
            precision = precision_score(testY, binary_predictions)
            recall = recall_score(testY, binary_predictions)
            auprc = average_precision_score(testY, predictions)
            auc = roc_auc_score(testY, predictions)
            f1 = f1_score(testY, binary_predictions)
            # test_loss, test_accuracy = model.evaluate(testX, testY, verbose=2)
            # print("测试集上的损失值：", test_loss)
            print("最终当前的这个测试集上的预测精度(accuracy, precision, recall, auprc, auc, f1)：", len(Alldata), input_dim, accuracy, precision, recall, auprc, auc, f1)
            ### 下面是想办法将这些精度结果进行保存！
            OriPath = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/AllActModelResult.csv'                ### 指定各轮的精度结果所要保存的地方
            ### 下面主要是将当前的这个激活网络的输出结果给写入到一个文件中保存下来
            import os
            def handle_ccf_file(ccf_file_path, data):
                Title = ['eleNum', 'accuracy', 'precision', 'recall', 'auprc', 'auc', 'f1']
                if not os.path.exists(ccf_file_path):  # 判断文件是否存在
                    print("此时当前的这个文件是不存在的！", data[0])
                    with open(ccf_file_path, 'w', newline="") as file:
                        # 2. 基于文件对象构建 csv写入对象
                        csv_writer = csv.writer(file)
                        csv_writer.writerow(Title)  ## 将这个标题写到对应的文件中！
                        csv_writer.writerow(data)  ## 将这个标题写到对应的文件中！
                        file.close()
                else:
                    with open(ccf_file_path, 'a+', newline="") as file:
                        # 2. 基于文件对象构建 csv写入对象
                        csv_writer = csv.writer(file)
                        csv_writer.writerow(data)  ## 将这个标题写到对应的文件中！
                        file.close()

            ActData = [input_dim, accuracy, precision, recall, auprc, auc, f1]      ### 这个就是激活网络的输出结果值了
            handle_ccf_file(OriPath, ActData)







            # # 绘制训练误差和验证误差的变化曲线
            # plt.plot(history.history['loss'], label='Training Loss')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.show()


            # 提取第一层连接权重和第二层连接权重
            weights_layer1 = model.layers[0].get_weights()[0]
            weights_layer2 = model.layers[1].get_weights()[0]


            ### 现在开始逐个特征的来遍历他们的权重分数！
            for i in range(len(weights_layer1)):
                FirstFeatureWeights = weights_layer1[i]  ### 属于当前这个特征的第一层连接权重的分数
                thisFeaScore = 0
                for j in range(len(weights_layer2)):
                    thisFeaScore = thisFeaScore + abs(FirstFeatureWeights[j] * weights_layer2[j][0])  ### 在这先加上绝对值！
                    # thisFeaScore = thisFeaScore + (FirstFeatureWeights[j] * weights_layer2[j][0])
                Var_Result[i] = thisFeaScore  ### 这个就得到当前这个神经元所对应的那个最终权重值的情况！

            return Var_Result  ### 当前层各个神经元的激活频率情况值！




        ### 现在是要拿各个神经元的输出激活值来拟合一个很简单的神经网络，之后拿这个神经网络中各神经元对应的权重参数作为原来大模型中各个神经元的重要性分数！
        ### 因为现在单个神经元的输出结果是单个数据，并不是一个向量，现在传进来的这个Result是一个二维数组，他的行是表示一个个的样本，他的列是表示对于该样本情况下各个神经元的输出结果值！现在就是要看某个神经元所代表的这一列他的输出结果正值的情况是有多少个，正值出现的次数作为他的频率，频率越高就代表这个神经元越重要！
        def get_NeuralActivateNNRanking(Result, logictActParams):
            Var_Result = {}  ### 现在这个字典中存储当前层中每个神经元在不同测试数据中输出结果的正值出现的次数！   key是神经元的编号（0-...），value就是具体的结果值
            labels = logictActParams['Y']        ### 现在是来得到当前各个样本所对应的那个标签是谁！
            data = Result     ### 现在对于这个Result他的每一列就代表一个神经元在不同测试输入数据情况下的输出结果值！  每一行就代表一个样本   他现在是一个二维数组

            ### 现在来创建对应的神经网络模型！
            # 定义神经网络的输入和输出
            input_dim = len(data[0])         ### 现在就代表输入的这个样本数据他的对应的维度，其实就是这一层中神经元的数目！
            output_dim = 1

            # 训练神经网络
            ### 这个神经网络分层来进行设置，浅层的那两层网络，这个FCN设计复杂一点！
            if len(data[0]) > 90000:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression()
                # 训练逻辑回归模型
                model.fit(data, labels)

                accuracy = model.score(data, labels)
                print("现在1对于这个逻辑回归而言这个模型的预测准确率:", accuracy)
                # 获取训练后的权重（参数）w
                weights = model.coef_  ### 那么现在这个参数也是按照对应的顺序排好的
                weights = weights[0]
                for i in range(len(weights)):
                    Var_Result[i] = abs(weights[i])  ### 在这考虑一下这个权重加绝对值的情况，即哪怕是负权重，那他队最终的输出影响也是极大的！
                return Var_Result  ### 当前层各个神经元的激活频率情况值！


            else:
                NeuronNum = 10    ### 这个来指定一下中间层这个神经元的数目！
                epochs = 200
                batch_size = 20


            # 定义神经网络的输入
            X = tf.placeholder(tf.float32, shape=[None, input_dim])
            y = tf.placeholder(tf.float32, shape=[None, output_dim])

            # 定义神经网络的权重和偏置
            weights_1 = tf.Variable(tf.random_normal([input_dim, NeuronNum]))
            bias_1 = tf.Variable(tf.random_normal([NeuronNum]))
            weights_2 = tf.Variable(tf.random_normal([NeuronNum, output_dim]))

            # weights_2 = tf.Variable(tf.random_normal([input_dim, output_dim]))
            bias_2 = tf.Variable(tf.random_normal([output_dim]))

            # 定义神经网络的第一层和第二层
            hidden_layer = tf.nn.relu(tf.matmul(X, weights_1) + bias_1)
            output_layer = tf.sigmoid(tf.matmul(hidden_layer, weights_2) + bias_2)

            # output_layer = tf.sigmoid(tf.matmul(X, weights_2) + bias_2)                 ### 现在只定义一层网络，只有一个神经元
            # output_layer = tf.sigmoid(tf.matmul(X, weights_2))  ### 现在只定义一层网络，只有一个神经元   而且还没有b参数！

            predicted_labels = tf.round(output_layer)

            # 定义损失函数和优化器
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output_layer))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

            # 定义准确率
            correct_prediction = tf.equal(predicted_labels, y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))






            ## 下面开始进行训练！
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for epoch in range(epochs):
                    # 随机从数据中选取一个batch进行训练
                    indices = np.random.randint(low=0, high=data.shape[0], size=batch_size)
                    batch_X = data[indices]
                    batch_y = labels[indices]

                    # 运行优化器进行训练，并计算准确率
                    _, acc = sess.run([optimizer, accuracy], feed_dict={X: batch_X, y: batch_y})
                    ### 这样吧！！考虑各层在训练时分层来整！浅层因为特征多，纬度高比较难训练，因此多训练几步！！
                    # if acc > 0.9:
                    #     print("此时的训练精度已经达到要求了，要终止训练了，那么此时的准确度是：", acc)
                    #     break
                    # print("Epoch {}: Accuracy = {:.3f}".format(epoch + 1, acc))

                # 训练完成后计算整个数据集的准确率
                test_acc = sess.run(accuracy, feed_dict={X: data, y: labels})
                print("Test Accuracy = {:.3f}".format(test_acc))

                # 获取训练后的权重值
                weights_1_value = sess.run(weights_1)
                weights_2_value = sess.run(weights_2)

            # 整理连接权重值成字典的形式
            Feature_weights_dict, Neuron_weights_dict = {}, {}
            Neuron_weights = []

            # for i in range(input_dim):
            #     Var_Result[i] = abs(weights_2_value[i][0])


            ### 现在下面的这个字典得到的结果就是输入的每个特征与第一层的各个神经元之间的的那个连接权重所组成的一个数组
            for i in range(input_dim):
                Feature_weights_dict["Feature{}".format(i + 1)] = []
                for j in range(NeuronNum):
                    Feature_weights_dict["Feature{}".format(i + 1)].append(weights_1_value[i][j])

            ### 下面这个分别代表了第一层的各个神经元与地输出神经元之间的那个连接权重的取值！    代表了各个神经元他的那个连接权重的取值
            for i in range(NeuronNum):
                Neuron_weights_dict["neuron{}".format(i + 1)] = weights_2_value[i][0]
                Neuron_weights.append(weights_2_value[i][0])

            ### 现在来处理一下每个特征的对应的权重情况！
            for i in range(input_dim):
                flag = 'Feature' + str(i + 1)
                FeatureWeigth = Feature_weights_dict[flag]
                Final = 0  ### 这个是对应这个最终这个特征他的权重值求和的情况！是所有的权重值进行的求和！
                for j in range(len(FeatureWeigth)):
                    Final = Final + abs(FeatureWeigth[j] * Neuron_weights[j])  ### 用跟这个神经元相连的那个权重乘上这个神经元所对殷大哥那个概率来得到最终的这个权杖取值！                ### 待会可以尝试一下不加绝对值走一趟！同时取消b这个参数！
                    # Final = Final + abs(FeatureWeigth[j]) + abs(Neuron_weights[j])

                Var_Result[i] = Final       ### 这个就得到当前这个神经元所对应的那个最终权重值的情况！


            return Var_Result  ### 当前层各个神经元的激活频率情况值！




        ### 下面这个是来计算单层神经网络内部各个神经元节点他们的相似度排名！
        def GetOneLayerRank(layerName, xtest):
            inputs, outputs = self.model.input, self.model.get_layer(layerName).output
            activation_model = NewModel(inputs=inputs,
                                        outputs=outputs)           ### 现在新构建一个模型，以原本的测试数据为输入来得到各层的输出结果！
            activations = activation_model.predict(xtest)         ### 根据测试数据来获取当前这一层的输出结果！
            # 现在的这个activations是一个二维数组，他的维度是：测试数据数目*当前层神经元数目   （相当于每个神经元最终只输出了一个单个数据）
            # 计算同一神经元在不同测试数据下输出结果的相似性，并根据相似性的高低对各神经元进行排名
            print("现在当前这层网络的输出·结果是怎样的！", len(activations), len(activations[0]))                       ### 现在对于这个而言，他的行数就是测试样本的数目，他的列数就是当前这一层中神经元的数目！
            if logictActParams is None:
                rankings = get_NeuralActivationsNumRanking(activations)  ### 现在读取当前层各个神经元的不同输出结果的方差情况（未排序）
            else:
                # rankings = get_NeuralActivateLogictRanking(activations, logictActParams)  ### 现在读取当前层各个神经元的不同输出结果的激活值，之后拿这些激活值来拟合一个逻辑回归，进而得到各个激活值所代表的那个神经元所对应的权重参数，然后拿这个权重参数来表示他们的重要性排名情况！（未排序）
                # rankings = get_NeuralActivateNNRanking(activations, logictActParams)
                rankings = get_NeuralActivateNNRankingTwo(activations, logictActParams)
            rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)         ### 对当前的这个字典按照value值从大到小排序！！   （因为在这是要求他被激活的次数越多越好的！）
            return rankings            ### 将当前这一层的个个神经元给返回出来！




        layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']          ### 定义具体的各个通路名！
        ### 现在一层层的过，求每一层网络中各个神经元的相似性排名！
        print("现在的这个特征的名字是谁！", type(self.feature_names), type(self.feature_names['h0']), np.array(self.feature_names['h0']))                           ### 现在这个self.feature_names 就是各层网络中各个元素的名字！
        LayerEleIDtoName = {}                 ### 现在这个字典来存放各层网络中各个神经元的元素他们名字与对应的位置编号；其中，这个key是这个元素在当前层中的位置编号（0、1、2、3）,编号是从0开始编的！！
        AllRank = {}
        for layer in layerNames:
            NowFeatureNames = np.array(self.feature_names[layer])
            NowLayerEleIDtoName = {}
            for i in range(len(NowFeatureNames)):
                NowLayerEleIDtoName[i] = NowFeatureNames[i]         ## 现在这个字典他的key是神经元所代表的元素所在的位置编号，value就是这个元素具体的名字！
            NowLayerRank = GetOneLayerRank(layer, xtest)             ### 现在得到的是一个列表，里面是一个个的元组，元组的第一项是这个元素的位置编号，元组第二项是这个元素的不同测试数据时输出激活的频率
            ThislayerRank = {}
            for ele in NowLayerRank:
                EleName = NowLayerEleIDtoName[ele[0]]            ### 根据当前提供的元素编号来获取这个元素的名字！
                ThislayerRank[EleName] = ele[1]              ### 将这个元素的方差传递进去！
            AllRank[layer] = ThislayerRank                    ### 现在这个存放所有网络的排名！
        # print("最终所有网络的排名是怎样的！", AllRank)
        return AllRank








    def get_CorrRanking(self, xtest):
        ## **** 现在来看一下交叉相似性的问题！
        layer_outputs = [layer.output for layer in self.model.layers]  # 获取所有层的输出
        # 以模型的输入作为输入，输出各层的输出
        # activation_model = keras.models.Model(inputs=self.model.input, outputs=layer_outputs)  ### 现在新构建一个模型，以原本的测试数据为输入来得到各层的输出结果！
        inputs, outputs = self.model.input, layer_outputs
        activation_model = NewModel(inputs=inputs,
                                              outputs=outputs)  ### 现在新构建一个模型，以原本的测试数据为输入来得到各层的输出结果！
        activations = activation_model.predict(xtest)
        # 计算同一神经元在不同测试数据下输出结果的相似性，并根据相似性的高低对各神经元进行排名
        corrcoeffs = []
        # 计算各神经元在不同测试数据下输出的相关系数
        for i in range(len(activations)):
            corrcoeffs.append(np.corrcoef(activations[i].T))
        # 对相关系数进行平均，得到同一神经元在不同测试数据下输出结果的相似性
        mean_corrcoeffs = [np.mean(corrcoeff) for corrcoeff in corrcoeffs]
        rankings = np.argsort(mean_corrcoeffs)[::-1]
        print("那么目前的这个排名情况是怎样的呢！！", rankings)
        return rankings







    ### 加载对应的预测模型进行预测   他先产生一个预测分数（概率值），之后根据目前的阈值情况，将预测结果归类，看是0还是1
    def predict(self, X_test, XiaoRongParams=None):    ### 加载对应的预测模型进行预测             ### 可以在这后面加一个默认参数，表示当前是五折运算中的第几折，以及当前当前消融的比例！
        if self.select_best_model:
            logging.info("目前为：nn.py 文件，loading model %s" % self.save_filename)
            self.model.load_weights(self.save_filename)    ### 加载以前训练好的模型

        print("那么目前所传过来的这个消融参数是谁！", XiaoRongParams)

        prediction_scores = self.get_prediction_score(X_test, XiaoRongParams)    ### 调用训练好的这个模型，根据测试数据来预测一下这些数据对应的预测分数      这个函数后面其实也是可以跟上一些默认参数的！

        std_th = .5
        if hasattr(self, 'th'):    ### 判断一下，此时有没有训练出最佳的阈值，如果还没有的话，则根据此时模型的损失函数来选择对应的阈值
            std_th = self.th
        elif self.loss == 'hinge':
            std_th = 1.
        elif self.loss == 'binary_crossentropy':
            std_th = .5

        if self.loss == 'mean_squared_error':
            prediction = prediction_scores
        else:
            prediction = np.where(prediction_scores >= std_th, 1., 0.)      ## 此时就是判断一下prediction_scores是否大于阈值，如果大于就是1，如果不大于的话那么预测结果就是0     而如果损失函数是MSE的话他自动会将预测分数进行归零，归1，不用再走这一步判断了！

        return prediction


    ## 就是调用模型，输入相应的数据，得出该数据对应的预测分数
    def get_prediction_score(self, X, XiaoRongParams=None):

        print("传到具体的这个函数中的这个参数是谁！", XiaoRongParams)

        ### 下面这一块是来进行梯度加权的！
        gradWeight = False
        if gradWeight:
            ### 下面为测试部分， 在这可以尝试着修改某层网络中的相应的参数值
            AllName = ['h0', 'h1', 'h2', 'h3', 'h4']           ### 在这，他存放所有要处理的层的名字   ['h1']
            for layer_name in AllName:        ### 每一层的来进行处理
                # 获取指定层的梯度
                # layer_name = 'h1'     ### 先试试第一层通路层
                ### 另一种的梯度计算方法
                layer = self.model.get_layer(layer_name)
                ### 求出对应的梯度值
                layer_output = self.model.get_layer(layer_name).output
                gradients = K.gradients(self.model.total_loss, layer_output)[0]  ### 求出对应的梯度值
                ### 将梯度值进行进一步的处理！
                gradients /= (K.sqrt(K.mean(K.square(gradients))) + K.epsilon())  ### 将梯度值进行处理，进行归一化操作！
                grad_values_normalise = K.relu(gradients)
                ### 将处理后的梯度值进行赋值
                layer.attentionWeights = grad_values_normalise  ### 每隔10个epoch便来将当前的



        ### 下面这块是来对模型内部的一些神经元进行消融处理的     直接来看 neuronAblation来决定是否进行消融操作！
        if self.XiaoRongFlag is True:
            neuronAblation = True
        else:
            print("当前传进来的这个消融标志是谁！", self.XiaoRongFlag)
            neuronAblation = False


        if neuronAblation:


            ### 第一阶段： 现在来构造各层神经网络中每个神经元元素他的W参数与B参数在参数列表中的起止位置   这个是专门针对基因层
            def GetNeuronParamLoc_GeneLayer(neuronEle, paramsLoc):
                ### 第一阶段： 现在来构造各层神经网络中每个神经元元素他的W参数与B参数在参数列表中的起止位置
                Wbegin, Bbegin = 0, 0
                for ele in neuronEle:
                    paramsLoc[ele] = [Wbegin, Wbegin + 3, Bbegin]
                    Wbegin, Bbegin = Wbegin + 3, Bbegin + 1
                return paramsLoc         ### 现在这个字典paramsLoc中的各个元素都是按照当前网络层中的各个元素的顺序排布的！

            ### 第一阶段： 现在来构造各层神经网络中每个神经元元素他的W参数与B参数在参数列表中的起止位置     这个1专门针对通路层！
            def GetNeuronParamLoc_PathwayLayer(neuronEle, paramsLoc, j):       ### 在这，这个j就表示fileIndex，表明当前进入到第几个通路层
                ### 第一阶段： 现在来构造各层神经网络中每个神经元元素他的W参数与B参数在参数列表中的起止位置
                ### 下面先来看看当前层中每个元素他的输入有多少个参数！
                LayerRelationship = LayerEleRelationship()
                if j == 1:  ### 此时代表第一层通路层！
                    RelationDict = LayerRelationship.getLayer1()
                elif j == 2:
                    RelationDict = LayerRelationship.getLayer2()
                elif j == 3:
                    RelationDict = LayerRelationship.getLayer3()
                elif j == 4:
                    RelationDict = LayerRelationship.getLayer4()
                elif j == 5:
                    RelationDict = LayerRelationship.getLayer5()
                else:
                    print("说明此时出错了！索引变量j超过了一定的范围！", j)
                Wbegin, Bbegin = 0, 0
                for ele in neuronEle:
                    WParamNum = len(RelationDict[ele])      ### 就是来看一下当前这个通路有多少个子通路，进而确定他有多少个输入参数！
                    if WParamNum > 0:            ### 此时表明当前这个元素是有输入参数的！
                        paramsLoc[ele] = [Wbegin, Wbegin + WParamNum, Bbegin]
                    else:                        ### 此时表明当前这个元素没有输入参数！  此时WParamNum 就是0
                        paramsLoc[ele] = [Bbegin]
                    Wbegin, Bbegin = Wbegin + WParamNum, Bbegin + 1

                return paramsLoc         ### 现在这个字典paramsLoc中的各个元素都是按照当前网络层中的各个元素的顺序排布的！



            ### 第二阶段：现在神经元参数的起止位置构造好了，开始来读取当前这一层中要删除的元素有哪些！  就是来求出具体要消融哪些元素！
            def GetXiaoRongEle(neuronEle, fileIndex, XiaoRongParams):
                ### 第二阶段：现在神经元参数的起止位置构造好了，开始来读取当前这一层中要删除的元素有哪些！  就是来求出具体要消融哪些元素！
                BiLi, Fold = XiaoRongParams['BiLi'], XiaoRongParams['Fold']               ### 这个是决定当前要消掉百分之多少的神经元！
                XiaoRongDirect = XiaoRongParams['Direct']  ### 现在这个参数是来决定待会来进行消融的时候是应该正着消（先消排名靠前的神经元），还是应该反着消（先消排名靠后的神经元）
                Num = int(len(neuronEle) * BiLi)  ### 现在这个是要删除的神经元的数目
                # ExplainScoreFilePath = base_path + '/CeShi/FunctionProcess&Result/ExplainResultProcess/Result/RankResult/NoKnowledge/Sort/RankAnalysis-'             ### 现在这个是不加知识的排序结果  _logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs/TempSave/Sortfold0_layerh0.csv
                ExplainScoreFilePath = base_path + '/CeShi/FunctionProcess&Result/ExplainResultProcess/Result/RankResult/WithKnowledge/Sort/RankAnalysis-'           ### 现在这个是不加知识的排序结果         D:\DailyCode\P-Net\CodeModel\(Father)pnet_prostate_paper-published_to_zenodo\pnet_prostate_paper-published_to_zenodo\_logs\p1000\pnet\crossvalidation_average_reg_10_tanh\fs\TempSave\Sortfold0_layerh0.csv
                ExplainScoreFilePath = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs/TempSave/Sortfold' + str(Fold) + '_layerh'            ### 现在来指定当前这一折所得到的哪个可解释性结果！

                ExplainScoreFilePath = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-Active/coef_nn_fold_' + str(Fold) + '_layerh'            ### 现在这个来选择TA方法1； coef_nn_fold_4_layerh5.csv
                ExplainScoreFilePath = base_path + '/CeShi/ZhongJianResult/23.8.30-AAAI-fs/fs-AAAI-1/coef_nn_fold_' + str(Fold) + '_layerh'            ### 现在这个来选择之前所保存的那个AAAI方法1！  CeShi/ZhongJianResult/23.8.30-AAAI-fs/fs-AAAI-1
                ExplainScoreFilePath = base_path + XiaoRongParams['Path'] + 'coef_nn_fold_' + str(Fold) + '_layerh'  ### 通过传过来的参数来指定当前是选用

                data = pd.read_csv(ExplainScoreFilePath + str(fileIndex) + '.csv')  # 读取文件中所有数据
                ### 目前假设现在的存储的这个可解释结果没有进行排序！


                ExplainScore = np.array(data[['element']])  # 读取具体的分数元素名字那一列（现在各个元素已经按照1重要性得分排过序了，直接读进来就行！！）
                if XiaoRongDirect == "Pos":       ### 说明此时应该正向消融！
                    XiaoRongEle = ExplainScore[0:Num]        ### 现在这个数据就是当前这一网络层要消融掉的元素都有哪些       现在这个是正向消融！  正向消融的时候是需要精度迅速下降
                else:                             ### 说明此时应该反向消！
                    XiaoRongEle = ExplainScore[len(ExplainScore)-Num:len(ExplainScore)]       ### 现在这个数据就是当前这一网络层要消融掉的元素都有哪些       现在这个是反向消融！  反向消融的时候是需要精度缓慢下降

                ### 因为现在这个XiaoRongEle是一个数组，而数组中的子元素还是一个数组，因此要将里面那个子元素的那个数组给提取出来作为一个具体的元素
                Temp = []
                for ele in XiaoRongEle:
                    Temp.append(ele[0])
                XiaoRongEle = Temp  ### 现在的这个XiaoRongEle 就是当前这一层网络中最终要消融的哪些元素！
                return XiaoRongEle



            ### 第三阶段：遍历那些要消融的元素开始进行消融！

            ### 下面这个是实际进行消融的那个函数！
            def XiaoRongProcess(layer, XiaoRongEle, paramsLoc):
                weights = layer.get_weights()
                # print("那么目前的这个参数列表是怎样的！！", len(paramsLoc))
                for ele in XiaoRongEle:
                    if ele not in paramsLoc:
                        continue
                    if len(paramsLoc[ele]) > 1:  ### 此时就说明当前的这个神经元元素他是有输入元素的！   因为它即使没有输入元素的话，他的对应的Value列表中至少也会有一个偏置B参数
                        WIndexBegin, WIndexEnd, BIndex = paramsLoc[ele][0], paramsLoc[ele][1], paramsLoc[ele][2]  ### 现在是来找到当前这个元素他的输入参数在参数列表中的起止下标以及偏置参数在参数列表中的位置
                        weights[0][WIndexBegin: WIndexEnd] = 0  # 权重矩阵的每一行都对应一个神经元的连接权重，[:, neuron_index] 将该神经元的权重设置为零    现在就是对当前的这个神经元的输入参数设置为0
                        weights[1][BIndex] = 0  ### 将对应的偏置参数也给设置为0
                    else:  ### 此时说明当前的这个神经元他是没有输入的，只有对应的输出
                        BIndex = paramsLoc[ele][0]
                        weights[1][BIndex] = 0  ### 将对应的偏置参数也给设置为0
                return weights


            ### 目前来读各层网络前后映射关系的那个CSV文件表！
            def ReadCSVFile(testFilePath):
                with open(testFilePath, 'r') as file:
                    # 创建CSV读取器
                    reader = csv.reader(file)
                    # 读取标题行
                    headers = next(reader)
                    headers = headers[1:len(headers)]
                    # 打印标题行
                    # print("当前的这个标题行是谁！！", len(headers))
                    # 读取每一行数据
                    Rows = []
                    for row in reader:
                        # 获取标题列数据
                        first_column = row[0]
                        # 打印标题列数据
                        Rows.append(first_column)
                    # print("最终的这个标题列是谁！", len(Rows))
                return Rows, headers







            ### 下面为测试部分， 在这可以尝试着修改某层网络中的相应的参数值
            AllName = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']  ### 在这，他存放所有要处理的层的名字   ['h1']
            fileIndex = 0
            for layer_name in AllName:  ### 每一层的来进行处理
                # print("现在是开始进行消融了！当前这个是第几层！", layer_name)
                ### 下面这个用来读取当前这层网络中各个元素的前后分布顺序！
                if fileIndex == 0 or fileIndex == 1:            ### 基因层与第一层通路层都是打开同一个网络层的前后关系文件！
                    # testFilePath = base_path + '\CeShi\OtherTest\GenePathMapp\Layer0.csv'
                    testFilePath = base_path + '\CeShi\OtherTest\GenePathMappTwo_Ceshi\Layer0.csv'
                    Gene, Pathway = ReadCSVFile(testFilePath)

                else:
                    # testFilePath = base_path + '\CeShi\OtherTest\GenePathMapp\Layer' + str(fileIndex-1) + '.csv'
                    testFilePath = base_path + '\CeShi\OtherTest\GenePathMappTwo_Ceshi\Layer' + str(fileIndex - 1) + '.csv'
                    Gene, Pathway = ReadCSVFile(testFilePath)

                ### 接下来想办法构造一个字典 key是每个神经元，value包含当前神经元的W参数在W列表中的起止下标，以及b参数在偏置列表中的下标！
                paramsLoc = {}
                layer = self.model.get_layer(layer_name)
                ###  下面这个第一层的基因层要单独的进行处理一下！
                if layer_name == 'h0':
                    neuronEle = Gene         ### 现在这个是当前这层网络中各个元素的排列列表！
                    ### 第一阶段： 现在来构造各层神经网络中每个神经元元素他的W参数与B参数在参数列表中的起止位置
                    paramsLoc = GetNeuronParamLoc_GeneLayer(neuronEle, paramsLoc)

                    ### 第二阶段：现在神经元参数的起止位置构造好了，开始来读取当前这一层中要删除的元素有哪些！  就是来求出具体要消融哪些元素！
                    XiaoRongEle = GetXiaoRongEle(neuronEle, fileIndex, XiaoRongParams)

                    ### 第三阶段：遍历那些要消融的元素开始进行消融！
                    weights = XiaoRongProcess(layer, XiaoRongEle, paramsLoc)

                    self.model.get_layer(layer_name).set_weights(weights)       ### 将最终处理后的权重参数再重新返回设置到网络当中！

                ###  剩下的所有的通路层都是这样相同的操作！
                else:
                    neuronEle = Pathway         ### 现在这个是当前这层网络中各个元素的排列列表！
                    ### 第一阶段： 现在来构造各层神经网络中每个神经元元素他的W参数与B参数在参数列表中的起止位置
                    paramsLoc = GetNeuronParamLoc_PathwayLayer(neuronEle, paramsLoc, fileIndex)
                    # print("现在是测试部分！！看看读取的这个字典是什么样子！", paramsLoc)

                    ### 第二阶段：现在神经元参数的起止位置构造好了，开始来读取当前这一层中要删除的元素有哪些！  就是来求出具体要消融哪些元素！
                    XiaoRongEle = GetXiaoRongEle(neuronEle, fileIndex, XiaoRongParams)

                    ### 第三阶段：遍历那些要消融的元素开始进行消融！
                    weights = XiaoRongProcess(layer, XiaoRongEle, paramsLoc)

                    self.model.get_layer(layer_name).set_weights(weights)       ### 将最终处理后的权重参数再重新返回设置到网络当中！

                fileIndex = fileIndex + 1      ### 开始读下一个前后关系的文件

        self.XiaoRongFlag = False             ### 当前这波消融操作处理好之后，那么之后（主要是predict_proba()函数，他也会待用当前这个函数)，就不用再进行消融了！






        prediction_scores = self.model.predict(X)         ### 现在在这所得到的这个预测分数的形状为（6, 727）   现在这里一共是有6组数据，每组数据内部又含有727个预测值，之所以是六组，是因为中间六层隐藏层，每一层都给一个预测结果， 现在输入的样本大小是727个，因此对每个样本各自得到一个预测结果！
        print("当前所处文件为: nn.py 目前这个模型输出的预测分数是多少！", len(prediction_scores), len(prediction_scores[0]), len(prediction_scores[1]), np.max(prediction_scores[0]), np.min(prediction_scores[0]), np.average(prediction_scores[0]))
        if (type(prediction_scores) == list):
            if len(prediction_scores) > 1:
                if self.prediction_output == 'average':
                    prediction_scores = np.mean(np.array(prediction_scores), axis=0)
                else:
                    prediction_scores = prediction_scores[-1]

        print("当前所处文件为: nn.py 目前所求出的预测分数的shape为：", np.array(prediction_scores).shape)
        print("白文超加油！！现在经过处理之后的这个预测分数情况是怎样的！！", len(prediction_scores), len(prediction_scores[0]), len(prediction_scores[1]), np.max(prediction_scores[0]), np.min(prediction_scores[0]), np.average(prediction_scores[0]))
        return np.array(prediction_scores)



    ## 返回的是一个二维矩阵（当前样本属于各类别的预测概率）。返回对各个样本预测概率的结果，因为目前是来做二分类的，那么返回的这个ret就是一个二维数组，就两行，横坐标代表每个样本，纵坐标代表每个类别，这样的话就得出该样本属于每个类别的概率是多少！
    def predict_proba(self, X_test):    ### 这个是根据输入的测试的数据，输出一下当前的这个数据他的预测概率是怎么样的！
        prediction_scores = self.get_prediction_score(X_test)
        print("当前所处文件为: nn.py ，现在是predict_proba函数，目前这个模型输出的预测分数是多少！", len(prediction_scores), len(prediction_scores[0]), len(prediction_scores[1]), np.max(prediction_scores[0]), np.min(prediction_scores[0]), np.average(prediction_scores[0]))
        if type(X_test) is list:
            n_samples = X_test[0].shape[0]
        else:
            n_samples = X_test.shape[0]
        ret = np.ones((n_samples, 2))
        ret[:, 0] = 1. - prediction_scores.ravel()             ## 这个ravel()函数就是将数组为度拉成一维数组
        ret[:, 1] = prediction_scores.ravel()
        print("当前所处文件为: nn.py 目前统计各个样本他被预测属于各个类别的概率，这个概率数组的形状为：", ret.shape)
        return ret        ## 返回对各个样本预测概率的结果，因为目前是来做二分类的，那么返回的这个ret就是一个二维数组，就两行，横坐标代表每个样本，纵坐标代表每个类别，这样的话就得出该样本属于每个类别的概率是多少！


    ## 当前预测结果与实际相比的准确率
    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return accuracy_score(y_test, y_pred)



    ## 获得当前指定层的输出！
    def get_layer_output(self, layer_name, X):
        layer = self.model.get_layer(layer_name)
        inp = self.model.input
        functor = K.function(inputs=[inp, K.learning_phase()], outputs=[layer.output])  # evaluation function
        layer_outs = functor([X, 0.])
        return layer_outs


    ## 获得当前模型中，每一个模型输出的一个汇总
    def get_layer_outputs(self, X):
        inp = self.model.input
        layers = get_layers(self.model)[1:]
        layer_names = []
        for l in layers:
            layer_names.append(l.name)
        outputs = [layer.get_output_at(0) for layer in layers]  # all layer outputs
        functor = K.function(inputs=[inp, K.learning_phase()], outputs=outputs)  # evaluation function
        layer_outs = functor([X, 0.])
        ret = dict(list(zip(layer_names, layer_outs)))
        return ret



    ## 将当前模型保存至指定的json文件中    这个是程序原版的保存模型的程序
    def save_model_Ori23_7_12(self, filename):
        print("目前是nn.py文件中的 save_model(self, filename) 函数，来将模型信息保存到.json文件中！")
        # # 获取模型配置信息
        # config = self.model.get_config()
        # model_json = json.dumps(config)

        print("当前这个self.model的情况是怎样的！", type(self.model), self.model)
        model_json = self.model.to_json()
        # self.model.index = self.model.values.tolist()
        # model_json = self.model.to_json()

        # df = pd.DataFrame(self.model, index=False)
        # model_json = df.to_json()

        json_file_name = filename.replace('.h5', '.json')            ### 这个是将h5后缀修改为json后缀！
        with open(json_file_name, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(filename)          ### 这个就相当于是h5与json文件各保存了一份啊！




    ## 将当前模型保存至指定的json文件中      这个是修改后的函数！
    def save_model(self, filename):
        print("目前是nn.py文件中的 save_model(self, filename) 函数，来将模型信息保存到.json文件中！")
        # # 获取模型配置信息
        # config = self.model.get_config()
        # model_json = json.dumps(config)

        print("当前这个self.model的情况是怎样的！", type(self.model), self.model)
        # filename = filename.replace('.h5', '.hdf5')
        # self.model.index = self.model.index.tolist()
        # self.model.save(filename)
        self.model.save_weights(filename)




    ## 加载之前保存的模型
    def load_model(self, filename):
        ret = self.build_fn(**self.model_params)
        if type(ret) == tuple:
            self.model, self.feature_names = ret
        else:
            self.model = ret

        self.model.load_weights(filename)

        return self


    ## 将获取的各个特征的重要性进行保存
    def save_feature_importance(self, filename):

        coef = self.coef_
        if type(coef) != list:
            coef = [self.coef_]

        for i, c in enumerate(coef):
            df = pd.DataFrame(c)
            df.to_csv(filename + str(i) + '.csv')



    ## 获取指定特征的重要性
    def get_named_coef(self):

        if not hasattr(self, 'feature_names'):
            return self.coef_
        coef = self.coef_
        coef_dfs = {}
        print("测试部分，目前是在model/nn.py 文件当中，测试一下这个系数是list吗？！", type(coef), len(coef))
        if type(coef) == list:        ### 按理来说这个位置应该是一个字典类型的
            print("目前是model/nn.py文件，当前的这个coef是一个list的形式！")
            for i in range(len(coef)):
                print("上天护佑！！！当前coef内部的这一项是谁！", len(coef[i]))
            if len(coef) == 7:             ## 就是输入外加后面的六项
                coef1 = {}
                coef1['inputs'] = coef[0]
                coef1['h0'] = coef[1]
                coef1['h1'] = coef[2]
                coef1['h2'] = coef[3]
                coef1['h3'] = coef[4]
                coef1['h4'] = coef[5]
                coef1['h5'] = coef[6]
                coef = coef1
            else:
                print("目前是model/nn.py文件，  说明此时得到的这个数据不太对，按理来说应该是七层的！现在的这个数据是谁！", coef, len(coef))
        common_keys = set(coef.keys()).intersection(list(self.feature_names.keys()))            ### 现在是来获取coef和self.feature_names是两个字典对象的交集
        print("现在是在model/nn.py文件中，当前这个common_keys效果怎么样！", len(common_keys), common_keys)
        # print("目前的这个特征名字是谁！", self.feature_names)
        for k in common_keys:
            c = coef[k]
            print("上天保佑，当前的这一轮中这个c的情况", len(c), k, c)
            names = self.feature_names[k]
            print("当前的这个名字的长度是怎样的！", len(c.ravel()), len(names))
            df = pd.DataFrame(c.ravel(), index=names, columns=['coef'])
            coef_dfs[k] = df
        # print("目前是model/nn.py文件，目前最终得到的这个分数结果是谁！", coef_dfs)
        return coef_dfs







    def get_coef(self):
        return self.coef_

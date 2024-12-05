### 现在这个文件进行备份主要是想来修改一下他的输入的数据集的数目，来验证一下不同的数据的大小对最终精度的影响！
### 现在这个文件备份是因为想要修改五折交叉运算的！！！因为总会有最后几折效果过差！！


import datetime
import logging
from copy import deepcopy
from os import makedirs
from os.path import join, exists
from posixpath import abspath

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

import nn
from data.data_access import Data
from model.model_factory import get_model
from pipeline.one_split import OneSplitPipeline
from utils.plots import plot_box_plot
from utils.rnd import set_random_seeds
import copy
import random


## 下面这几个包的导入是因为相认计算一下各层网络他们内部的各个神经元节点的情况！

from model.model_utils import get_layers
from keras.layers import Dropout, BatchNormalization
from keras.models import Sequential
import keras
from keras import backend as K
from keras.callbacks import Callback
import os

from config_path import *
base_path = BASE_PATH                   ### 现在的这个base_path是D:\DailyCode\P-Net\CodeModel\(Father)pnet_prostate_paper-published_to_zenodo\pnet_prostate_paper-published_to_zenodo\
import csv

from tensorflow.keras.callbacks import LambdaCallback
import tensorflow as tf

import layers_custom
from sklearn.preprocessing import MinMaxScaler



timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())


def save_model(model, model_name, directory_name):
    # filename = join(abspath(directory_name), 'fs')
    filename = join(directory_name, 'fs')
    logging.info('saving model {} coef to dir ({})'.format(model_name, filename))
    if not exists(filename.strip()):
        makedirs(filename)
    filename = join(filename, model_name + '.h5')
    logging.info('FS dir ({})'.format(filename))
    print("最终模型所保存的文件地址是！", filename)
    model.save_model(filename)


def get_mean_variance(scores):
    df = pd.DataFrame(scores)
    return df, df.mean(), df.std()




### 下面这个函数是新一个求网络梯度的一个函数
class GradientModifier(Callback):

    def __init__(self, layer_name):
        super(GradientModifier, self).__init__()
        self.layer_name = layer_name

    def on_batch_end(self, batch, logs={}):
        layer = self.model.get_layer(self.layer_name)
        gradients = K.gradients(self.model.total_loss, layer.output)[0]
        modify_func = K.function([self.model.input], [gradients])
        layer_output, = modify_func([self.validation_data[0]])
        for i in range(layer_output.shape[-1]):
            if layer_output[0, i] > 0:  # 梯度为正，强化节点
                layer_output[:, i] = layer_output[:, i] * 2
            elif layer_output[0, i] < 0:  # 梯度为负，将输出置为0
                layer_output[:, i] = 0
            else:  # 梯度为0，不做处理
                continue
        K.set_value(layer.output, layer_output)



# ### 下面这个函数为自定义的一个激活函数，想要在此根据模型的梯度值来修改当前网络层的输出值，梯度值高的就强化，小的就弱化！
# def my_activation(x):
#     # x = K.tanh(x)        #### 现在这个x就是当前网络层的输出值！
#     # # 计算梯度值
#     # with tf.GradientTape() as tape:
#     #     tape.watch(inputs)
#     #     outputs = self.layer(inputs)
#     # grads = tape.gradient(outputs, inputs)
#
#
#     gradients = K.gradients(model.total_loss, layer_output)
#     # 处理梯度并与原输出相乘得到处理后的输出
#     gradients /= (K.sqrt(K.mean(K.square(gradients))) + K.epsilon())
#     grad_values_normalise = K.relu(gradients)
#     return K.tanh(x)





class CrossvalidationPipeline(OneSplitPipeline):
    def __init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params, exp_name):
        OneSplitPipeline.__init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params,
                                  exp_name)

    def run(self, n_splits=5):     ### 这个即为五折交叉验证    在实际的操作中我在这并没有按照五折来，一共是1011个样本数据，选用其中的909（90%）个来作为训练数据

        list_model_scores = []
        model_names = []

        for data_params in self.data_params:
            data_id = data_params['id']
            # logging
            logging.info('当前处于crossvalidation_pipeline.py文件中, loading data....')
            data = Data(**data_params)

            x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data.get_train_validate_test()     ### 获取训练所用的数据

            X = np.concatenate((x_train, x_validate_), axis=0)             ### 现在他是这个训练与这个验证集合在了一块，现在一共是有600，每一条数据中含有的元素的数目是27687
            y = np.concatenate((y_train, y_validate_), axis=0)
            info = np.concatenate((info_train, info_validate_), axis=0)


            ### 下面这个是23.7.25 尝试着进行来测试一下！尝试着把测试数据也给融入到X中，就是说现在传给无折交叉验证的X是一个完整的1011个样本 现在先别拆分训练集与验证集，等到五折交叉验证的时候，交给他来进行拆分！
            ### 原来不是这么操作的，没有下面这个代码的！
            X = np.concatenate((X, x_test_), axis=0)
            y = np.concatenate((y, y_test_), axis=0)
            info = np.concatenate((info, info_test_), axis=0)


            print("当前处于crossvalidation_pipeline.py文件中，测试输入数据部分，目前的输入数据中，X部分取值为：", len(X), len(X[0]))          ### 用以输入进P-Net模型的整合的数据       测试一下看看一共是有多少个样本，每个样本内部又是有多少个数据
            print("白文超，自领帝国大总理，开府仪同三司，Y部分的取值为：", len(y), len(y_test_))     ## 每个输入数据所应该对应的输出标签     （目前这批输入的训练数据是909个，每个数据里面又包含27687个数据）
            print("此时得到的训练和测试的数据集中阳性样本是有多少个！", y.sum(), y_test_.sum())

            print("问忠臣之安在！Info的信息是：", len(info))

            # get model
            logging.info('当前处于crossvalidation_pipeline.py文件中， 现在开始来拟合当前1的这个模型！  fitting model ...')

            for model_param in self.model_params:             ### 注意啊！！ 这里是一个循环，在这self.model_params 他是多个算法的参数合并到一起了！   比如说本实验的网络模型以及逻辑回归、SVM等对标算法都在这里面了！  现在通过model_param 来一个算法一个算法的取出对应算法的相应参数！
                if 'id' in model_param:
                    model_name = model_param['id']
                else:
                    model_name = model_param['type']

                set_random_seeds(random_seed=20080808)       ### 搞不懂这样随机设置的意义
                model_name = model_name + '_' + data_id              ### 这块就决定了最终各个算法的名字最后永远跟了一个'_ALL' 比如说 Logistic Regression_ALL
                m_param = deepcopy(model_param)    ### 此函数用以深度复制列表
                m_param['id'] = model_name
                logging.info('fitting model ...')

                scores = self.train_predict_crossvalidation(m_param, X, y, info, cols, model_name)     ## 现在开始进行训练，并得到最终的预测分数     ## 这就是整体的入口，训练模型并预测结果
                scores_df, scores_mean, scores_std = get_mean_variance(scores)
                list_model_scores.append(scores_df)
                model_names.append(model_name)
                self.save_score(data_params, m_param, scores_df, scores_mean, scores_std, model_name)
                logging.info('scores')
                logging.info(scores_df)
                logging.info('mean')
                logging.info(scores_mean)
                logging.info('std')
                logging.info(scores_std)

        df = pd.concat(list_model_scores, axis=1, keys=model_names)
        df.to_csv(join(self.directory, 'folds.csv'))
        plot_box_plot(df, self.directory)

        return scores_df

    def save_prediction(self, info, y_pred, y_pred_score, y_test, fold_num, model_name, training=False):
        if training:
            file_name = join(self.directory, model_name + '_traing_fold_' + str(fold_num) + '.csv')
        else:
            file_name = join(self.directory, model_name + '_testing_fold_' + str(fold_num) + '.csv')
        logging.info("目前是在crossvalidation_pipeline.py文件中，现在是来保存模型的预测结果  saving : %s" % file_name)
        info['pred'] = y_pred
        info['pred_score'] = y_pred_score
        info['y'] = y_test         ### 将最终各个样本的预测分数进行保存！
        info.to_csv(file_name)


    ###  这个是向输入数据中来添加噪音，进而来测试这个数据的鲁棒性！
    def TestDataAddNoiseTest5_19(self, TestData):
        for i in range(len(TestData)):
            # for j in range(len(TestData[0])):
            for j in range(int(1 * len(TestData[0]))):
                TestData[i][j] = TestData[i][j] + np.random.uniform(-0.04, 0.04)
        print("进行噪音干扰之后的这个数据情况是怎样的！", TestData)
        return TestData



    def TestDataAddNoiseTest10_19(self, TestData):
        import random
        BiLi = 0.08
        NoisyNum = int(BiLi * len(TestData[0]))
        numbers = random.sample(range(0, len(TestData[0])), NoisyNum)            ### 选出具体那几列来进行噪音添加！
        for i in range(len(TestData)):
            for j in numbers:             #### 现在这个j就是具体的那一列的标题
                if TestData[i][j] == 0:
                    TestData[i][j] = 1
                elif TestData[i][j] == 1:
                    TestData[i][j] = 0
                elif TestData[i][j] == -1:
                    TestData[i][j] = random.choice([1, 0])
                else:
                    print("此时是鲁棒性加噪音的函数，为TestDataAddNoiseTest10_19，现在出错了！！警告信息")

        return TestData




    def XiaoRongProcess(self, SuanFaName, nowFold, x_test, y_test, model, NegFlag=None):
        BiLi_All = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  ### 这个就是各轮的笑容比例！
        BiLi_All = [0.05, 0.1, 0.15, 0.2, 0.25]  ### 这个就是各轮的笑容比例！
        All_score_test = {}  ### 这个来保存最终各轮运算中每一轮的消融测试的结果！
        for nowBiLi in BiLi_All:  ### 逐比例的来进行消融！
            ### 注意哈！ 下面这一块都是来进行复制当前的网络模型的！ 因为当前的这个Model 并不是直接的keras 这个copy.copy()
            new_model = copy.copy(model)  ## copy.deepcopy(original_model)         keras.models.clone_model(model.model)
            temp_model = keras.models.clone_model(model.model)
            temp_model.set_weights(model.model.get_weights())
            new_model.model = temp_model
            new_model.XiaoRongFlag = True
            ### 下面来确定一下该怎么进行消融，是正着消还是反着消！
            if NegFlag is None:
                XiaoRongDirect = "Pos"          ### 默认情况下，他在消融的时候都是正着消
            else:
                XiaoRongDirect = "Neg"         ### 这种都是要在笑容时反着消！
            print("目前赋值完成的这个神经网络模型是怎样的！", new_model.XiaoRongFlag, model.XiaoRongFlag)
            XiaoRongDataPath = '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-' + SuanFaName + '/'          ### 指定当前所依赖的那些消融数据存放在哪个位置
            XiaoRongParams = {'BiLi': nowBiLi, 'Fold': nowFold, 'Path': XiaoRongDataPath, 'Direct': XiaoRongDirect}  ### 现在来构造属于当前这一折的消融参数，主要是当前要消融多少个元素参数（就是比例），以及当前是第几折（主要是来找对应的可解释结果的保存文件！）
            # XiaoRongParams = {'BiLi': nowBiLi, 'Fold': 2}  ### 现在来构造属于当前这一折的消融参数，主要是当前要消融多少个元素参数（就是比例），以及当前是第几折（主要是来找对应的可解释结果的保存文件！）
            y_pred_test, y_pred_test_scores = self.predict(new_model, x_test, y_test,
                                                           XiaoRongParams)  ### 训练完成后，拿测试数据来验证一下，获得测试数据的预测得分     PS!!千万注意！在这的这个self.predict()是指来调用当前CrossvalidationPipeline类内部的predict()函数，而当前的CrossvalidationPipeline类继承自OneSplitPipeline这个类
            ### 而OneSplitPipeline这个类在 pipeline.one_split.py 这个文件中。    而且对于predict()这个函数，当前CrossvalidationPipeline类并没有对他进行重写  因此他是直接调的pipeline.one_split.py 这个文件中  OneSplitPipeline这个类的predict()这个函数
            ### 当前处于crossvalidation_pipeline.py文件中，现在已经获取关于训练数据的预测评分了！下面开始对这个得分进行1评估
            score_test = self.evaluate(y_test, y_pred_test, y_pred_test_scores)
            print("上天啊！！！目前这个是进行消融操作时，单论测试时这个测试分数是谁！", score_test)
            All_score_test[str(nowBiLi)] = score_test
        print("那么最终得到的各折消融结果分数是怎样的：", All_score_test)

        ### 现在对这几折消融后的结果进行保存，保存为CSV文件！
        ### 现在，正向消融与反向消融这两种策略，是需要保存在不同的文件夹中！
        if NegFlag is None:  ### 此时意味着应该进行正向消融！
            XiaoRongResult_Path = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs/' + SuanFaName + '/'  ### 不同的消融算法在不同的文件夹下！
        else:  ### 此时意味着应该反向消！
            XiaoRongResult_Path = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs/' + SuanFaName + '-Neg/'  ### 不同的消融算法在不同的文件夹下！
        if not os.path.exists(XiaoRongResult_Path):
            os.mkdir(XiaoRongResult_Path)
        ### 对各折消融之后的结果分精度指标来进行保存一下！
        accuracyFile, aucFile, f1File, auprFile = XiaoRongResult_Path + 'accuracy.csv', XiaoRongResult_Path + 'auc.csv', XiaoRongResult_Path + 'f1.csv', XiaoRongResult_Path + 'aupr.csv'
        precisionFile, recallFile = XiaoRongResult_Path + 'precision.csv', XiaoRongResult_Path + 'recall.csv'
        zhibiaoFiles = [accuracyFile, precisionFile, aucFile, f1File, auprFile, recallFile]
        ZhiBiaos = ['accuracy', 'precision', 'auc', 'f1', 'aupr', 'recall']
        for nowFlag in range(len(ZhiBiaos)):
            nowZhibiao, nowfile = ZhiBiaos[nowFlag], zhibiaoFiles[nowFlag]
            # 1. 创建文件对象
            if nowFold == 0:  ### 在第一折的时候直接新创建一个数据文件   以写的方式('w')来进行读取这个文件的，原来文件中的那些数据全都给覆盖掉！
                f = open(nowfile, 'w', newline='',
                         encoding='utf-8')  ### 这个就是最终要写入的这个文件      在这，它是以写的方式('w')来进行读取这个文件的，这种写法会将文件中原来的数据给覆盖掉的！
            else:
                f = open(nowfile, 'a+', newline='',
                         encoding='utf-8')  ### 在这，这个是以追加(a+)的方式来读取这个数据文件的！，后面那几折数据必须是在第一折的基础之上再往后添加数据！
            # 2. 基于文件对象构建 csv写入对象
            csv_writer = csv.writer(f)
            # 3. 构建列表头     注！！只有当前的fold=0，处于第一折的时候才会进行写入操作
            if nowFold == 0:  ### 只有在第一折的时候才会进行写入表头的操作！
                RowHeader = ["Fold"] + BiLi_All
                csv_writer.writerow(RowHeader)  ## 将这个标题写到对应的文件中！
            # 3. 写入数据  将当前这一指标的这一折数据的不同比例的消融结果进行写入
            NowValue = [nowFold]  ## 这里面存放不同消融比例的结果！
            for key in BiLi_All:
                NowScore = All_score_test[str(key)][nowZhibiao]  ### 具体某个消融比例下的各个精度结果分数
                NowValue.append(NowScore)
            csv_writer.writerow(NowValue)  ## 将当前这一行写入！




    def predict(self, model, x_test, y_test, XiaoRongParams=None):
        logging.info('现在这个是one_split.py文件,  predicitng ...')
        # y_pred_test = model.predict(x_test, XiaoRongFlag)
        if XiaoRongParams is None:
            y_pred_test = model.predict(x_test)
        else:
            y_pred_test = model.predict(x_test, XiaoRongParams)

        if hasattr(model, 'predict_proba'):
            y_pred_test_scores = model.predict_proba(x_test)[:, 1]
        else:
            y_pred_test_scores = y_pred_test

        print('现在这个是one_split.py文件, y_pred_test', y_pred_test.shape, y_pred_test_scores.shape)
        return y_pred_test, y_pred_test_scores


    ### 下面这个函数就是将最终激活网络的输出值在分层的将各折运算结果求一下平均值！
    def getAvgActNodel(self):
        # 读取csv文件
        filePath = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/AllActModelResult.csv'  ### 指定各轮的精度结果所要保存的地方
        data = pd.read_csv(filePath)

        # 判断第一列数据是否为float类型的函数
        def is_float(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        # 按第一列进行分组，并计算每组的平均值
        grouped_data = data[data[data.columns[0]].apply(is_float)].groupby(data.columns[0]).mean()
        # 添加新行到原始数据的末尾
        new_rows = []
        for index, row in grouped_data.iterrows():
            new_row = ['Avg' + str(index)] + list(row)
            new_rows.append(new_row)

        new_data = data.append(pd.DataFrame(new_rows, columns=data.columns))
        # 追加空行
        new_data = new_data.append(pd.Series(), ignore_index=True)
        # 将修改后的数据保存回原文件
        new_data.to_csv(filePath, index=False)


    def train_predict_crossvalidation(self, model_params, X, y, info, cols, model_name):
        logging.info('当前处于crossvalidation_pipeline.py文件中， 目前的模型参数情况为：model_params: {}'.format(model_params))
        n_splits = self.pipeline_params['params']['n_splits']
        skf = StratifiedKFold(n_splits=n_splits, random_state=123, shuffle=True)       ## 进行划分进行交叉验证！    此时数据已经打乱了
        i = 0
        scores = []
        model_list = []

        ### 下面开始尝试一下不同实验数据集的规模下的各个对标算法的情况！
        OriLen = len(X)           ### 这个是原来的数据长度
        ShiYanDiffDatasetScale = False
        if ShiYanDiffDatasetScale:
            BiLi = 0.01
            X, y, info = X[0:int(BiLi*len(X))], y[0:int(BiLi*len(y))], info[0:int(BiLi*len(info))]
            print("现在已经是对这个数据进行了过滤选取，现在精简后的数据是多少：", len(X)/OriLen)



        KuoZengFlag = True                ### 这个表示对当前的这个数据集要不要进行扩增！
        if model_params['type'] == 'nn' and KuoZengFlag:
            ### 下方这个为2023.8.1修改代码，因为输入数据的正负样本极不均衡，接近1:2，因此之前的代码手段是从这些样本数据中随机的抽取数据，这样很有可能在其中一折随机抽取的数据中阳性样本非常多，阴性样本非常少，使得总体更不均衡！
            ### 因此针对上述这个问题，即为阳性样本与阴性样本分开来抽，保证每一折的训练数据中，阴性与阳性样本的比例均与总体样本保持一致！
            ### 现在的这个训练数据样本分布不均衡，得想办法在训练的时候使其均衡！
            print("现在传过来的这批训练数据的情况是怎样的！", type(X), type(y), len(X), len(y))
            ### 现在就是想办法将这个训练数据给打乱！
            dataIndex = []
            for j in range(len(X)):
                dataIndex.append(j)
            # np.random.shuffle(dataIndex)
            NewNum = int(0.20*len(dataIndex))             ### 现在是需要补充多少数据！
            NowIndex = dataIndex[0:NewNum]
            NewX, Newy, Newinfo = X[NowIndex], y[NowIndex], info[NowIndex]
            ### 现在要往数据集中补充哪些阳性数据已经知道了，现在开始重新构造这个数据集！
            X, y, info = np.concatenate([X, NewX], axis=0), np.concatenate([y, Newy], axis=0), np.concatenate([info, Newinfo], axis=0)


        BalanceFlag = False
        if BalanceFlag:      ### 此时开始调整这批数据的类别平衡情况
            print("24.3.13现在这里数据集的正负情况为：", np.sum(y), len(y)-np.sum(y))
            ### 现在开始从两个类别中各自挑选一定数目的样本数目
            SetTotalNum = 400
            # PosNum, NegNum = 320, 80        ### 现在指定正负样本的占比！  现在这个表示正负样本的比例为4:1
            # PosNum, NegNum = 300, 100  ### 现在指定正负样本的占比！  现在这个表示正负样本的比例为3:1
            # PosNum, NegNum = 200, 200  ### 现在指定正负样本的占比！  现在这个表示正负样本的比例为1:1
            # PosNum, NegNum = 100, 300  ### 现在指定正负样本的占比！  现在这个表示正负样本的比例为1:3
            PosNum, NegNum = 80, 320  ### 现在指定正负样本的占比！  现在这个表示正负样本的比例为1:4
            PosList = []
            NegList = []
            for nowi in range(len(X)):
                if y[nowi] > 0 and PosNum > 0:     ### 此时是正样本了!而且这个正样本还没有选够
                    PosList.append(nowi)
                    PosNum = PosNum-1
                if y[nowi] == 0 and NegNum > 0:     ### 此时是正样本了!而且这个正样本还没有选够
                    NegList.append(nowi)
                    NegNum = NegNum-1
                if PosNum<=0 and NegNum <=0:
                    break
            SelectData = PosList + NegList
            random.shuffle(SelectData)
            print("现在选择之后的数据是！", len(SelectData), SelectData)
            X, y, info = X[SelectData], y[SelectData], info[SelectData]



        ### 下面开始进行五折交叉运算！
        for train_index, test_index in skf.split(X, y.ravel()):
            print("现在是测试环节！当前处于crossvalidation_pipeline.py文件中，现在所求出来的索引和原始数据的情况是怎样的！", type(train_index), len(train_index), len(test_index))

            ### 在这，这个就相当于是 每次从中截取固定长度的一批数据用作训练数据进行训练模型！
            model = get_model(model_params)         ## 其实就是根据模型参数文件中的那些模型的超参数来初始化设置这个模型！
            logging.info('fold # ----------------%d---------' % i)
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            info_train = pd.DataFrame(index=info[train_index])
            info_test = pd.DataFrame(index=info[test_index])
            x_train, x_test = self.preprocess(x_train, x_test)
            # feature extraction
            logging.info('当前处于crossvalidation_pipeline.py文件中，现在进行特征提取！  feature extraction....')
            x_train, x_test = self.extract_features(x_train, x_test)
            print("当前是在pipeline/crossvalidation_pipeline.py文件中，现在在五折交叉验证中所用的数据的长度是怎样的！", len(x_train), len(x_test))   ### 因为是五折交叉验证，因此在原来分好的训练集上在此分成五份，前四分用来训练后一份用来测试
            print("上天保佑！！现在这个测试样本以及info的详细信息是谁！", x_test, info)

            FCN_x_train, FCN_y_train, FCN_x_test, FCN_y_test = x_train, y_train, x_test, y_test

            ### 下面可以尝试一下修改他的测试！



            ### 可以在这构造一个简单的神经网络测试一下
            FCNCeshiFlag, GeneCeshiFlag = False, False
            if FCNCeshiFlag:
                input_dim = len(FCN_x_train[0])  ### 现在就代表输入的这个样本数据他的对应的维度，其实就是这一层中神经元的数目！
                # 创建模型
                from keras.models import Sequential
                from keras.layers import Dense
                FCNmodel = Sequential()
                FCNmodel.add(Dense(10, activation='relu', input_shape=(input_dim,)))
                FCNmodel.add(Dense(1, activation='sigmoid'))
                # 编译模型
                FCNmodel.compile(optimizer='adam', loss='binary_crossentropy')

                # 训练模型
                FCNmodel.fit(FCN_x_train, FCN_y_train, epochs=100, batch_size=32)
                # 进行模型测试
                # valPredict = model.predict(valX)  ### 这个是对验证数据的预测结果！！
                # # threshold = self.get_th(valY, valPredict)       ### 根据验证数据来获取最佳阈值！    他的输入参数有两个，一个是验证数据标签，一个是输入参数！
                threshold = 0.5
                predictions = FCNmodel.predict(FCN_x_test)  ### 现在根据测试样本来进行预测获取对应的预测结果！
                binary_predictions = np.where(predictions > threshold, 1, 0)  ### 根据获得的阈值来对预测结果进行二值化处理！
                # 计算Accuracy、Precision、Recall、AUPRC、AUC和F1值
                from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score, \
                    roc_auc_score, f1_score
                accuracy = accuracy_score(FCN_y_test, binary_predictions)
                precision = precision_score(FCN_y_test, binary_predictions)
                recall = recall_score(FCN_y_test, binary_predictions)
                auprc = average_precision_score(FCN_y_test, predictions)
                auc = roc_auc_score(FCN_y_test, predictions)
                f1 = f1_score(FCN_y_test, binary_predictions)
                # test_loss, test_accuracy = model.evaluate(testX, testY, verbose=2)
                # print("测试集上的损失值：", test_loss)
                print("现在是来测试一下拿原始的输入数据作为简单的全连接模型的输入，最终当前的这个测试集上的预测精度：", len(FCN_x_train), input_dim, accuracy, precision, recall, auprc,
                      auc, f1)


            if GeneCeshiFlag:
                input_dim = len(FCN_x_train[0])  ### 现在就代表输入的这个样本数据他的对应的维度，其实就是这一层中神经元的数目！
                # 创建模型
                from keras.models import Sequential
                from keras.layers import Dense
                from model.layers_custom import Diagonal, SparseTF
                from keras.regularizers import l2
                from data.pathways.reactome import ReactomeNetwork
                import itertools
                from data.data_access import Data

                selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'
                data_params = {'id': 'ALL', 'type': 'prostate_paper',
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
                data = Data(**data_params)
                x, y, info, cols = data.get_data()  ### 再次获取训练所用的数据情况！
                if hasattr(cols, 'levels'):
                    genes = cols.levels[0]
                else:
                    genes = cols
                ## 获取当前层与下一层之间的关联关系
                def get_map_from_layer(layer_dict):
                    pathways = list(layer_dict.keys())
                    print("目前所处的文件为：builders_utils.py")
                    print('目前从层字典中读取的通路的数目，pathways', len(pathways))
                    genes = list(itertools.chain.from_iterable(
                        list(layer_dict.values())))  ## 在这itertools.chain.from_iterable表示是将多个迭代器进行高效的连接，连接成一个迭代器！
                    genes = list(np.unique(genes))
                    print('目前从层字典中读取的基因的数目，genes', len(genes))
                    ### 在这里，各个网络层中的基因与通路并不一定真的就代表给基因和通路，此二者只是表示当前层的上下级关系，基因下一级，通路上一级！

                    n_pathways = len(pathways)
                    n_genes = len(genes)

                    mat = np.zeros((n_pathways,
                                    n_genes))  ### 在这，纵坐标就代表通路（就是下一层的大通路），横坐标就代表基因（就是当前层的小通路）   现在就是来构造上一层和下一层之间的关联关系矩阵图 mat   最开始全都初始化为0，如果彼此之间有关系的就设置为1
                    for p, gs in list(layer_dict.items()):
                        g_inds = [genes.index(g) for g in gs]  ### 就是挨个的遍历来找一下，当前的这个通路中是否包含这个基因，有的话，那么对应的位置就是1
                        p_ind = pathways.index(p)
                        mat[p_ind, g_inds] = 1  ## 就是说如果该位置的基因和通路彼此之间确实存在一定的关系，那么这个关系图中的这个位置就设置为1

                    df = pd.DataFrame(mat, index=pathways, columns=genes)
                    # for k, v in layer_dict.items():
                    #     print k, v
                    #     df.loc[k,v] = 1
                    # df= df.fillna(0)
                    return df.T
                ## 获取网络中每一层他的前后连接关系，并将各个连接关系进行汇总！
                def get_layer_maps(genes, n_levels, direction, add_unk_genes):
                    reactome_layers = ReactomeNetwork().get_layers(n_levels,
                                                                   direction)  ## 现在的这个reactome_layers就是一个字典，对应着上一层与他对应的下一层元素之间的关系
                    filtering_index = genes
                    # print("目前是在 builders/builders_utils.py 文件中，当前的这个genes是谁！", genes)
                    # print("现在的这个网络层是怎样的！", reactome_layers)
                    # print("目前是在 builders/builders_utils.py 文件中，现在将这个genes详细展开会是怎样的！！", list(genes))
                    maps = []
                    for i, layer in enumerate(reactome_layers[::-1]):  ## 这里面的每一层都蕴含上下级关系！
                        print('layer #', i)
                        # print("测一下，目前所读取的这个layer是谁！", layer)
                        mapp = get_map_from_layer(layer)  ## 获取当前层他的上下级关系（0、1矩阵图）        纵轴代表小通路，横轴代表大通路
                        filter_df = pd.DataFrame(index=filtering_index)
                        print('目前所处的文件为：builders_utils.py  ， filtered_map', filter_df.shape)
                        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True,
                                                       how='left')  ## 将两个DataFrame进行合并   就是将filter_df与mapp进行合并
                        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
                        print('filtered_map', filter_df.shape)
                        # print("当前是在 builders/builders_utils.py 文件中，现在，他的这个mapp图是怎样的！", mapp)
                        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')

                        # UNK, add a node for genes without known reactome annotation    UNK，为没有已知reactome注释的基因添加一个节点
                        if add_unk_genes:
                            print('UNK ')
                            filtered_map['UNK'] = 0
                            ind = filtered_map.sum(axis=1) == 0  ### 二维数组中他的水平行上的数据进行求和
                            filtered_map.loc[ind, 'UNK'] = 1
                        ####

                        filtered_map = filtered_map.fillna(0)
                        print('目前所在的文件是：builders_utils.py  filtered_map', filter_df.shape)
                        # filtering_index = list(filtered_map.columns)
                        filtering_index = filtered_map.columns
                        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
                        # print("当前所在的文件是pnet_prostate_paper-published_to_zenodo/model/builders/builders_utils.py，当前具体某一层映射他的关系具体情况为！", filtered_map)      ### 当前所构造的这个filtered_map，他的纵轴的标题为上一层的元素（基因），他的横轴标题为下一层的元素（通路）
                        # print("当前的这个map，他的列情况是怎样的！", len(filtered_map.columns), filtered_map.columns)

                        # print("当前是在pnet_prostate_paper-published_to_zenodo/model/builders/builders_utils.py文件中，当前的这个关系表的具体数据内容是怎样的！", filtered_map.value)
                        # print("当前是在pnet_prostate_paper-published_to_zenodo/model/builders/builders_utils.py文件中，读取一下这个关系表的某一行数据！", np.sum(filtered_map.iloc[1]), filtered_map.iloc[1])
                        # print("当前的这个map，他的行情况是怎样的！", len(filtered_map.rows), filtered_map.rows)
                        maps.append(filtered_map)
                    return maps

                n_hidden_layers, direction, add_unk_genes = 5, 'root_to_leaf', True
                maps = get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes)
                mapp = maps[0]

                OriMapp = mapp  ### 保留一下原始的映射关系图！
                mapp = mapp.values
                n_genes, n_pathways = mapp.shape

                FCNmodel = Sequential()
                layer1 = Diagonal(9229, input_shape=(input_dim,), activation='tanh', W_regularizer=l2(0.001),
                                  use_bias=True, name='h0', kernel_initializer='lecun_uniform')
                hidden_layer1 = SparseTF(1387, mapp, activation='tanh', W_regularizer=l2(0.001),
                                        name='h1', kernel_initializer='lecun_uniform',
                                        use_bias=True)       #### 构建其中的一个隐藏层！   在这是需要构建一个稀疏的网络层

                FCNmodel.add(layer1)
                FCNmodel.add(hidden_layer1)
                FCNmodel.add(Dense(1, activation='sigmoid'))
                # 编译模型
                FCNmodel.compile(optimizer='adam', loss='binary_crossentropy')

                # 训练模型
                FCNmodel.fit(FCN_x_train, FCN_y_train, epochs=80, batch_size=32)
                # 进行模型测试
                # valPredict = model.predict(valX)  ### 这个是对验证数据的预测结果！！
                # # threshold = self.get_th(valY, valPredict)       ### 根据验证数据来获取最佳阈值！    他的输入参数有两个，一个是验证数据标签，一个是输入参数！
                threshold = 0.5
                predictions = FCNmodel.predict(FCN_x_test)  ### 现在根据测试样本来进行预测获取对应的预测结果！
                binary_predictions = np.where(predictions > threshold, 1, 0)  ### 根据获得的阈值来对预测结果进行二值化处理！
                # 计算Accuracy、Precision、Recall、AUPRC、AUC和F1值
                from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score, \
                    roc_auc_score, f1_score
                accuracy = accuracy_score(FCN_y_test, binary_predictions)
                precision = precision_score(FCN_y_test, binary_predictions)
                recall = recall_score(FCN_y_test, binary_predictions)
                auprc = average_precision_score(FCN_y_test, predictions)
                auc = roc_auc_score(FCN_y_test, predictions)
                f1 = f1_score(FCN_y_test, binary_predictions)
                # test_loss, test_accuracy = model.evaluate(testX, testY, verbose=2)
                # print("测试集上的损失值：", test_loss)
                print("现在是来测试一下拿原始的输入数据作为简单的全连接模型的输入，最终当前的这个测试集上的预测精度：", len(FCN_x_train), input_dim, accuracy, precision, recall, auprc,
                      auc, f1)




            # print("现在这个训练之前这个列标题是谁！", type(cols), cols)
            ### 下面是测试部分，来将这个列标题保存到指定的位置！
            newcols = cols.to_frame()
            newcols.to_csv("../CeShi/OtherTest/Other/GeneColsNames.csv")
            print("现在这个输入数据的列标题（27687个）数据文件保存好了！！")


            if model_params['type'] == 'nn':
                model = model.fit(x_train, y_train, None, None, i)       ### 提取好特征之后，现在开始进行拟合模型进行训练           最后一个参数为fold表示当前处于第几折当中，主要是为了获得当前折可解释结果后保存文件所用！
            else:
                model = model.fit(x_train, y_train)

            print("现在是在pipeline/crossvalidation_pipeline.py文件中，这个测试数据的样子是什么样的！", len(x_test), len(x_test[0]), type(x_test), type(x_test[0]), x_test[0], x_test)
            # x_test = self.TestDataAddNoiseTest5_19(x_test)          ### 现在是来进行测试的部分，对这个测试数据来进行噪音的干扰
            AddNoisyFlag = False
            if AddNoisyFlag:         ### 由这个标志来决定是否添加噪音！
                x_test = self.TestDataAddNoiseTest10_19(x_test)  ### 现在是来进行测试的部分，对这个测试数据来进行噪音的干扰



            ### 现在这玩意是来根据根据不同的测试数据来获取每个神经元的相似性！     这块其实就是AAAI论文的那个算法的复现手段！
            AAAI = False                    ### 下面这块决定是否要使用AAAI的算法来获取各层网络中各个元素的重要性分数，而且再将它们进行保存！
            if model_params['type'] == 'nn' and AAAI:                  ### 在这，必须专门指明一下 model_params['type'] == 'nn'  因为只有当前的这个模型才会执行这个AAAI的操作！！其他的对标均不会执行！
                rankings = model.get_CorrAll(x_test)                           ### 现在获取得到的这个rankings是一个字典，key是各层网络的名字，value又是一个字典，他的key是这层网络的具体的元素的名字，value则是这个元素对应的方差值！
                print("上天保佑啊！！现在这个神经元的排名情况是怎样的！", rankings)
                ### 现在就要想办法将目前所读取的这个AAAI进行排序得到的元素的排名给保存一下！
                layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']           ### 分别读取各层来进行处理！
                path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-AAAI'        ## _logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs         下面这块来指定各轮中各层各个元素他们的AAAI方法得到的排名结果中的元素情况
                if not os.path.exists(path):
                    os.mkdir(path)
                for realName in layerNames:
                    nowPath = path + '/coef_nn_fold_' + str(i) + '_layer' + realName + '.csv'
                    nowData = rankings[realName]       ### 获取当前具体某一层他的各个元素所对应的AAAI可解释结果！
                    mid = pd.DataFrame(list(nowData.items()))
                    mid.to_csv(nowPath, header=['element', 'coef'], index=False)


            ### 现在这玩意是来根据根据不同的测试数据来获取每个神经元的相似性！     这块其实就是AAAI论文的那个算法的复现手段！
            Active = False                    ### 下面这块决定是否要使用AAAI的算法来获取各层网络中各个元素的重要性分数，而且再将它们进行保存！
            if model_params['type'] == 'nn' and Active:                  ### 在这，必须专门指明一下 model_params['type'] == 'nn'  因为只有当前的这个模型才会执行这个AAAI的操作！！其他的对标均不会执行！
                rankings = model.get_ActivationsAll(X)          ### 本来这个参数位置是 x_test                   ### 现在获取得到的这个rankings是一个字典，key是各层网络的名字，value又是一个字典，他的key是这层网络的具体的元素的名字，value则是这个元素对应的方差值！
                print("上天保佑啊！！目前是按照神经元激活频率来进行排序！现在这个神经元的排名情况是怎样的！", rankings)
                ### 现在就要想办法将目前所读取的这个AAAI进行排序得到的元素的排名给保存一下！
                layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']           ### 分别读取各层来进行处理！
                path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-Active'        ## _logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs         下面这块来指定各轮中各层各个元素他们的AAAI方法得到的排名结果中的元素情况
                if not os.path.exists(path):
                    os.mkdir(path)
                for realName in layerNames:
                    nowPath = path + '/coef_nn_fold_' + str(i) + '_layer' + realName + '.csv'
                    nowData = rankings[realName]       ### 获取当前具体某一层他的各个元素所对应的AAAI可解释结果！
                    mid = pd.DataFrame(list(nowData.items()))
                    mid.to_csv(nowPath, header=['element', 'coef'], index=False)




            ### 现在这玩意是来根据根据不同的测试数据来获取每个神经元的相似性！     这块其实就是AAAI论文的那个算法的复现手段！
            Logict_Active = True                    ### 下面这块决定是否要使用AAAI的算法来获取各层网络中各个元素的重要性分数，而且再将它们进行保存！
            if model_params['type'] == 'nn' and Logict_Active:                  ### 在这，必须专门指明一下 model_params['type'] == 'nn'  因为只有当前的这个模型才会执行这个AAAI的操作！！其他的对标均不会执行！
                logictActParams = {'Y': y}            ### 这个是来获取拟合逻辑回归所用的y标签参数！
                rankings = model.get_ActivationsAll(X, logictActParams)                           ### 现在获取得到的这个rankings是一个字典，key是各层网络的名字，value又是一个字典，他的key是这层网络的具体的元素的名字，value则是这个元素对应的方差值！
                print("上天保佑啊！！目前是按照神经元激活频率来进行排序！现在这个神经元的排名情况是怎样的！", rankings)
                ### 现在就要想办法将目前所读取的这个AAAI进行排序得到的元素的排名给保存一下！
                layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']           ### 分别读取各层来进行处理！
                path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-Logict-Active-Kno'  ## _logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs         下面这块来指定各轮中各层各个元素他们的AAAI方法得到的排名结果中的元素情况
                path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-Logict-Active'
                if not os.path.exists(path):
                    os.mkdir(path)
                for realName in layerNames:
                    nowPath = path + '/coef_nn_fold_' + str(i) + '_layer' + realName + '.csv'
                    nowData = rankings[realName]       ### 获取当前具体某一层他的各个元素所对应的AAAI可解释结果！
                    mid = pd.DataFrame(list(nowData.items()))
                    mid.to_csv(nowPath, header=['element', 'coef'], index=False)





            ### 下面尝试一下将SHAP的可解释结果与这个激活结果进行组合一下，看看最终的效果是怎样的！    而且这玩意最终
            SHAP_Active = False                    ### 下面这块决定是否要使用AAAI的算法来获取各层网络中各个元素的重要性分数，而且再将它们进行保存！
            if model_params['type'] == 'nn' and SHAP_Active:                  ### 在这，必须专门指明一下 model_params['type'] == 'nn'  因为只有当前的这个模型才会执行这个AAAI的操作！！其他的对标均不会执行！
                rankings = model.get_ActivationsAll(x_test)                           ### 现在获取得到的这个rankings是一个字典，key是各层网络的名字，value又是一个字典，他的key是这层网络的具体的元素的名字，value则是这个元素对应的方差值！
                print("上天保佑啊！！目前是按照神经元激活频率来进行排序！现在这个神经元的排名情况是怎样的！", rankings)
                ### 接下来就要分别对SHAP的结果以及当前的这个激活算法的结果先进行一下归一化都归一化到0~1之间的范围之后再进行操作
                def normalize_dict_values(input_dict):
                    # 提取原字典的值构成一个二维数组
                    X = [[value] for value in input_dict.values()]

                    # 使用MinMaxScaler对值进行归一化处理
                    scaler = MinMaxScaler()                 ### 现在这个 feature_range=(-1, 1)参数决定了最终的归一化范围，不写的话就是（0,1）
                    normalized_values = scaler.fit_transform(X).flatten()

                    # 构建归一化后的字典，键值对关系与原字典保持一致
                    normalized_dict = {}
                    for index, (key, value) in enumerate(input_dict.items()):
                        normalized_dict[key] = normalized_values[index]
                    return normalized_dict
                ### 下面这块就对激活方法获取的各个字典进行归一化处理！
                for active_key in rankings:
                    rankings[active_key] = normalize_dict_values(rankings[active_key])        ### 现在对其中的各个字典来进行归一化处理


                ### 下面来获取SHAP文件夹中的数据
                SHAP_path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-SHAP'
                ### 下面这个文件是具体的来读取SHAP文件中的数据并将其构造为一个字典形式
                def read_csv_to_dict(filename, key_col, value_col):
                    data_dict = {}
                    with open(filename, 'r', newline='') as file:
                        reader = csv.reader(file)
                        for row in reader:
                            key = row[key_col]
                            value = row[value_col]
                            data_dict[key] = value
                    return data_dict
                layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']
                key_col = 0  # key所在列的索引，假设第一列为key
                value_col = 1  # value所在列的索引，假设第二列为value
                SHAP_Active_rankings = {}
                for shapLayerName in layerNames:
                    nowSHAP_path = SHAP_path + '/coef_nn_fold_' + str(i) + '_layer' + shapLayerName + '.csv'
                    data_dict = read_csv_to_dict(nowSHAP_path, key_col, value_col)

                    del data_dict['element']            ### 首行元素给删掉！
                    # ### 因为现在这个SHAP文件有些特殊，没有element那个标题，因此要想办法根据他的value值来进行删除！
                    # def remove_items_with_value(d, value):
                    #     return {key: val for key, val in d.items() if val != value}
                    # data_dict = remove_items_with_value(data_dict, 'coef')  ### 按照value值来进行删除！

                    data_dict = normalize_dict_values(data_dict)
                    ### 此时得到的data_dict 这个字典就是经过归一化的当前这一层中SHAP的结果值
                    Act_dict = rankings[shapLayerName]
                    result_dict = {}   ### 这个就是最终的融合后的结果
                    for key in data_dict:
                        if key in Act_dict:
                            value1 = data_dict[key]
                            value2 = Act_dict[key]
                            result_dict[key] = 0.8*value1 + 0.3*value2

                    ### 别忘了，此时还有最关键的一步！一定要记得排序！
                    def sort_dict_by_value(d):       ### 现在就是按照value取值从大到校进行排序！  同时还要保证排序后的数据还是一个字典的形式，方便后续的操作！
                        sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
                        return {k: v for k, v in sorted_items}
                    result_dict = sort_dict_by_value(result_dict)             ### 现在就是按照value取值从大到校进行排序！
                    SHAP_Active_rankings[shapLayerName] = result_dict




                ### 现在就要想办法将目前所读取的这个AAAI进行排序得到的元素的排名给保存一下！
                layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']           ### 分别读取各层来进行处理！
                path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-Active'        ## _logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs         下面这块来指定各轮中各层各个元素他们的AAAI方法得到的排名结果中的元素情况
                SHAP_Active_path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-SHAP-Active'
                if not os.path.exists(path):
                    os.mkdir(path)
                if not os.path.exists(SHAP_Active_path):
                    os.mkdir(SHAP_Active_path)
                for realName in layerNames:
                    nowPath = path + '/coef_nn_fold_' + str(i) + '_layer' + realName + '.csv'
                    nowData = rankings[realName]       ### 获取当前具体某一层他的各个元素所对应的AAAI可解释结果！
                    mid = pd.DataFrame(list(nowData.items()))
                    mid.to_csv(nowPath, header=['element', 'coef'], index=False)

                    Our_nowPath = SHAP_Active_path + '/coef_nn_fold_' + str(i) + '_layer' + realName + '.csv'
                    Our_nowData = SHAP_Active_rankings[realName]       ### 获取当前具体某一层他的各个元素所对应的AAAI可解释结果！
                    print("那么现在的这个数据是什么样子的呢？", type(Our_nowData))
                    mid = pd.DataFrame(list(Our_nowData.items()))
                    mid.to_csv(Our_nowPath, header=['element', 'coef'], index=False)





            # 0.821052634       23.9.7-fs-Active
            ### 此时其实可以在消融前求出当前这一fold所对应的可解释性结果，根据当前折一折的可解释结果来消融当前这一fold的结果更加具有真实性！
            ### 因为上面所调用的fit()函数是进行了可解释性操作的，会计算出当前所对应的可解释性结果！



            #### 下面这块是来进行消融操作的！！
            ## 在进行消融之前，首先要明白！他是只对当前的这个网络模型来进行消融的，其他的对标算法则不进行消融处理！因此在进行消融之前，还需要提前分析判断一下当前是不是P-net_ALL 网络模型！
            IFXiaoRong = False           ### 这个决定着当前是否要来进行消融！！！
            if model_params['type'] == 'nn' and IFXiaoRong:          ### 这个就表明当前所处理的这个算法是这个P-net网络模型！   此时是可以进行消融的，其他的算法一律不可！！
                ### 下面开始进行消融操作！
                ## 消融之前，首先需要先构造一下当前要消融的的一些参数：当前是五折运算中的哪一折（这个决定来选取哪个可解释结果！）；以及当前要进行消除的元素的比例（各层具体是要消除百分之多少的数据！）
                nowFold = i
                SuanFaName = 'AAAI'
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # SuanFaName = 'Active'
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                SuanFaName = 'SHAP'
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)         ## SHAP-Active     Logict-Active
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # # SuanFaName = 'SHAP-Active'
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                SuanFaName = 'Logict-Active'
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                SuanFaName = 'Grad'
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                SuanFaName = 'DeepLIFT'
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                print("3.13现在进行到进行可解释性消融了！")
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')



                ### 下面来处理一些加知识的操作！
                SuanFaName = 'AAAI-Kno'
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # SuanFaName = 'Active-Kno'
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                SuanFaName = 'SHAP-Kno'
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)         ## SHAP-Active     Logict-Active
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # # SuanFaName = 'SHAP-Active'
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                SuanFaName = 'Logict-Active-Kno'
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                SuanFaName = 'Grad-Kno'
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                SuanFaName = 'DeepLIFT-Kno'
                print("现在测试的！现在·正在进行消融处理！！")
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')






            y_pred_test, y_pred_test_scores = self.predict(model, x_test, y_test)    ### 训练完成后，拿测试数据来验证一下，获得测试数据的预测得分     PS!!千万注意！在这的这个self.predict()是指来调用当前CrossvalidationPipeline类内部的predict()函数，而当前的CrossvalidationPipeline类继承自OneSplitPipeline这个类
            ### 而OneSplitPipeline这个类在 pipeline.one_split.py 这个文件中。    而且对于predict()这个函数，当前CrossvalidationPipeline类并没有对他进行重写  因此他是直接调的pipeline.one_split.py 这个文件中  OneSplitPipeline这个类的predict()这个函数


            print("当前处于crossvalidation_pipeline.py文件中，现在已经获取关于训练数据的预测评分了！下面开始对这个得分进行1评估")
            score_test = self.evaluate(y_test, y_pred_test, y_pred_test_scores)
            print("上天护佑！！现在来测试一下，单论测试时这个测试分数是谁！", score_test)



            logging.info('当前处于crossvalidation_pipeline.py文件中，model {} -- Test score {}'.format(model_name, score_test))
            self.save_prediction(info_test, y_pred_test, y_pred_test_scores, y_test, i, model_name)

            if hasattr(model, 'save_model'):           ### 当当前这个模型训练好之后对其进行保存！
                logging.info('目前在crossvalidation_pipeline.py这个文件中，saving coef')
                save_model(model, model_name + '_' + str(i), self.directory)                         ### 测试的，这个位置模型进行保存时会出错！       正常情况下这个位置是不应该注释的！

            if self.save_train:
                logging.info('目前在crossvalidation_pipeline.py这个文件中，predicting training ...')
                y_pred_train, y_pred_train_scores = self.predict(model, x_train, y_train)
                self.save_prediction(info_train, y_pred_train, y_pred_train_scores, y_train, i, model_name,
                                     training=True)

            scores.append(score_test)

            fs_parmas = deepcopy(model_params)
            if hasattr(fs_parmas, 'id'):
                fs_parmas['id'] = fs_parmas['id'] + '_fold_' + str(i)
            else:
                fs_parmas['id'] = fs_parmas['type'] + '_fold_' + str(i)

            model_list.append((model, fs_parmas))
            i += 1
        self.save_coef(model_list, cols)        ## 此处极为关键，它意味着要使用one_split.py文件中的OneSplitPipeline类的save_coef() 函数    此处就是将算出来的这个重要性分数进行保存！
        logging.info(scores)

        ### 最后在各轮循环结束之后，想办法将激活网络的输出结果值，拿出来取一下平均
        self.getAvgActNodel()

        return scores


    ### 下面这个专门是用来处理经过PCA降维后输入到激活网络模型，来获取对应的预测精度！
    def train_predict_crossvalidation_PCAACt(self, model_params, X, y, info, cols, model_name):
        logging.info('当前处于crossvalidation_pipeline.py文件中， 目前的模型参数情况为：model_params: {}'.format(model_params))
        n_splits = self.pipeline_params['params']['n_splits']
        skf = StratifiedKFold(n_splits=n_splits, random_state=123, shuffle=True)       ## 进行划分进行交叉验证！    此时数据已经打乱了
        i = 0
        scores = []
        model_list = []

        KuoZengFlag = True                ### 这个表示对当前的这个数据集要不要进行扩增！
        if model_params['type'] == 'nn' and KuoZengFlag:
            ### 下方这个为2023.8.1修改代码，因为输入数据的正负样本极不均衡，接近1:2，因此之前的代码手段是从这些样本数据中随机的抽取数据，这样很有可能在其中一折随机抽取的数据中阳性样本非常多，阴性样本非常少，使得总体更不均衡！
            ### 因此针对上述这个问题，即为阳性样本与阴性样本分开来抽，保证每一折的训练数据中，阴性与阳性样本的比例均与总体样本保持一致！
            ### 现在的这个训练数据样本分布不均衡，得想办法在训练的时候使其均衡！
            print("现在传过来的这批训练数据的情况是怎样的！", type(X), type(y), len(X), len(y))
            ### 现在就是想办法将这个训练数据给打乱！
            dataIndex = []
            for j in range(len(X)):
                dataIndex.append(j)
            # np.random.shuffle(dataIndex)
            NewNum = int(0.20*len(dataIndex))             ### 现在是需要补充多少数据！
            NowIndex = dataIndex[0:NewNum]
            NewX, Newy, Newinfo = X[NowIndex], y[NowIndex], info[NowIndex]
            ### 现在要往数据集中补充哪些阳性数据已经知道了，现在开始重新构造这个数据集！
            X, y, info = np.concatenate([X, NewX], axis=0), np.concatenate([y, Newy], axis=0), np.concatenate([info, Newinfo], axis=0)

        ### 下面开始进行PCA降维操作！
        PCAFlag = False
        if PCAFlag:
            # PCA降维
            from sklearn.decomposition import PCA
            pca = PCA(n_components=28)  ### 下面这个就是将输入数据的维度给降到28维
            X = pca.fit_transform(X)
            print("上天保佑啊！！现在经过PCA降维后的x数据形状为：", X.shape)


        ### 下面开始进行五折交叉运算！
        for train_index, test_index in skf.split(X, y.ravel()):
            print("现在是测试环节！当前处于crossvalidation_pipeline.py文件中，现在所求出来的索引和原始数据的情况是怎样的！", type(train_index), len(train_index), len(test_index))

            ### 在这，这个就相当于是 每次从中截取固定长度的一批数据用作训练数据进行训练模型！
            # model = get_model(model_params)         ## 其实就是根据模型参数文件中的那些模型的超参数来初始化设置这个模型！

            ### 首先在这再重新构造一个简单的网络模型
            from keras.models import Sequential
            from keras.layers import Dense
            model = Sequential()

            input_dim = X.shape[1]
            model.add(Dense(10, activation='relu', input_shape=(input_dim,)))
            model.add(Dense(1, activation='sigmoid'))
            # 编译模型
            model.compile(optimizer='adam', loss='binary_crossentropy')




            logging.info('fold # ----------------%d---------' % i)
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            info_train = pd.DataFrame(index=info[train_index])
            info_test = pd.DataFrame(index=info[test_index])
            x_train, x_test = self.preprocess(x_train, x_test)
            # feature extraction
            logging.info('当前处于crossvalidation_pipeline.py文件中，现在进行特征提取！  feature extraction....')
            x_train, x_test = self.extract_features(x_train, x_test)
            print("当前是在pipeline/crossvalidation_pipeline.py文件中，现在在五折交叉验证中所用的数据的长度是怎样的！", len(x_train), len(x_test))   ### 因为是五折交叉验证，因此在原来分好的训练集上在此分成五份，前四分用来训练后一份用来测试

            FCN_x_train, FCN_y_train, FCN_x_test, FCN_y_test = x_train, y_train, x_test, y_test


            ### 可以在这构造一个简单的神经网络测试一下
            FCNCeshiFlag, GeneCeshiFlag = False, False
            if FCNCeshiFlag:
                input_dim = len(FCN_x_train[0])  ### 现在就代表输入的这个样本数据他的对应的维度，其实就是这一层中神经元的数目！
                # 创建模型
                from keras.models import Sequential
                from keras.layers import Dense
                FCNmodel = Sequential()
                FCNmodel.add(Dense(10, activation='relu', input_shape=(input_dim,)))
                FCNmodel.add(Dense(1, activation='sigmoid'))
                # 编译模型
                FCNmodel.compile(optimizer='adam', loss='binary_crossentropy')

                # 训练模型
                FCNmodel.fit(FCN_x_train, FCN_y_train, epochs=100, batch_size=32)
                # 进行模型测试
                # valPredict = model.predict(valX)  ### 这个是对验证数据的预测结果！！
                # # threshold = self.get_th(valY, valPredict)       ### 根据验证数据来获取最佳阈值！    他的输入参数有两个，一个是验证数据标签，一个是输入参数！
                threshold = 0.5
                predictions = FCNmodel.predict(FCN_x_test)  ### 现在根据测试样本来进行预测获取对应的预测结果！
                binary_predictions = np.where(predictions > threshold, 1, 0)  ### 根据获得的阈值来对预测结果进行二值化处理！
                # 计算Accuracy、Precision、Recall、AUPRC、AUC和F1值
                from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score, \
                    roc_auc_score, f1_score
                accuracy = accuracy_score(FCN_y_test, binary_predictions)
                precision = precision_score(FCN_y_test, binary_predictions)
                recall = recall_score(FCN_y_test, binary_predictions)
                auprc = average_precision_score(FCN_y_test, predictions)
                auc = roc_auc_score(FCN_y_test, predictions)
                f1 = f1_score(FCN_y_test, binary_predictions)
                # test_loss, test_accuracy = model.evaluate(testX, testY, verbose=2)
                # print("测试集上的损失值：", test_loss)
                print("现在是来测试一下拿原始的输入数据作为简单的全连接模型的输入，最终当前的这个测试集上的预测精度：", len(FCN_x_train), input_dim, accuracy, precision, recall, auprc,
                      auc, f1)


            if GeneCeshiFlag:
                input_dim = len(FCN_x_train[0])  ### 现在就代表输入的这个样本数据他的对应的维度，其实就是这一层中神经元的数目！
                # 创建模型
                from keras.models import Sequential
                from keras.layers import Dense
                from model.layers_custom import Diagonal, SparseTF
                from keras.regularizers import l2
                from data.pathways.reactome import ReactomeNetwork
                import itertools
                from data.data_access import Data

                selected_genes = 'tcga_prostate_expressed_genes_and_cancer_genes.csv'
                data_params = {'id': 'ALL', 'type': 'prostate_paper',
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
                data = Data(**data_params)
                x, y, info, cols = data.get_data()  ### 再次获取训练所用的数据情况！
                if hasattr(cols, 'levels'):
                    genes = cols.levels[0]
                else:
                    genes = cols
                ## 获取当前层与下一层之间的关联关系
                def get_map_from_layer(layer_dict):
                    pathways = list(layer_dict.keys())
                    print("目前所处的文件为：builders_utils.py")
                    print('目前从层字典中读取的通路的数目，pathways', len(pathways))
                    genes = list(itertools.chain.from_iterable(
                        list(layer_dict.values())))  ## 在这itertools.chain.from_iterable表示是将多个迭代器进行高效的连接，连接成一个迭代器！
                    genes = list(np.unique(genes))
                    print('目前从层字典中读取的基因的数目，genes', len(genes))
                    ### 在这里，各个网络层中的基因与通路并不一定真的就代表给基因和通路，此二者只是表示当前层的上下级关系，基因下一级，通路上一级！

                    n_pathways = len(pathways)
                    n_genes = len(genes)

                    mat = np.zeros((n_pathways,
                                    n_genes))  ### 在这，纵坐标就代表通路（就是下一层的大通路），横坐标就代表基因（就是当前层的小通路）   现在就是来构造上一层和下一层之间的关联关系矩阵图 mat   最开始全都初始化为0，如果彼此之间有关系的就设置为1
                    for p, gs in list(layer_dict.items()):
                        g_inds = [genes.index(g) for g in gs]  ### 就是挨个的遍历来找一下，当前的这个通路中是否包含这个基因，有的话，那么对应的位置就是1
                        p_ind = pathways.index(p)
                        mat[p_ind, g_inds] = 1  ## 就是说如果该位置的基因和通路彼此之间确实存在一定的关系，那么这个关系图中的这个位置就设置为1

                    df = pd.DataFrame(mat, index=pathways, columns=genes)
                    # for k, v in layer_dict.items():
                    #     print k, v
                    #     df.loc[k,v] = 1
                    # df= df.fillna(0)
                    return df.T
                ## 获取网络中每一层他的前后连接关系，并将各个连接关系进行汇总！
                def get_layer_maps(genes, n_levels, direction, add_unk_genes):
                    reactome_layers = ReactomeNetwork().get_layers(n_levels,
                                                                   direction)  ## 现在的这个reactome_layers就是一个字典，对应着上一层与他对应的下一层元素之间的关系
                    filtering_index = genes
                    # print("目前是在 builders/builders_utils.py 文件中，当前的这个genes是谁！", genes)
                    # print("现在的这个网络层是怎样的！", reactome_layers)
                    # print("目前是在 builders/builders_utils.py 文件中，现在将这个genes详细展开会是怎样的！！", list(genes))
                    maps = []
                    for i, layer in enumerate(reactome_layers[::-1]):  ## 这里面的每一层都蕴含上下级关系！
                        print('layer #', i)
                        # print("测一下，目前所读取的这个layer是谁！", layer)
                        mapp = get_map_from_layer(layer)  ## 获取当前层他的上下级关系（0、1矩阵图）        纵轴代表小通路，横轴代表大通路
                        filter_df = pd.DataFrame(index=filtering_index)
                        print('目前所处的文件为：builders_utils.py  ， filtered_map', filter_df.shape)
                        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True,
                                                       how='left')  ## 将两个DataFrame进行合并   就是将filter_df与mapp进行合并
                        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
                        print('filtered_map', filter_df.shape)
                        # print("当前是在 builders/builders_utils.py 文件中，现在，他的这个mapp图是怎样的！", mapp)
                        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')

                        # UNK, add a node for genes without known reactome annotation    UNK，为没有已知reactome注释的基因添加一个节点
                        if add_unk_genes:
                            print('UNK ')
                            filtered_map['UNK'] = 0
                            ind = filtered_map.sum(axis=1) == 0  ### 二维数组中他的水平行上的数据进行求和
                            filtered_map.loc[ind, 'UNK'] = 1
                        ####

                        filtered_map = filtered_map.fillna(0)
                        print('目前所在的文件是：builders_utils.py  filtered_map', filter_df.shape)
                        # filtering_index = list(filtered_map.columns)
                        filtering_index = filtered_map.columns
                        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
                        # print("当前所在的文件是pnet_prostate_paper-published_to_zenodo/model/builders/builders_utils.py，当前具体某一层映射他的关系具体情况为！", filtered_map)      ### 当前所构造的这个filtered_map，他的纵轴的标题为上一层的元素（基因），他的横轴标题为下一层的元素（通路）
                        # print("当前的这个map，他的列情况是怎样的！", len(filtered_map.columns), filtered_map.columns)

                        # print("当前是在pnet_prostate_paper-published_to_zenodo/model/builders/builders_utils.py文件中，当前的这个关系表的具体数据内容是怎样的！", filtered_map.value)
                        # print("当前是在pnet_prostate_paper-published_to_zenodo/model/builders/builders_utils.py文件中，读取一下这个关系表的某一行数据！", np.sum(filtered_map.iloc[1]), filtered_map.iloc[1])
                        # print("当前的这个map，他的行情况是怎样的！", len(filtered_map.rows), filtered_map.rows)
                        maps.append(filtered_map)
                    return maps

                n_hidden_layers, direction, add_unk_genes = 5, 'root_to_leaf', True
                maps = get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes)
                mapp = maps[0]

                OriMapp = mapp  ### 保留一下原始的映射关系图！
                mapp = mapp.values
                n_genes, n_pathways = mapp.shape

                FCNmodel = Sequential()
                layer1 = Diagonal(9229, input_shape=(input_dim,), activation='tanh', W_regularizer=l2(0.001),
                                  use_bias=True, name='h0', kernel_initializer='lecun_uniform')
                hidden_layer1 = SparseTF(1387, mapp, activation='tanh', W_regularizer=l2(0.001),
                                        name='h1', kernel_initializer='lecun_uniform',
                                        use_bias=True)       #### 构建其中的一个隐藏层！   在这是需要构建一个稀疏的网络层

                FCNmodel.add(layer1)
                FCNmodel.add(hidden_layer1)
                FCNmodel.add(Dense(1, activation='sigmoid'))
                # 编译模型
                FCNmodel.compile(optimizer='adam', loss='binary_crossentropy')

                # 训练模型
                FCNmodel.fit(FCN_x_train, FCN_y_train, epochs=80, batch_size=32)
                # 进行模型测试
                # valPredict = model.predict(valX)  ### 这个是对验证数据的预测结果！！
                # # threshold = self.get_th(valY, valPredict)       ### 根据验证数据来获取最佳阈值！    他的输入参数有两个，一个是验证数据标签，一个是输入参数！
                threshold = 0.5
                predictions = FCNmodel.predict(FCN_x_test)  ### 现在根据测试样本来进行预测获取对应的预测结果！
                binary_predictions = np.where(predictions > threshold, 1, 0)  ### 根据获得的阈值来对预测结果进行二值化处理！
                # 计算Accuracy、Precision、Recall、AUPRC、AUC和F1值
                from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score, \
                    roc_auc_score, f1_score
                accuracy = accuracy_score(FCN_y_test, binary_predictions)
                precision = precision_score(FCN_y_test, binary_predictions)
                recall = recall_score(FCN_y_test, binary_predictions)
                auprc = average_precision_score(FCN_y_test, predictions)
                auc = roc_auc_score(FCN_y_test, predictions)
                f1 = f1_score(FCN_y_test, binary_predictions)
                # test_loss, test_accuracy = model.evaluate(testX, testY, verbose=2)
                # print("测试集上的损失值：", test_loss)
                print("现在是来测试一下拿原始的输入数据作为简单的全连接模型的输入，最终当前的这个测试集上的预测精度：", len(FCN_x_train), input_dim, accuracy, precision, recall, auprc,
                      auc, f1)




            # print("现在这个训练之前这个列标题是谁！", type(cols), cols)
            ### 下面是测试部分，来将这个列标题保存到指定的位置！
            newcols = cols.to_frame()
            newcols.to_csv("../CeShi/OtherTest/Other/GeneColsNames.csv")
            print("现在这个输入数据的列标题（27687个）数据文件保存好了！！")


            # 训练模型
            model.fit(x_train, y_train, epochs=100, batch_size=32)

            # if model_params['type'] == 'nn':
            #     model = model.fit(x_train, y_train, None, None, i)       ### 提取好特征之后，现在开始进行拟合模型进行训练           最后一个参数为fold表示当前处于第几折当中，主要是为了获得当前折可解释结果后保存文件所用！
            # else:
            #     model = model.fit(x_train, y_train)

            print("现在是在pipeline/crossvalidation_pipeline.py文件中，这个测试数据的样子是什么样的！", len(x_test), len(x_test[0]), type(x_test), type(x_test[0]), x_test[0], x_test)
            # x_test = self.TestDataAddNoiseTest5_19(x_test)          ### 现在是来进行测试的部分，对这个测试数据来进行噪音的干扰



            ### 现在这玩意是来根据根据不同的测试数据来获取每个神经元的相似性！     这块其实就是AAAI论文的那个算法的复现手段！
            AAAI = False                    ### 下面这块决定是否要使用AAAI的算法来获取各层网络中各个元素的重要性分数，而且再将它们进行保存！
            if model_params['type'] == 'nn' and AAAI:                  ### 在这，必须专门指明一下 model_params['type'] == 'nn'  因为只有当前的这个模型才会执行这个AAAI的操作！！其他的对标均不会执行！
                rankings = model.get_CorrAll(x_test)                           ### 现在获取得到的这个rankings是一个字典，key是各层网络的名字，value又是一个字典，他的key是这层网络的具体的元素的名字，value则是这个元素对应的方差值！
                print("上天保佑啊！！现在这个神经元的排名情况是怎样的！", rankings)
                ### 现在就要想办法将目前所读取的这个AAAI进行排序得到的元素的排名给保存一下！
                layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']           ### 分别读取各层来进行处理！
                path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-AAAI'        ## _logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs         下面这块来指定各轮中各层各个元素他们的AAAI方法得到的排名结果中的元素情况
                if not os.path.exists(path):
                    os.mkdir(path)
                for realName in layerNames:
                    nowPath = path + '/coef_nn_fold_' + str(i) + '_layer' + realName + '.csv'
                    nowData = rankings[realName]       ### 获取当前具体某一层他的各个元素所对应的AAAI可解释结果！
                    mid = pd.DataFrame(list(nowData.items()))
                    mid.to_csv(nowPath, header=['element', 'coef'], index=False)


            ### 现在这玩意是来根据根据不同的测试数据来获取每个神经元的相似性！     这块其实就是AAAI论文的那个算法的复现手段！
            Active = False                    ### 下面这块决定是否要使用AAAI的算法来获取各层网络中各个元素的重要性分数，而且再将它们进行保存！
            if model_params['type'] == 'nn' and Active:                  ### 在这，必须专门指明一下 model_params['type'] == 'nn'  因为只有当前的这个模型才会执行这个AAAI的操作！！其他的对标均不会执行！
                rankings = model.get_ActivationsAll(X)          ### 本来这个参数位置是 x_test                   ### 现在获取得到的这个rankings是一个字典，key是各层网络的名字，value又是一个字典，他的key是这层网络的具体的元素的名字，value则是这个元素对应的方差值！
                print("上天保佑啊！！目前是按照神经元激活频率来进行排序！现在这个神经元的排名情况是怎样的！", rankings)
                ### 现在就要想办法将目前所读取的这个AAAI进行排序得到的元素的排名给保存一下！
                layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']           ### 分别读取各层来进行处理！
                path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-Active'        ## _logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs         下面这块来指定各轮中各层各个元素他们的AAAI方法得到的排名结果中的元素情况
                if not os.path.exists(path):
                    os.mkdir(path)
                for realName in layerNames:
                    nowPath = path + '/coef_nn_fold_' + str(i) + '_layer' + realName + '.csv'
                    nowData = rankings[realName]       ### 获取当前具体某一层他的各个元素所对应的AAAI可解释结果！
                    mid = pd.DataFrame(list(nowData.items()))
                    mid.to_csv(nowPath, header=['element', 'coef'], index=False)




            ### 现在这玩意是来根据根据不同的测试数据来获取每个神经元的相似性！     这块其实就是AAAI论文的那个算法的复现手段！
            Logict_Active = False                    ### 下面这块决定是否要使用AAAI的算法来获取各层网络中各个元素的重要性分数，而且再将它们进行保存！
            if model_params['type'] == 'nn' and Logict_Active:                  ### 在这，必须专门指明一下 model_params['type'] == 'nn'  因为只有当前的这个模型才会执行这个AAAI的操作！！其他的对标均不会执行！
                logictActParams = {'Y': y}            ### 这个是来获取拟合逻辑回归所用的y标签参数！
                rankings = model.get_ActivationsAll(X, logictActParams)                           ### 现在获取得到的这个rankings是一个字典，key是各层网络的名字，value又是一个字典，他的key是这层网络的具体的元素的名字，value则是这个元素对应的方差值！
                print("上天保佑啊！！目前是按照神经元激活频率来进行排序！现在这个神经元的排名情况是怎样的！", rankings)
                ### 现在就要想办法将目前所读取的这个AAAI进行排序得到的元素的排名给保存一下！
                layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']           ### 分别读取各层来进行处理！
                path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-Logict-Active-Kno'  ## _logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs         下面这块来指定各轮中各层各个元素他们的AAAI方法得到的排名结果中的元素情况
                path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-Logict-Active'
                if not os.path.exists(path):
                    os.mkdir(path)
                for realName in layerNames:
                    nowPath = path + '/coef_nn_fold_' + str(i) + '_layer' + realName + '.csv'
                    nowData = rankings[realName]       ### 获取当前具体某一层他的各个元素所对应的AAAI可解释结果！
                    mid = pd.DataFrame(list(nowData.items()))
                    mid.to_csv(nowPath, header=['element', 'coef'], index=False)





            ### 下面尝试一下将SHAP的可解释结果与这个激活结果进行组合一下，看看最终的效果是怎样的！    而且这玩意最终
            SHAP_Active = False                    ### 下面这块决定是否要使用AAAI的算法来获取各层网络中各个元素的重要性分数，而且再将它们进行保存！
            if model_params['type'] == 'nn' and SHAP_Active:                  ### 在这，必须专门指明一下 model_params['type'] == 'nn'  因为只有当前的这个模型才会执行这个AAAI的操作！！其他的对标均不会执行！
                rankings = model.get_ActivationsAll(x_test)                           ### 现在获取得到的这个rankings是一个字典，key是各层网络的名字，value又是一个字典，他的key是这层网络的具体的元素的名字，value则是这个元素对应的方差值！
                print("上天保佑啊！！目前是按照神经元激活频率来进行排序！现在这个神经元的排名情况是怎样的！", rankings)
                ### 接下来就要分别对SHAP的结果以及当前的这个激活算法的结果先进行一下归一化都归一化到0~1之间的范围之后再进行操作
                def normalize_dict_values(input_dict):
                    # 提取原字典的值构成一个二维数组
                    X = [[value] for value in input_dict.values()]

                    # 使用MinMaxScaler对值进行归一化处理
                    scaler = MinMaxScaler()                 ### 现在这个 feature_range=(-1, 1)参数决定了最终的归一化范围，不写的话就是（0,1）
                    normalized_values = scaler.fit_transform(X).flatten()

                    # 构建归一化后的字典，键值对关系与原字典保持一致
                    normalized_dict = {}
                    for index, (key, value) in enumerate(input_dict.items()):
                        normalized_dict[key] = normalized_values[index]
                    return normalized_dict
                ### 下面这块就对激活方法获取的各个字典进行归一化处理！
                for active_key in rankings:
                    rankings[active_key] = normalize_dict_values(rankings[active_key])        ### 现在对其中的各个字典来进行归一化处理


                ### 下面来获取SHAP文件夹中的数据
                SHAP_path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-SHAP'
                ### 下面这个文件是具体的来读取SHAP文件中的数据并将其构造为一个字典形式
                def read_csv_to_dict(filename, key_col, value_col):
                    data_dict = {}
                    with open(filename, 'r', newline='') as file:
                        reader = csv.reader(file)
                        for row in reader:
                            key = row[key_col]
                            value = row[value_col]
                            data_dict[key] = value
                    return data_dict
                layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']
                key_col = 0  # key所在列的索引，假设第一列为key
                value_col = 1  # value所在列的索引，假设第二列为value
                SHAP_Active_rankings = {}
                for shapLayerName in layerNames:
                    nowSHAP_path = SHAP_path + '/coef_nn_fold_' + str(i) + '_layer' + shapLayerName + '.csv'
                    data_dict = read_csv_to_dict(nowSHAP_path, key_col, value_col)

                    del data_dict['element']            ### 首行元素给删掉！
                    # ### 因为现在这个SHAP文件有些特殊，没有element那个标题，因此要想办法根据他的value值来进行删除！
                    # def remove_items_with_value(d, value):
                    #     return {key: val for key, val in d.items() if val != value}
                    # data_dict = remove_items_with_value(data_dict, 'coef')  ### 按照value值来进行删除！

                    data_dict = normalize_dict_values(data_dict)
                    ### 此时得到的data_dict 这个字典就是经过归一化的当前这一层中SHAP的结果值
                    Act_dict = rankings[shapLayerName]
                    result_dict = {}   ### 这个就是最终的融合后的结果
                    for key in data_dict:
                        if key in Act_dict:
                            value1 = data_dict[key]
                            value2 = Act_dict[key]
                            result_dict[key] = 0.8*value1 + 0.3*value2

                    ### 别忘了，此时还有最关键的一步！一定要记得排序！
                    def sort_dict_by_value(d):       ### 现在就是按照value取值从大到校进行排序！  同时还要保证排序后的数据还是一个字典的形式，方便后续的操作！
                        sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
                        return {k: v for k, v in sorted_items}
                    result_dict = sort_dict_by_value(result_dict)             ### 现在就是按照value取值从大到校进行排序！
                    SHAP_Active_rankings[shapLayerName] = result_dict




                ### 现在就要想办法将目前所读取的这个AAAI进行排序得到的元素的排名给保存一下！
                layerNames = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5']           ### 分别读取各层来进行处理！
                path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-Active'        ## _logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs         下面这块来指定各轮中各层各个元素他们的AAAI方法得到的排名结果中的元素情况
                SHAP_Active_path = '../_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-SHAP-Active'
                if not os.path.exists(path):
                    os.mkdir(path)
                if not os.path.exists(SHAP_Active_path):
                    os.mkdir(SHAP_Active_path)
                for realName in layerNames:
                    nowPath = path + '/coef_nn_fold_' + str(i) + '_layer' + realName + '.csv'
                    nowData = rankings[realName]       ### 获取当前具体某一层他的各个元素所对应的AAAI可解释结果！
                    mid = pd.DataFrame(list(nowData.items()))
                    mid.to_csv(nowPath, header=['element', 'coef'], index=False)

                    Our_nowPath = SHAP_Active_path + '/coef_nn_fold_' + str(i) + '_layer' + realName + '.csv'
                    Our_nowData = SHAP_Active_rankings[realName]       ### 获取当前具体某一层他的各个元素所对应的AAAI可解释结果！
                    print("那么现在的这个数据是什么样子的呢？", type(Our_nowData))
                    mid = pd.DataFrame(list(Our_nowData.items()))
                    mid.to_csv(Our_nowPath, header=['element', 'coef'], index=False)





            # 0.821052634       23.9.7-fs-Active
            ### 此时其实可以在消融前求出当前这一fold所对应的可解释性结果，根据当前折一折的可解释结果来消融当前这一fold的结果更加具有真实性！
            ### 因为上面所调用的fit()函数是进行了可解释性操作的，会计算出当前所对应的可解释性结果！



            #### 下面这块是来进行消融操作的！！
            ## 在进行消融之前，首先要明白！他是只对当前的这个网络模型来进行消融的，其他的对标算法则不进行消融处理！因此在进行消融之前，还需要提前分析判断一下当前是不是P-net_ALL 网络模型！
            IFXiaoRong = False           ### 这个决定着当前是否要来进行消融！！！
            if model_params['type'] == 'nn' and IFXiaoRong:          ### 这个就表明当前所处理的这个算法是这个P-net网络模型！   此时是可以进行消融的，其他的算法一律不可！！
                ### 下面开始进行消融操作！
                ## 消融之前，首先需要先构造一下当前要消融的的一些参数：当前是五折运算中的哪一折（这个决定来选取哪个可解释结果！）；以及当前要进行消除的元素的比例（各层具体是要消除百分之多少的数据！）
                nowFold = i
                # SuanFaName = 'AAAI'
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # # SuanFaName = 'Active'
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # SuanFaName = 'SHAP'
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)         ## SHAP-Active     Logict-Active
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # # # SuanFaName = 'SHAP-Active'
                # # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                SuanFaName = 'Logict-Active'
                self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # SuanFaName = 'Grad'
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # SuanFaName = 'DeepLIFT'
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)



                # ### 下面来处理一些加知识的操作！
                # SuanFaName = 'AAAI-Kno'
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # # SuanFaName = 'Active-Kno'
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # SuanFaName = 'SHAP-Kno'
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)         ## SHAP-Active     Logict-Active
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # # # SuanFaName = 'SHAP-Active'
                # # # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # SuanFaName = 'Logict-Active-Kno'
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # SuanFaName = 'Grad-Kno'
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')
                # SuanFaName = 'DeepLIFT-Kno'
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model)
                # self.XiaoRongProcess(SuanFaName, nowFold, x_test, y_test, model, 'Neg')




            threshold = 0.5
            predictions = model.predict(x_test)        ### 现在根据测试样本来进行预测获取对应的预测结果！
            binary_predictions = np.where(predictions > threshold, 1, 0)           ### 根据获得的阈值来对预测结果进行二值化处理！
            # 计算Accuracy、Precision、Recall、AUPRC、AUC和F1值
            from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score, \
                roc_auc_score, f1_score
            accuracy = accuracy_score(y_test, binary_predictions)
            precision = precision_score(y_test, binary_predictions)
            recall = recall_score(y_test, binary_predictions)
            auprc = average_precision_score(y_test, predictions)
            auc = roc_auc_score(y_test, predictions)
            f1 = f1_score(y_test, binary_predictions)
            score_test = {}
            score_test['accuracy'], score_test['precision'], score_test['auc'], score_test['f1'], score_test['aupr'], score_test['recall'] = accuracy, precision, auc, f1, auprc, recall




            # y_pred_test, y_pred_test_scores = self.predict(model, x_test, y_test)    ### 训练完成后，拿测试数据来验证一下，获得测试数据的预测得分     PS!!千万注意！在这的这个self.predict()是指来调用当前CrossvalidationPipeline类内部的predict()函数，而当前的CrossvalidationPipeline类继承自OneSplitPipeline这个类
            # ### 而OneSplitPipeline这个类在 pipeline.one_split.py 这个文件中。    而且对于predict()这个函数，当前CrossvalidationPipeline类并没有对他进行重写  因此他是直接调的pipeline.one_split.py 这个文件中  OneSplitPipeline这个类的predict()这个函数
            #
            #
            # print("当前处于crossvalidation_pipeline.py文件中，现在已经获取关于训练数据的预测评分了！下面开始对这个得分进行1评估")
            # score_test = self.evaluate(y_test, y_pred_test, y_pred_test_scores)
            # print("上天护佑！！现在来测试一下，单论测试时这个测试分数是谁！", score_test)
            #
            #
            #
            # logging.info('当前处于crossvalidation_pipeline.py文件中，model {} -- Test score {}'.format(model_name, score_test))
            # self.save_prediction(info_test, y_pred_test, y_pred_test_scores, y_test, i, model_name)

            if hasattr(model, 'save_model') and False:           ### 当当前这个模型训练好之后对其进行保存！
                logging.info('目前在crossvalidation_pipeline.py这个文件中，saving coef')
                save_model(model, model_name + '_' + str(i), self.directory)                         ### 测试的，这个位置模型进行保存时会出错！       正常情况下这个位置是不应该注释的！

            if self.save_train and False:
                logging.info('目前在crossvalidation_pipeline.py这个文件中，predicting training ...')
                y_pred_train, y_pred_train_scores = self.predict(model, x_train, y_train)
                self.save_prediction(info_train, y_pred_train, y_pred_train_scores, y_train, i, model_name,
                                     training=True)

            scores.append(score_test)

            fs_parmas = deepcopy(model_params)
            if hasattr(fs_parmas, 'id'):
                fs_parmas['id'] = fs_parmas['id'] + '_fold_' + str(i)
            else:
                fs_parmas['id'] = fs_parmas['type'] + '_fold_' + str(i)

            model_list.append((model, fs_parmas))
            i += 1
        self.save_coef(model_list, cols)        ## 此处极为关键，它意味着要使用one_split.py文件中的OneSplitPipeline类的save_coef() 函数    此处就是将算出来的这个重要性分数进行保存！
        logging.info(scores)
        return scores










    def save_score(self, data_params, model_params, scores, scores_mean, scores_std, model_name):
        file_name = join(self.directory, model_name + '_params' + '.yml')
        logging.info("目前在crossvalidation_pipeline.py这个文件中，saving yml : %s" % file_name)
        with open(file_name, 'w') as yaml_file:
            yaml_file.write(
                yaml.dump({'data': data_params, 'models': model_params, 'pre': self.pre_params,
                           'pipeline': self.pipeline_params, 'scores': scores.to_json(),
                           'scores_mean': scores_mean.to_json(), 'scores_std': scores_std.to_json()},
                          default_flow_style=False))

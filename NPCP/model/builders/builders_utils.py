

# from data.pathways.pathway_loader import get_pathway_files
import itertools
import logging

import numpy as np
import pandas as pd

import random

### 下面这块为测试部分
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Dense, Dropout, Activation, BatchNormalization, multiply, Layer
from keras.regularizers import l2
from keras.layers.core import Lambda


# from data.pathways.pathway_loader import get_pathway_files
from data.pathways.reactome import ReactomeNetwork
from model.layers_custom import Diagonal, SparseTF



from keras import backend as K
from keras.layers import Activation
import tensorflow


from CeShi.OtherTest.ModelParamSave.ModelParam_Two import LayerEleGSEA, LayerEleRelationship
import tensorflow as tf
from keras.layers import LeakyReLU

from sklearn.preprocessing import MinMaxScaler
from config_path import *
base_path = BASE_PATH                   ### 现在的这个base_path是D:\DailyCode\P-Net\CodeModel\(Father)pnet_prostate_paper-published_to_zenodo\pnet_prostate_paper-published_to_zenodo\






## 获取当前层与下一层之间的关联关系
def get_map_from_layer(layer_dict):
    pathways = list(layer_dict.keys())
    print("目前所处的文件为：builders_utils.py")
    print('目前从层字典中读取的通路的数目，pathways', len(pathways))
    genes = list(itertools.chain.from_iterable(list(layer_dict.values())))      ## 在这itertools.chain.from_iterable表示是将多个迭代器进行高效的连接，连接成一个迭代器！
    genes = list(np.unique(genes))
    print('目前从层字典中读取的基因的数目，genes', len(genes))
    ### 在这里，各个网络层中的基因与通路并不一定真的就代表给基因和通路，此二者只是表示当前层的上下级关系，基因下一级，通路上一级！

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))       ### 在这，纵坐标就代表通路（就是下一层的大通路），横坐标就代表基因（就是当前层的小通路）   现在就是来构造上一层和下一层之间的关联关系矩阵图 mat   最开始全都初始化为0，如果彼此之间有关系的就设置为1
    for p, gs in list(layer_dict.items()):
        g_inds = [genes.index(g) for g in gs]     ### 就是挨个的遍历来找一下，当前的这个通路中是否包含这个基因，有的话，那么对应的位置就是1
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1      ## 就是说如果该位置的基因和通路彼此之间确实存在一定的关系，那么这个关系图中的这个位置就设置为1

    df = pd.DataFrame(mat, index=pathways, columns=genes)
    # for k, v in layer_dict.items():
    #     print k, v
    #     df.loc[k,v] = 1
    # df= df.fillna(0)
    return df.T


## 获取网络中每一层他的前后连接关系，并将各个连接关系进行汇总！
def get_layer_maps(genes, n_levels, direction, add_unk_genes):
    reactome_layers = ReactomeNetwork().get_layers(n_levels, direction)                 ## 现在的这个reactome_layers就是一个字典，对应着上一层与他对应的下一层元素之间的关系
    filtering_index = genes
    # print("目前是在 builders/builders_utils.py 文件中，当前的这个genes是谁！", genes)
    # print("现在的这个网络层是怎样的！", reactome_layers)
    # print("目前是在 builders/builders_utils.py 文件中，现在将这个genes详细展开会是怎样的！！", list(genes))
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):      ## 这里面的每一层都蕴含上下级关系！
        print('layer #', i)
        # print("测一下，目前所读取的这个layer是谁！", layer)
        mapp = get_map_from_layer(layer)    ## 获取当前层他的上下级关系（0、1矩阵图）        纵轴代表小通路，横轴代表大通路
        filter_df = pd.DataFrame(index=filtering_index)
        print('目前所处的文件为：builders_utils.py  ， filtered_map', filter_df.shape)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')         ## 将两个DataFrame进行合并   就是将filter_df与mapp进行合并
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        print('filtered_map', filter_df.shape)
        # print("当前是在 builders/builders_utils.py 文件中，现在，他的这个mapp图是怎样的！", mapp)
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')

        # UNK, add a node for genes without known reactome annotation    UNK，为没有已知reactome注释的基因添加一个节点
        if add_unk_genes:
            print('UNK ')
            filtered_map['UNK'] = 0
            ind = filtered_map.sum(axis=1) == 0             ### 二维数组中他的水平行上的数据进行求和
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



def shuffle_genes_map(mapp):
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    # logging.info('shuffling the map')
    # mapp = mapp.T
    # np.random.shuffle(mapp)
    # mapp= mapp.T
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    logging.info('shuffling')
    ones_ratio = np.sum(mapp) / np.prod(mapp.shape)         ## 在这np.prod()  返回给定数组上的元素的乘积  让数组内的元素进行相乘！
    logging.info('ones_ratio {}'.format(ones_ratio))
    mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
    logging.info('random map ones_ratio {}'.format(ones_ratio))
    print(
        "当前所在的文件是pnet_prostate_paper-published_to_zenodo/model/builders/builders_utils.py，大乱之后，所返回的这个关系图为！",
        mapp)
    return mapp







### 现在这个是对某一层网络层的输出值中添加属于下一层网络神经元的GSEA值！   现在这个mapp表示当前层与下一层之间的映射关系！outcome表示当前层的输出！
### 最终返回的是当前这一层网络输入经过融合之后的那个输入！      这个也就相当于是在当前这层网络的输入中添加属于这层网络自己的GSEA值！
def AddGSEAScore(mapp, outcome, j):
    Gene = np.array(mapp.index)       ### 这个指代的是当前层网络中各个神经元所指代的生物学含义！   就是上一层！
    Pathway = np.array(mapp.columns)       ### 这个指代的是下一层网络中各个神经元所指代的生物学含义！    这两行其实主要是为了获取当前这层网络内部各个元素在此时他的分布顺序！

    ### 下面来获取当前这一层中神经元所代表的基因/通路他们所对应的GSEA数值！以及当前层中各个神经元节点所包含的上一层的神经元节点！
    LayerGSEA = LayerEleGSEA()
    LayerRelationship = LayerEleRelationship()
    if j == 0:                 ### 此时代表第一层通路层！
        GSEADict = LayerGSEA.getLayer1()
        RelationDict = LayerRelationship.getLayer1()
    elif j == 1:
        GSEADict = LayerGSEA.getLayer2()
        RelationDict = LayerRelationship.getLayer2()
    elif j == 2:
        GSEADict = LayerGSEA.getLayer3()
        RelationDict = LayerRelationship.getLayer3()
    elif j == 3:
        GSEADict = LayerGSEA.getLayer4()
        RelationDict = LayerRelationship.getLayer4()
    elif j == 4:
        GSEADict = LayerGSEA.getLayer5()
        RelationDict = LayerRelationship.getLayer5()
    else:
        print("说明此时出错了！索引变量j超过了一定的范围！", j)


    ### 现在下面这步应该是将得到的关系字典进行反向存储一下！即key与value两者进行对调一下！，看一下当前层中的元素包含下一层中的哪些元素！(就是上一层中某个神经元对应于当前层中的哪个神经元)  这决定当前网络的输出值对应加上谁！
    dict_ori, dict_new2 = RelationDict, {}
    for ele in dict_ori:
        Values = dict_ori[ele]  ### 现在这个Value就是一个数组
        for value in Values:
            if value in dict_new2.keys():
                dict_new2[value].append(ele)
                # print("按理来说，这个位置是不可能走到的！走到就报错了！！一个孩子是不能对应于多个父亲的", value, ele)                       ### 按理来说，这个位置是不可能走到的！走到就报错了！！一个孩子是不能对应于多个父亲的    但是现在在这确实是有错！   存在一个基因对应多个父亲通路
            else:
                dict_new2[value] = [ele]
                # print("正常应该在这啊！只有一个父亲！")
    # print("现在最终的这个字典的情况是怎样的！", len(RelationDict), len(dict_new2), dict_new2)
    NewRelationDict = dict_new2      ### 现在这个字典就是上一层中的神经元所对应的当前层的神经元有哪些！     现在这个字典中key是孩子基因，value是父亲通路！


    ### 现在开始构造当前这个输出值所对应的GSEA分数！   因为现在这个输出值他是对应于上一层网络的输出！因此在这是要将当前层的GSEA值融合到上一层网络中！   要用Gene
    GseaScore = []
    for ele in Gene:                  ### 逐个，按顺序的来读取各个基因（子通路）！在这，这个顺序不能乱！
        if ele not in NewRelationDict.keys():
            ### 这种情况就说明，当前层的基因节点并不与下一层相连，没有GSEA分数
            GseaScore.append(0)               ### 不与下一层相连，因此这里是没有GSEA分数的！
        elif len(NewRelationDict[ele]) == 1:          ### 说明当前的这个基因他只有一个父亲！  那就直接1拿他父亲的那个GSEA分数作为当前1要加的这个分数
            CorFather = NewRelationDict[ele][0]       ### 读取当前的这一元素它所对应的父亲层的元素都有谁！    一般来说他的父亲层的节点只能有一个，因此取首位[0]
            GseaScore.append(GSEADict[CorFather])         ### 将他对应的父节点元素的分数值取过来！放到当前节点对应的位置处！   保持顺序不乱！
        else:
            CorFathers = NewRelationDict[ele]            ### 现在他父亲是由多个，现在这个CorFathers他是一个数组！
            FathersScore = []         ### 他所有父亲的GSEA分数
            for nowFather in CorFathers:               ### 来读取他每个父亲的分数
                FathersScore.append(GSEADict[nowFather])         ### 将当前父亲分数加进去！
            # FinalScore = np.average(FathersScore)         ### 最终要加的分数是所有父亲分数的平均值！       其实这个位置再算的时候应该加上的是加权平均值才对呀！毕竟他输出到1各个父亲节点的数值也不同！
            FinalScore = np.sum(FathersScore)
            GseaScore.append(FinalScore)                    ### 将他对应的父节点元素的分数值取过来！放到当前节点对应的位置处！   保持顺序不乱！

    # print("现在GseaScore就是当前层输出值需要加上的下一层的GSEA分数，现在他的长度已经具体内容是！", len(GseaScore), GseaScore)

    NoisyFlag = False
    if NoisyFlag:                  ### 现在这个就是说考虑一个加知识的，但是所加的只是全都是随机噪音，不是正确的知识！
        GseaScore = []
        print("加油文超！周二之前实验部分写完！！测试一下现在加噪音这块走进来没！")
        for i in range(len(Gene)):  ### 逐个，按顺序的来读取各个基因（子通路）！在这，这个顺序不能乱！
            GseaScore.append(random.uniform(-0.1, 0.1))               ### 现在在这来指定这个基因的取值范围，它的取值范围是需要与所加知识的归一化范围保持一致！





    ### 现在就需要将获取到的GSEA分数值给加到对应的outcome，因为是在网络的内部，因此需要定义Lambda函数
    def constant(outcome):
        zero_array = np.array(GseaScore)
        zero_array = np.reshape(zero_array, (-1, len(zero_array)))
        import tensorflow as tf
        weights = tf.constant(zero_array, dtype='float32')
        # input = tf.keras.backend.variable(tf.convert_to_tensor(input_tensor))
        tf.squeeze(weights, 0)  ### 先删掉他的第二维
        print("现在最开始设定的这个数据的维度是怎样的！", len(zero_array), weights.shape, weights)
        tf.expand_dims(weights, 0)  ### 再为他增加第一维
        print("当前tile函数之前这个数据的维度是怎样的！", weights.shape, weights)
        weights = tf.tile(weights, [tf.shape(outcome)[0],
                                    1])  ### 此时就说明将weights这个数据分别在各个维度上进行扩充，其中第一维扩充的倍数为tf.shape(outcome)[0]，第二维扩充的倍数为1（相当于在这第二维不变）

        ### 正常情况下到这就可以停了，但是现在想要一步到位，在获取了当前层的注意力权重之后，直接与当前层的输出结果相加，得到当前这一层的GSEA指导后的分数值
        attention_probs = weights
        ### 注意！！注意！！下面的这步操作极为关键！必须得有这一步，否则的话使用SHAP可解释性方法处理这种加知识的操作就将会因为数据reshape问题而报错！！
        ### 因为目前所传过来的这个outcome其实是一个list的形式，是[outcome]这种格式，因此说这个函数最终生成的数据形式为(1, ?, 1387)， 可是实际应该需要的是(?, 1387)这个样子！！
        ### 这种操作在平时看着可能没啥，也确实不会出错！但是如果使用SHAP这种可解释方法的时候，那就会报错了，因为在SHAP方法的 shap.DeepExplainer((x, y), map2layer(model, X.copy(), layer_name))  这句代码中，他是需要重新构建一下各个网络层的输出，那这样的新构建的输入输出数据是要求二维而不是三维，因此就会发生报错！
        ### 所以说要进行下面的这个操作！取消他的列表形式！将三维数据转化为二维！
        if type(outcome) == list:
            outcome = outcome[0]
        print("现在处于对通路层加知识！这两个数据的形状是怎样的！", type(attention_probs), type(outcome), outcome, attention_probs)
        outcome = tf.add(outcome, attention_probs)
        return outcome                     ### 最终还是需要保证数据的取值范围为（-1，1）

    ## 下面这块是来引入GSEA这个知识的！！
    outcome = Lambda(constant)([outcome])
    outcome = tf.convert_to_tensor(outcome)
    print("现在是处于通路层！最终经过融合之后的当前这一层的输出是什么！", outcome)
    return outcome









### 现在这个是对某一层网络层的输出值中添加属于当前层网络神经元的GSEA值！   现在这个mapp表示当前层与下一层之间的映射关系！outcome表示当前层的输出！
### 最终返回的是当前这一层网络输出经过融合处理之后的那个输出！！   他与上面的那个AddGSEAScore()函数不同，那个是将下一层网络的GSEA值拿过来进行处理，而这个是将当前层的GSEA值拿过来进行处理！
def AddGSEAScoreAfter(mapp, outcome, j):
    Gene = np.array(mapp.index)       ### 这个指代的是当前层网络中各个神经元所指代的生物学含义！   就是上一层！
    Pathway = np.array(mapp.columns)       ### 这个指代的是下一层网络中各个神经元所指代的生物学含义！    这两行其实主要是为了获取当前这层网络内部各个元素在此时他的分布顺序！

    ### 下面来获取当前这一层中神经元所代表的基因/通路他们所对应的GSEA数值！以及当前层中各个神经元节点所包含的上一层的神经元节点！
    LayerGSEA = LayerEleGSEA()
    LayerRelationship = LayerEleRelationship()
    if j == 'Gene':              ### 此时代表当前的这个是基因层！
        GSEADict = LayerGSEA.getLayer0()            ### 获取当前这一层中各个神经元所对应的GSEA分数值！
    elif j == 0:                 ### 此时代表第一层通路层！
        GSEADict = LayerGSEA.getLayer1()
        print("此时这个通路第一层有没有走到！", len(GSEADict))
        Gene = Pathway                ### 注意看当前传进来的这个mapp，他的行是上一层网络的元素，列是当前网络的元素！  因此除了基因层，剩下的通路层都是需要选用列这里的各个元素的！
    elif j == 1:
        GSEADict = LayerGSEA.getLayer2()
        Gene = Pathway
    elif j == 2:
        GSEADict = LayerGSEA.getLayer3()
        Gene = Pathway
    elif j == 3:
        GSEADict = LayerGSEA.getLayer4()
        Gene = Pathway
    elif j == 4:
        GSEADict = LayerGSEA.getLayer5()
        Gene = Pathway
    else:
        print("说明此时出错了！索引变量j超过了一定的范围！", j)


    ### 现在开始构造当前这个输出值所对应的GSEA分数！   因为现在这个输出值他是对应于当前层网络的输出！因此在这是要将当前层的GSEA值融合到这个输出中！   要用Gene
    GseaScore = []
    for ele in Gene:                  ### 逐个，按顺序的来读取各个基因（子通路）！在这，这个顺序不能乱！
        if ele not in GSEADict:         ### 此时，这个就说明当前这个神经元是没有对应的GSEA分数的，按理来说是不应该的！这个位置不应该出现的！
            GseaScore.append(0)
            print("按理来说这个位置是不应该的，找不到这个神经元所对应的GSEA分数！", ele)
        else:
            GseaScore.append(GSEADict[ele])               ### 将当前这个神经元节点对应的分数值取过来！放到当前节点对应的位置处！   保持顺序不乱！


    # print("现在GseaScore就是当前层输出值需要加上的下一层的GSEA分数，现在他的长度已经具体内容是！", len(GseaScore), GseaScore)

    ### 现在就需要将获取到的GSEA分数值给加到对应的outcome，因为是在网络的内部，因此需要定义Lambda函数
    def constant(outcome):
        zero_array = np.array(GseaScore)
        zero_array = np.reshape(zero_array, (-1, len(zero_array)))
        weights = tf.constant(zero_array, dtype='float32')
        # input = tf.keras.backend.variable(tf.convert_to_tensor(input_tensor))
        tf.squeeze(weights, 0)  ### 先删掉他的第二维
        print("现在最开始设定的这个数据的维度是怎样的！", len(zero_array), weights.shape, weights)
        tf.expand_dims(weights, 0)  ### 再为他增加第一维
        print("当前tile函数之前这个数据的维度是怎样的！", weights.shape, weights)
        weights = tf.tile(weights, [tf.shape(outcome)[0],
                                    1])  ### 此时就说明将weights这个数据分别在各个维度上进行扩充，其中第一维扩充的倍数为tf.shape(outcome)[0]，第二维扩充的倍数为1（相当于在这第二维不变）

        ### 正常情况下到这就可以停了，但是现在想要一步到位，在获取了当前层的注意力权重之后，直接与当前层的输出结果相加，得到当前这一层的GSEA指导后的分数值
        attention_probs = weights
        outcome = tf.add(outcome, attention_probs)
        return outcome                     ### 最终还是需要保证数据的取值范围为（-1，1）

    ## 下面这块是来引入GSEA这个知识的！！
    outcome = Lambda(constant)([outcome])
    outcome = tf.convert_to_tensor(outcome)
    print("最终经过融合之后的当前这一层的输出是什么！", outcome)
    return outcome


def RegSqureGSEAScoreAfter(mapp, outcome, j):
    Gene = np.array(mapp.index)       ### 这个指代的是当前层网络中各个神经元所指代的生物学含义！   就是上一层！
    Pathway = np.array(mapp.columns)       ### 这个指代的是下一层网络中各个神经元所指代的生物学含义！    这两行其实主要是为了获取当前这层网络内部各个元素在此时他的分布顺序！

    print("现在的这个输入数据的情况！！", type(outcome), outcome)
    ### 下面来获取当前这一层中神经元所代表的基因/通路他们所对应的GSEA数值！以及当前层中各个神经元节点所包含的上一层的神经元节点！
    LayerGSEA = LayerEleGSEA()
    LayerRelationship = LayerEleRelationship()
    if j == 'Gene':              ### 此时代表当前的这个是基因层！
        GSEADict = LayerGSEA.getLayer0()            ### 获取当前这一层中各个神经元所对应的GSEA分数值！
    elif j == 0:                 ### 此时代表第一层通路层！
        GSEADict = LayerGSEA.getLayer1()
        print("此时这个通路第一层有没有走到！", len(GSEADict))
        Gene = Pathway                ### 注意看当前传进来的这个mapp，他的行是上一层网络的元素，列是当前网络的元素！  因此除了基因层，剩下的通路层都是需要选用列这里的各个元素的！
    elif j == 1:
        GSEADict = LayerGSEA.getLayer2()
        Gene = Pathway
    elif j == 2:
        GSEADict = LayerGSEA.getLayer3()
        Gene = Pathway
    elif j == 3:
        GSEADict = LayerGSEA.getLayer4()
        Gene = Pathway
    elif j == 4:
        GSEADict = LayerGSEA.getLayer5()
        Gene = Pathway
    else:
        print("说明此时出错了！索引变量j超过了一定的范围！", j)

    ### 现在开始构造当前这个输出值所对应的GSEA分数！   因为现在这个输出值他是对应于当前层网络的输出！因此在这是要将当前层的GSEA值融合到这个输出中！   要用Gene
    GseaScore = []
    for ele in Gene:                  ### 逐个，按顺序的来读取各个基因（子通路）！在这，这个顺序不能乱！
        if ele not in GSEADict:         ### 此时，这个就说明当前这个神经元是没有对应的GSEA分数的，按理来说是不应该的！这个位置不应该出现的！
            GseaScore.append(0)
            print("按理来说这个位置是不应该的，找不到这个神经元所对应的GSEA分数！", ele)
        else:
            GseaScore.append(GSEADict[ele])               ### 将当前这个神经元节点对应的分数值取过来！放到当前节点对应的位置处！   保持顺序不乱！

    ### 现在就需要将获取到的GSEA分数值给加到对应的outcome，因为是在网络的内部，因此需要定义Lambda函数
    def constant(outcome):
        ### 先想办法将tf类型转化为float类型
        # 将tf类型的数据转换为float类型
        tf_data = outcome.numpy().astype(float)
        ### 两组数据还需要进行归一化，到1指定的数据范围！
        ### 这个主要是对传入的GSEA分数值（在这其实就是原始的GSEA分数）进行归一化操作，指定最终归一化到的范围
        def OriGSEADataProcess_Normilse(Data, MIN, MAX):
            NorData = []
            Value = Data
            d_min = min(Value)  # 当前数据最大值
            d_max = max(Value)  # 当前数据最小值
            for ele in Data:  ### 抽取当前网络层中的各个元素！
                newdata = MIN + ((MAX - MIN) / (d_max - d_min)) * (ele - d_min)  ### 逐个元素的进行归一化到指定的范围内！
                NorData.append(newdata)
            return NorData

        tf_data = OriGSEADataProcess_Normilse(tf_data, 0, 0.1)
        GseaScore = OriGSEADataProcess_Normilse(GseaScore, 0, 0.1)
        # 计算差异
        tf_data = np.array(tf_data)
        GseaScore = np.array(GseaScore)
        diff = tf_data - GseaScore
        squared_diff = np.square(diff)
        # 计算平均差异
        mean_squared_diff = np.mean(squared_diff)
        return mean_squared_diff                     ### 最终还是需要保证数据的取值范围为（-1，1）

    def constantTwo(outcome):
        zero_array = np.array(GseaScore)
        zero_array = np.reshape(zero_array, (-1, len(zero_array)))
        weights = tf.constant(zero_array, dtype='float32')
        # input = tf.keras.backend.variable(tf.convert_to_tensor(input_tensor))
        tf.squeeze(weights, 0)  ### 先删掉他的第二维
        print("现在最开始设定的这个数据的维度是怎样的！", len(zero_array), weights.shape, weights)
        tf.expand_dims(weights, 0)  ### 再为他增加第一维
        print("当前tile函数之前这个数据的维度是怎样的！", weights.shape, weights)
        weights = tf.tile(weights, [tf.shape(outcome)[0],
                                    1])  ### 此时就说明将weights这个数据分别在各个维度上进行扩充，其中第一维扩充的倍数为tf.shape(outcome)[0]，第二维扩充的倍数为1（相当于在这第二维不变）
        # 找到数据的最小值和最大值
        min_value = tf.minimum(tf.reduce_min(weights), tf.reduce_min(outcome))
        max_value = tf.maximum(tf.reduce_max(weights), tf.reduce_max(outcome))
        # 归一化数据
        normalized_a = (weights - min_value) / (max_value - min_value)
        normalized_b = (outcome - min_value) / (max_value - min_value)
        # 计算平均方差
        diff = normalized_a - normalized_b
        squared_diff = tf.square(diff)
        mean_squared_diff = tf.reduce_mean(squared_diff)
        return mean_squared_diff

    ## 下面这块是来引入GSEA这个知识的！！
    Diff = Lambda(constantTwo)([outcome])
    print("最终经过融合之后的当前这一层的输出是什么！", Diff)
    return Diff





### 现在这个是只对基因层的输入数据中添加属于当前层网络神经元的GSEA值！   现在这个geneName表示当前这个基因层中他的输入数据所对应的各个基因名字！outcome表示当前层的输入！
### 最终返回的是当前这一层网络输入数据经过融合处理之后的那个输入！！   他是只对基因层进行处理！
def AddGSEAScoreGeneBefore(geneName, outcome):
    print("现在这个是对基因的输入数据进行知识迁移！！")
    ### 下面来获取当前这一层中神经元所代表的基因/通路他们所对应的GSEA数值！以及当前层中各个神经元节点所包含的上一层的神经元节点！     在这单指基因层中的各个基因！
    LayerGSEA = LayerEleGSEA()
    GSEADict = LayerGSEA.getLayer0()  ### 获取当前这一层中各个神经元所对应的GSEA分数值！


    ### 现在开始构造当前这个输出值所对应的GSEA分数！   因为现在这个输出值他是对应于当前层网络的输出！因此在这是要将当前层的GSEA值融合到这个输出中！   要用Gene
    GseaScore = []
    for ele in geneName:                  ### 逐个，按顺序的来读取各个基因（子通路）！在这，这个顺序不能乱！
        if ele not in GSEADict:         ### 此时，这个就说明当前这个神经元是没有对应的GSEA分数的，按理来说是不应该的！这个位置不应该出现的！
            GseaScore.append(0)
            print("现在是在基因层的输入数据中来添加GSEA分数的！按理来说这个位置是不应该的，找不到这个神经元所对应的GSEA分数！", ele)
        else:
            GseaScore.append(GSEADict[ele])               ### 将当前这个神经元节点对应的分数值取过来！放到当前节点对应的位置处！   保持顺序不乱！


    NoisyFlag = False
    if NoisyFlag:                  ### 现在这个就是说考虑一个加知识的，但是所加的只是全都是随机噪音，不是正确的知识！
        GseaScore = []
        for i in range(len(geneName)):  ### 逐个，按顺序的来读取各个基因（子通路）！在这，这个顺序不能乱！
            GseaScore.append(random.uniform(-0.1, 0.1))               ### 现在在这来指定这个基因的取值范围，它的取值范围是需要与所加知识的归一化范围保持一致！


    ### 现在就需要将获取到的GSEA分数值给加到对应的outcome，因为是在网络的内部，因此需要定义Lambda函数
    def constant(outcome):
        zero_array = np.array(GseaScore)
        zero_array = np.reshape(zero_array, (-1, len(zero_array)))
        import tensorflow as tf
        weights = tf.constant(zero_array, dtype='float32')
        # input = tf.keras.backend.variable(tf.convert_to_tensor(input_tensor))
        tf.squeeze(weights, 0)  ### 先删掉他的第二维
        print("现在最开始设定的这个数据的维度是怎样的！", len(zero_array), weights.shape, weights)
        tf.expand_dims(weights, 0)  ### 再为他增加第一维
        print("当前tile函数之前这个数据的维度是怎样的！", weights.shape, weights, outcome, tf.shape(outcome)[0])
        weights = tf.tile(weights, [tf.shape(outcome)[0],
                                    1])  ### 此时就说明将weights这个数据分别在各个维度上进行扩充，其中第一维扩充的倍数为tf.shape(outcome)[0]，第二维扩充的倍数为1（相当于在这第二维不变）

        ### 正常情况下到这就可以停了，但是现在想要一步到位，在获取了当前层的注意力权重之后，直接与当前层的输出结果相加，得到当前这一层的GSEA指导后的分数值
        attention_probs = weights
        ### 注意！！注意！！下面的这步操作极为关键！必须得有这一步，否则的话使用SHAP可解释性方法处理这种加知识的操作就将会因为数据reshape问题而报错！！
        ### 因为目前所传过来的这个outcome其实是一个list的形式，是[outcome]这种格式，因此说这个函数最终生成的数据形式为(1, ?, 27687)， 可是实际应该需要的是(?, 27687)这个样子！！
        ### 这种操作在平时看着可能没啥，也确实不会出错！但是如果使用SHAP这种可解释方法的时候，那就会报错了，因为在SHAP方法的 shap.DeepExplainer((x, y), map2layer(model, X.copy(), layer_name))  这句代码中，他是需要重新构建一下各个网络层的输出，那这样的新构建的输入输出数据是要求二维而不是三维，因此就会发生报错！
        ### 所以说要进行下面的这个操作！取消他的列表形式！将三维数据转化为二维！
        if type(outcome) == list:
            outcome = outcome[0]
        print("现在处于对基因层加知识！这两个数据的形状是怎样的！", type(attention_probs), type(outcome), outcome, attention_probs)
        outcome = tf.add(outcome, attention_probs)
        return outcome                     ### 最终还是需要保证数据的取值范围为（-1，1）

    ## 下面这块是来引入GSEA这个知识的！！
    outcome = Lambda(constant)([outcome])
    outcome = tf.convert_to_tensor(outcome)
    print("最终经过融合之后的当前这一层的输出是什么！", outcome)
    return outcome





### 下面这部分来实现相应神经元消融实验  可以考虑从ModelParam 这个文件中来给定各个神经元他们各自所对应的分数（这个分数其实就只有0、1两种取值，要消的哪些取值为零，留的那些取值为1）   之后在当前的这个函数当中来跟各个位置对应的分数取值相乘！
def XiaoRong(mapp, outcome, j):
    Gene = np.array(mapp.index)       ### 这个指代的是当前层网络中各个神经元所指代的生物学含义！   就是上一层！
    Pathway = np.array(mapp.columns)       ### 这个指代的是下一层网络中各个神经元所指代的生物学含义！    这两行其实主要是为了获取当前这层网络内部各个元素在此时他的分布顺序！
    print("现在这个outcome数据是怎样的！", outcome.shape, outcome)

    ### 下面来获取当前这一层中神经元所代表的基因/通路他们所对应的消融数值（不是0就是1）！这块就决定了那些神经元要保留，哪些神经元是要消掉的！
    LayerGSEA = LayerEleGSEA()
    if j == 'Gene':              ### 此时代表当前的这个是基因层！
        GSEADict = LayerGSEA.getLayer0()            ### 获取当前这一层中各个神经元所对应的GSEA分数值！
    elif j == 0:                 ### 此时代表第一层通路层！
        GSEADict = LayerGSEA.getLayer1()
        print("此时这个通路第一层有没有走到！", len(GSEADict))
        Gene = Pathway                ### 注意看当前传进来的这个mapp，他的行是上一层网络的元素，列是当前网络的元素！  因此除了基因层，剩下的通路层都是需要选用列这里的各个元素的！
    elif j == 1:
        GSEADict = LayerGSEA.getLayer2()
        Gene = Pathway
    elif j == 2:
        GSEADict = LayerGSEA.getLayer3()
        Gene = Pathway
    elif j == 3:
        GSEADict = LayerGSEA.getLayer4()
        Gene = Pathway
    elif j == 4:
        GSEADict = LayerGSEA.getLayer5()
        Gene = Pathway
    else:
        print("说明此时出错了！索引变量j超过了一定的范围！", j)


    ### 现在开始构造当前这个输出值所对应的GSEA分数！   因为现在这个输出值他是对应于当前层网络的输出！因此在这是要将当前层的GSEA值融合到这个输出中！   要用Gene       只不过就是现在的这个GSEA分数现在不是0就是1，用以决定当前的这个神经元是否要进行保留！！
    GseaScore = []
    for ele in Gene:                  ### 逐个，按顺序的来读取各个基因（子通路）！在这，这个顺序不能乱！
        if ele not in GSEADict:         ### 此时，这个就说明当前这个神经元是没有对应的GSEA分数的，按理来说是不应该的！这个位置不应该出现的！
            GseaScore.append(1)              ### 现在这个地方跟前面融入知识的不太一样！
            print("按理来说这个位置是不应该的，找不到这个神经元所对应的GSEA分数！", ele)
        else:
            GseaScore.append(GSEADict[ele])               ### 将当前这个神经元节点对应的分数值取过来！放到当前节点对应的位置处！   保持顺序不乱！


    # print("现在GseaScore就是当前层输出值需要加上的下一层的GSEA分数，现在他的长度已经具体内容是！", len(GseaScore), GseaScore)

    ### 现在就需要将获取到的GSEA分数值给加到对应的outcome，因为是在网络的内部，因此需要定义Lambda函数
    ### 现在就需要将获取到的GSEA分数值给加到对应的outcome，因为是在网络的内部，因此需要定义Lambda函数
    def constant(outcome):
        zero_array = np.array(GseaScore)
        zero_array = np.reshape(zero_array, (-1, len(zero_array)))
        weights = tf.constant(zero_array, dtype='float32')
        if type(outcome) == list:
            outcome = outcome[0]
            print("目前是来消融的，现在取的这个数据的形状是怎样的！", tf.shape(outcome), tf.shape(outcome)[0])
        # input = tf.keras.backend.variable(tf.convert_to_tensor(input_tensor))
        tf.squeeze(weights, 0)  ### 先删掉他的第二维
        print("目前是来消融的，现在最开始设定的这个数据的维度是怎样的！", len(zero_array), weights.shape, weights, outcome)
        tf.expand_dims(weights, 0)  ### 再为他增加第一维
        print("目前是来消融的，当前tile函数之前这个数据的维度是怎样的！", weights.shape, weights, outcome, tf.shape(outcome)[0], outcome.shape)
        weights = tf.tile(weights, [tf.shape(outcome)[0],
                                    1])  ### 此时就说明将weights这个数据分别在各个维度上进行扩充，其中第一维扩充的倍数为tf.shape(outcome)[0]，第二维扩充的倍数为1（相当于在这第二维不变）

        ### 正常情况下到这就可以停了，但是现在想要一步到位，在获取了当前层的注意力权重之后，直接与当前层的输出结果相加，得到当前这一层的GSEA指导后的分数值
        attention_probs = weights
        print("目前是来消融的，现在这两个数据的形状是怎样的！", type(attention_probs), type(outcome), outcome, attention_probs)
        # outcome = tf.matmul(outcome, attention_probs)
        outcome = outcome * attention_probs
        return outcome                     ### 最终还是需要保证数据的取值范围为（-1，1）


        ## 下面这块是来引入GSEA这个知识的！！
    outcome = Lambda(constant)([outcome])
    outcome = tf.convert_to_tensor(outcome)
    print("最终经过融合之后的当前这一层的输出是什么！", outcome)
    return outcome




### 用这个网络层来为基因层添加GSEA分数！
class GeneGSEALayer(Layer):
    def __init__(self, name, **kwargs):
        super(GeneGSEALayer, self).__init__(**kwargs)
        self.name = name

    def call(self, inputs):
        # 自定义运算
        ##  先读取一下当前这个基因层中各个基因数据的排布情况！
        df = pd.read_csv('../CeShi/OtherTest/Other/GeneColsNames.csv')
        geneData = df.iloc[:, 0]  # 取第一列数据
        # print("当前第一列的基因数据情况是怎样的！", type(geneData), len(geneData), geneData)
        geneData = np.array(geneData)
        output = AddGSEAScoreGeneBefore(geneData, inputs)
        return output


### 用这个网络层来为通路层添加GSEA分数！
class PathwayGSEALayer(Layer):
    def __init__(self, name, OriMapp, j, **kwargs):
        super(PathwayGSEALayer, self).__init__(**kwargs)
        self.name = name
        self.OriMapp = OriMapp
        self.j = j

    def call(self, inputs):
        # 自定义运算
        nowoutput = AddGSEAScore(self.OriMapp, inputs, self.j)         ## AddGSEAScore(OriMapp, outcome, j)
        return nowoutput




###  自定义层   这个是又自定义的一层网络，用于对网络的输出结果继续进行加权！
class WeightedOutput(Layer):
    def __init__(self, **kwargs):
        super(WeightedOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1],),
                                      initializer='ones',
                                      trainable=True)
        super(WeightedOutput, self).build(input_shape)

    def call(self, inputs):
        gradients = tf.gradients(self.loss, self.kernel)[0]
        gradients /= (K.sqrt(K.mean(K.square(gradients))) + K.epsilon())  ### 将梯度值进行处理，进行归一化操作！
        grad_values_normalise = K.relu(gradients)
        weighted_output = tf.multiply(inputs, grad_values_normalise)
        return weighted_output





def get_pnet(inputs, features, genes, n_hidden_layers, direction, activation, activation_decision, w_reg,
             w_reg_outcomes, dropout, sparse, add_unk_genes, batch_normal, kernel_initializer, use_bias=False,
             shuffle_genes=False, attention=False, dropout_testing=False, non_neg=False, sparse_first_layer=True, WeightOutputLayers=None, gradients=None, gradients_Flag=False):
    feature_names = {}
    n_features = len(features)
    n_genes = len(genes)
    print("目前来构建网络，他的基因以及输入数据的情况是怎样的！", inputs.shape, n_features, n_genes)
    if gradients_Flag == False:
        print("现在还没办法通过梯度来进行修改！", gradients)
    else:
        print("这可真的是上天护佑啊，那么此时的这个梯度值是谁呢！", gradients)

    gradients = 'All'                              ### AllPath   All    NoKno         OnlyGene   Path1
    AblatNeurFlag = True       ### 这个来表示是否对各层网络中的神经元进行消融！
    AblatRate = 0.01               ### 这个表示对网络层进行消融的时候，每一层消融多少个神经元
    AblatNum = False        ### 这个是表示各层按照给定数目进行删除  此时上面的那个消融比例也就失效了！


    FCNFlag = False                   ### 这个标志意味着是否令当前的这个代码作为FCN算法的代码   网络的基本架构不变，主要就是各层之间的连接关系是随便练的！
    # shuffle_genes = 'all'  ### 现在就是让这个最开始的基因层连接随机的进行连！   现在这个标志就是打乱基因层的连接！   但是这样的话它的运行速度会极慢！

    RegLoss = None




    print("现在是在 model/builders/builders_utils.py 文件中，未修改之前的这个激活函数的情况是谁！！", activation)
    # activation = LeakyReLU(alpha=0.05)
    # activation = 'sigmoid'

    if not type(w_reg) == list:
        w_reg = [w_reg] * 10

    if not type(w_reg_outcomes) == list:
        w_reg_outcomes = [w_reg_outcomes] * 10

    if not type(dropout) == list:
        dropout = [w_reg_outcomes] * 10

    w_reg0 = w_reg[0]
    w_reg_outcome0 = w_reg_outcomes[0]
    w_reg_outcome1 = w_reg_outcomes[1]
    # w_reg_outcome1 = w_reg_outcomes[0]
    reg_l = l2
    constraints = {}
    if non_neg:
        from keras.constraints import nonneg
        constraints = {'kernel_constraint': nonneg()}
        # constraints= {'kernel_constraint': nonneg(), 'bias_constraint':nonneg() }
    if sparse:

        if shuffle_genes == 'all':
            ones_ratio = float(n_features) / np.prod([n_genes, n_features])                ###  这个np.prod([n_genes, n_features])就表示 让n_genes与 n_features彼此相乘    那么现在这个 ones_ratio就表示当前这一层中需要连接的那些边占总边数的比例
            logging.info('ones_ratio random {}'.format(ones_ratio))
            mapp = np.random.choice([0, 1], size=[n_features, n_genes], p=[1 - ones_ratio, ones_ratio])            ### 以1 - ones_ratio的概率选取不连接的边，以ones_ratio的概率选取连接的边！
            print("现在这个基因层随机选取的结果为！", type(mapp), mapp)
            layer1 = SparseTF(n_genes, mapp, activation=activation, W_regularizer=reg_l(w_reg0),             ### 接下来第一层算是一个基因层，他连接的是输入特征与基因，他是需要全连
                              name='h{}'.format(0), kernel_initializer=kernel_initializer, use_bias=use_bias,
                              **constraints)
            # layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg),
            #           use_bias=use_bias, name='h0', kernel_initializer= kernel_initializer )
        else:
            print("此时是在model/builders/builders_utils.py文件中，目前的这个shuffle_genes是谁", shuffle_genes)       ### 目前这个基因层走的是这里！！
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                              use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)               ### 在这，首个基因层选用Diagonal 这个类来建立


    else:
        if sparse_first_layer:
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                              use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)
        else:
            layer1 = Dense(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                           use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer)

    IFGeneBefore = False
    ### 现在在数据流向基因层之前，先对他们进行GSEA知识迁移
    if gradients == 'All' or gradients == 'OnlyGene':
        IFGeneBefore = True
    if IFGeneBefore:
        ##  先读取一下当前这个基因层中各个基因数据的排布情况！
        df = pd.read_csv('../CeShi/OtherTest/Other/GeneColsNames.csv')
        geneData = df.iloc[:, 0]  # 取第一列数据
        # print("当前第一列的基因数据情况是怎样的！", type(geneData), len(geneData), geneData)
        geneData = np.array(geneData)
        # print("现在这个基因数据情况是怎样的！", len(geneData), geneData)
        ##  现在将读取的数据扔进去，获取对应的GSEA分数，并将这个GSEA分数知识迁移到对应的基因输入数据中！
        inputs = AddGSEAScoreGeneBefore(geneData, inputs)

        # inputs = GeneGSEALayer(name='GeneGsea')(inputs)     ### 这个是将加知识的那个操作完全的封装到一个网络层中！







    outcome = layer1(inputs)          ## 现在来得到第一层神经网络的输出        这个输出值必须得提前先得到，即使后面要来进行梯度加权！  因为算梯度的时候，必须知道这个位置的输入与输出值！


    print("算一下此时的这个梯度值的情况是怎样的！", K.gradients(outcome, inputs), K.gradients(outcome, inputs)==None)



    decision_outcomes = []


    decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(0), W_regularizer=reg_l(w_reg_outcome0))(inputs)        ## 构建一个全连接网络，他的输入就是inputs  并获得当前层的输出   这一层只有一个神经元，就代表输出结果只有一个数据！      这个是输入数据（还没经过模型的那个数据）的决策输出值！

    # testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)             ### 对最终的决策输出再加上一个批归一化！

    # decision_outcome = Activation( activation=activation_decision, name='o{}'.format(0))(decision_outcome)

    # first outcome layer
    # decision_outcomes.append(decision_outcome)

    # if reg_outcomes:
    # decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(1), W_regularizer=reg_l(w_reg_outcome1/2.), **constraints)(outcome)
    decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(1),
                             W_regularizer=reg_l(w_reg_outcome1 / 2.))(outcome)         ### 这个是注意力层输出数据的决策输出值
    # else:
    #     decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(1))(outcome)

    # drop2 = Dropout(dropout, name='dropout_{}'.format(0))

    IFDropout = False       ### 决定着当前网络内部各层网络的输出结果是否加Dropout机制！
    if IFDropout:         ### 这个是在加完知识之后又加的Dropout
        drop2 = Dropout(dropout[0], name='dropout_{}'.format(0))
        outcome = drop2(outcome, training=dropout_testing)         ### 对输出结果进行Dropout处理，删掉一部分！

    # testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)

    decision_outcome = Activation(activation=activation_decision, name='o{}'.format(1))(decision_outcome)
    decision_outcomes.append(decision_outcome)      ### 在这是将每一层的预测结果进行了汇总！

    if n_hidden_layers > 0:
        maps = get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes)
        ### 目前的这个maps就表示中间的隐藏层他们各层之间的连接关系

        ### 下面这步为强行消掉某些神经元彼此之间的那些连接，从而达到消掉该神经元的作用！
        ### 这个是设计的新的方法来对1其中各个神经元进行消融，其中考虑了三方面的知识来决定哪个神经元消融！

        if AblatNeurFlag:
            print("现在这个字典的数目是多少个！！", len(maps))
            LayerGSEA = LayerEleGSEA()
            ### 下面这个函数主要是对传进来的那个字典进行归一化处理！
            def DictProcess(data_dict):
                # 创建MinMaxScaler对象
                scaler = MinMaxScaler()
                # 对字典进行归一化操作
                scaled_data = scaler.fit_transform(pd.DataFrame(data_dict.values()))
                # print("现在这个归一化后的这个字典形式是怎样的！", type(scaled_data), scaled_data)
                valueData = []
                for ele in scaled_data:
                    valueData.append(ele[0])
                valueData = np.array(valueData)
                # 将归一化后的值更新到字典中
                scaled_dict = dict(zip(data_dict.keys(), valueData))
                return scaled_dict

            ### 下面这个是来读取可解释分数，将其转化成一个字典的形式，并且将读取后的结果进行归一化操作（在这默认是归一化到0~1）
            def getExplainScore(filename):
                # 读取CSV文件
                nowPath = base_path + '/_logs/p1000/pnet/crossvalidation_average_reg_10_tanh/fs-DeepLIFT-Kno/coef_nn_fold_0_layerh' + str(filename) + '.csv'
                df = pd.read_csv(nowPath)
                # 获取两列数据          element,coef
                column1_data = df['element']
                column2_data = df['coef']
                # 创建字典
                data_dict = dict(zip(column1_data, column2_data))

                ### 下面这个是对构建的字典进行归一化操作！
                scaled_dict = DictProcess(data_dict)
                return scaled_dict


            ### 现在这个就是根据元素名字来综合该元素的GSEA分数、可解释分数以及连接分数
            def GetEleScore(sums, filename, GeneXiaoRongNum):
                ### 现在这个sums是一个pandas.core.series.Series类型的数据，现在是需要想办法将其转化为一个dict类型的数据
                # 将pandas.Series对象转换为pandas.DataFrame对象
                df = sums.to_frame().reset_index()
                # 使用to_dict()方法
                ConnectDict = df.set_index('index')[0].to_dict()           ### 现在是根据连接数目得到的分数值数据
                # ConnectDict = sums.to_dict(orient='index')     ### 现在是根据连接数目得到的分数值数据
                ConnectDict = DictProcess(ConnectDict)        ### 下面这个是对连接分数的字典进行归一化操作处理！
                ### 下面需要得到可解释分数以及GSEA分数值对应的元素对应的分数值字典
                ExplainDict = getExplainScore(filename)        ### 获取可解释分数的字典
                if filename == 0:
                    GSEADict = LayerGSEA.getLayer0()  ### 获取当前这一层中各个神经元所对应的GSEA分数值！   现在这个指代的就是基因层！
                elif filename == 1:
                    GSEADict = LayerGSEA.getLayer1()          ### 现在这个是第一层通路层
                elif filename == 2:
                    GSEADict = LayerGSEA.getLayer2()          ### 现在这个是第一层通路层
                elif filename == 3:
                    GSEADict = LayerGSEA.getLayer3()          ### 现在这个是第一层通路层
                elif filename == 4:
                    GSEADict = LayerGSEA.getLayer4()          ### 现在这个是第一层通路层
                elif filename == 5:
                    GSEADict = LayerGSEA.getLayer5()          ### 现在这个是第一层通路层
                GSEADict = DictProcess(GSEADict)        ### 获取GSEA分数的字典并进行归一化处理
                ## 现在开始将三项分数进行综合！得到最终的分数结果！
                FinalScore = {}
                print("现在这个连接字典中的各个元素是谁！", ConnectDict.keys())
                if 'root' in ConnectDict:
                    return None
                for ele in ConnectDict:
                    # if ele not in GSEADict:
                    #     GSEADict[ele] = 0.5
                    FinalScore[ele] = 0.1*ExplainDict[ele] + 0.6*ConnectDict[ele] + 0.3*GSEADict[ele]
                # 使用sorted()函数对value进行排序，并获取前十个元素      现在是来获取那些分数取值最小的那些元素！
                sorted_items = sorted(FinalScore.items(), key=lambda x: x[1])[:GeneXiaoRongNum]
                # 提取前十个元素的key值
                rows_to_clear = [item[0] for item in sorted_items]           ### 获取要消融的元素的名字
                return rows_to_clear


            ### 下面开始对各个掩码矩阵进行消融删除！掩码矩阵一共有六个！第一个的行标题是基因层，列标题是第一层通路层；最后一个矩阵的行标题是第五层通路层，列标题是输出层（只有一个神经元）；当前矩阵的列标题是下一个矩阵的行标题
            ### 在进行消融的时候先按照行标题进行删，之后再按照列标题进行删
            RowTitle = None
            for ij in range(len(maps)):
                nowMap = maps[ij]      ### 这个就是当前的这个映射关系图
                print("先简单观察一下当前这个映射图中的各map之间的关系情况！", type(nowMap), len(list(nowMap.index)), len(list(nowMap.columns)))       ### 现在这个index行标题就是基因的数目，列标题就是对应的通路的数目！

                ### 下面这个是按照行标题进行清零！！！
                if RowTitle is not None:          ### 说明现在不是第一个映射图(就是基因对第一层通路层！)     那么现在主要是来消除那些通路层
                    ## 那么现在需要根据行标题来消除各行数据！
                    # 给定要清零的行标题
                    rows_to_clear = np.array(RowTitle)
                    # # 根据行标题选取对应的行数据
                    # rows_to_clear_data = nowMap.loc[rows_to_clear]
                    # # 将选取的行数据全部清零
                    # rows_to_clear_data.loc[:, :] = 0

                    ### 下面的这个则是根据给定的行标题来将对应的哪一行给删掉！
                    # 删除指定的行  下面的这个删除他是真删除！！整个这一行压根就不存在了！
                    nowMap = nowMap.drop(rows_to_clear)
                    print("当前行标题清零成功，所请的行标题的数目是！", len(rows_to_clear), rows_to_clear)
                else:                     ### 那么现在也就是说想办法将那些无关紧要的基因也给删掉！  现在那个基因的神经元并不删除，只是将其与通路连接的那些边给删掉！    也就是相当于变相的消除了这些基因神经元了，只不过没有具体体现而已！
                    # GeneXiaoRongNum = int(1 * len(list(nowMap.columns)))      ### 基因层这块先全部保留
                    # GeneXiaoRongNum = int(0.1 * len(list(nowMap.columns)))
                    GeneXiaoRongNum = 0        ### 当前基因层是什么都不删！
                    # 计算每行数据的和
                    sums = nowMap.sum(axis=1)
                    filename = 0  ### 现在0代表基因层！
                    rows_to_clear = GetEleScore(sums, filename, GeneXiaoRongNum)       ### 现在这个来获取每个元素的最终分数！
                    # 选择求和值最小的行数据
                    # rows_to_clear = sums.nsmallest(GeneXiaoRongNum).index
                    nowMap.loc[rows_to_clear] = 0              ### 将该基因所在的哪一行（即所对应的与通路的连接）都给删掉！   但是这个神经元还是保持不动得！
                    # print("现在是基因层的行标题消融，这个行标题是谁！", type(sums), sums)


                ### 对于最后一个映射图，只用消行标题，不用消列标题！
                if ij == len(maps)-1:           ### 说明此时选用的是最后一个映射图了！也就是最后一层通路层（28个通路）作为行标题，root作为列标题的操作 那么此时这个循环立马终止不用再进行后面的列标题消融了，只用消当前的这个行标题就行！
                    maps[ij] = nowMap  ### 将处理后的这个映射图再返回回去！使得构建神经网络的时候根据新的映射关系来！！
                    break



                ### 下面这个是按照给定数目来删除各层还是按照指定比例来进行删除！
                if AblatNum:
                    ### 现在前半部分是来按行进行删除的  现在下面是要开始对每个矩阵按列进行删除！  因此ij=0时对应着第一个矩阵，他的列标题就是第一层通路层
                    ## 下面这块是来进行分层消融，一次只消一层！
                    if ij == 0:   ## 现在这个对应第一层通路层
                        XiaoRongNum = int(len(list(nowMap.columns)) - 550)
                    elif ij == 1:
                        XiaoRongNum = int(len(list(nowMap.columns)) - 243)
                    elif ij == 2:
                        XiaoRongNum = int(len(list(nowMap.columns)) - 115)
                    elif ij == 3:
                        XiaoRongNum = int(len(list(nowMap.columns)) - 21)
                    elif ij == 4:
                        XiaoRongNum = int(len(list(nowMap.columns)) - 18)
                    else:
                        print("24.10.17 此时是到最后一个矩阵了！他的列标题对应这最后的那个输出层（只有一个神经元），因此不用处理！")
                else:
                    ### 下面是按照列标题进行清零！！
                    if ij >= 0:  #### 这个是保留了第一层的通路曾！      其中基因层是0，因此如果想要包括基因层那么就要使得ij>=0     也可以通过这个来决定删除后几层的网络！！
                        # XiaoRongNum = int(0.95*len(list(nowMap.columns)))           ### 这个决定着当前下一层网络要消掉多少个神经元！
                        # XiaoRongNum = int(len(list(nowMap.columns))-1)              ### 各个通路层先只保留一个神经元！
                        XiaoRongNum = int(AblatRate * len(list(nowMap.columns)))  ### 这个决定着当前下一层网络要消掉多少个神经元！
                    else:
                        XiaoRongNum = 0
                        XiaoRongNum = int(AblatRate * len(list(nowMap.columns)))  ### 这个决定着当前下一层网络要消掉多少个神经元！




                # if ij == 5:             #### 这个是保留了第一层的通路曾！      其中基因层是0，因此如果想要包括基因层那么就要使得ij>=0     也可以通过这个来决定删除后几层的网络！！
                #     # XiaoRongNum = int(0.95*len(list(nowMap.columns)))           ### 这个决定着当前下一层网络要消掉多少个神经元！
                #     # XiaoRongNum = int(len(list(nowMap.columns))-1)              ### 各个通路层先只保留一个神经元！
                #     XiaoRongNum = int(0.1 * len(list(nowMap.columns)))  ### 这个决定着当前下一层网络要消掉多少个神经元！
                # else:
                #     XiaoRongNum = 0
                #     # XiaoRongNum = int(0.1 * len(list(nowMap.columns)))  ### 这个决定着当前下一层网络要消掉多少个神经元！



                # 计算每一列的数据之和
                sums = nowMap.sum()     ### 这个是对应求列的和！   他现在计算的是每列数据的和！  sum()函数中如果没有明确参数的情况下就是对列求和！
                filename = ij + 1  ### 现在代表各层通路层！  因为现在是开始按照列进行删除了，所以在这ij=0就表示第一层通路层
                # if ij == 5:
                #     filename = ij
                # else:
                #     filename = ij+1      ### 现在代表各层通路层！  因为现在是开始按照列进行删除了，所以在这ij=0就表示第一层通路层
                min_cols = GetEleScore(sums, filename, XiaoRongNum)  ### 现在这个来获取每个元素的最终分数！
                # if min_cols is None:
                #     break
                # # 找出和最小的10列
                # min_cols = sums.nsmallest(XiaoRongNum).index
                # # 将这10列的数据全部清零并返回列标题
                # # nowMap[min_cols] = 0     ### 对应的那一列清零！   这个是对应对列进行清零！

                nowMap = nowMap.drop(min_cols, axis=1)        ### 现在是将对应的那10%的列给删除！  现在是按列进行删除
                RowTitle = min_cols       ### 这个是作为下一个列表的行！
                maps[ij] = nowMap              ### 将处理后的这个映射图再返回回去！使得构建神经网络的时候根据新的映射关系来！！ 他是先删行再删列，到此时这步说明这个矩阵算是删结束了！
                print("当前清零的这一列的标题是谁！", type(sums), type(RowTitle), len(RowTitle))



        OldAblatNeurFlag = False  ### 这个来表示是否对各层网络中的神经元进行消融！    这个是利用旧有的方法来对各层神经网络进行消融
        if OldAblatNeurFlag:
            ### 下面这步为强行消掉某些神经元彼此之间的那些连接，从而达到消掉该神经元的作用！
            RowTitle = None
            for ij in range(len(maps)):
                nowMap = maps[ij]  ### 这个就是当前的这个映射关系图
                print("先简单观察一下当前这个映射图中的各map之间的关系情况！", type(nowMap), len(list(nowMap.index)),
                      len(list(nowMap.columns)))  ### 现在这个index行标题就是基因的数目，列标题就是对应的通路的数目！
                ### 下面这个是按照行标题进行清零！！！
                if RowTitle is not None:  ### 说明现在不是第一个映射图(就是基因对第一层通路层！)     那么现在主要是来消除那些通路层
                    ## 那么现在需要根据行标题来消除各行数据！
                    # 给定要清零的行标题
                    rows_to_clear = np.array(RowTitle)
                    # # 根据行标题选取对应的行数据
                    # rows_to_clear_data = nowMap.loc[rows_to_clear]
                    # # 将选取的行数据全部清零
                    # rows_to_clear_data.loc[:, :] = 0

                    ### 下面的这个则是根据给定的行标题来将对应的哪一行给删掉！
                    # 删除指定的行
                    nowMap = nowMap.drop(rows_to_clear)
                    print("当前行标题清零成功，所请的行标题的数目是！", len(rows_to_clear))
                else:  ### 那么现在也就是说想办法将那些无关紧要的基因也给删掉！  现在那个基因的神经元并不删除，只是将其与通路连接的那些边给删掉！    也就是相当于变相的消除了这些基因神经元了，只不过没有具体体现而已！
                    # GeneXiaoRongNum = int(1 * len(list(nowMap.columns)))      ### 基因层这块先全部保留
                    GeneXiaoRongNum = int(0.2 * len(list(nowMap.columns)))
                    # GeneXiaoRongNum = 0
                    # 计算每行数据的和
                    sums = nowMap.sum(axis=1)
                    # 选择求和值最小的10行数据
                    rows_to_clear = sums.nsmallest(GeneXiaoRongNum).index
                    nowMap.loc[rows_to_clear] = 0  ### 将该基因所在的哪一行（即所对应的与通路的连接）都给删掉！   但是这个神经元还是保持不动得！

                ### 下面是按照列标题进行清零！！
                if ij >= 4:  #### 这个是保留了第一层的通路曾！      其中基因层是0，因此如果想要包括基因层那么就要使得ij>=0     也可以通过这个来决定删除后几层的网络！！
                    # XiaoRongNum = int(0.95*len(list(nowMap.columns)))           ### 这个决定着当前下一层网络要消掉多少个神经元！
                    # XiaoRongNum = int(len(list(nowMap.columns))-1)              ### 各个通路层先只保留一个神经元！
                    XiaoRongNum = int(0.1 * len(list(nowMap.columns)))  ### 这个决定着当前下一层网络要消掉多少个神经元！
                else:
                    XiaoRongNum = 0
                    XiaoRongNum = int(0.1 * len(list(nowMap.columns)))  ### 这个决定着当前下一层网络要消掉多少个神经元！
                # 计算每一列的数据之和
                sums = nowMap.sum()  ### 这个是对应求列的和！
                # print("当前各列求和后的结果是怎样的！", type(sums), sums)
                # 找出和最小的10列
                min_cols = sums.nsmallest(XiaoRongNum).index
                # 将这10列的数据全部清零并返回列标题
                # nowMap[min_cols] = 0     ### 对应的那一列清零！   这个是对应对列进行清零！
                nowMap = nowMap.drop(min_cols, axis=1)  ### 现在是将对应的那10%的列给删除！
                RowTitle = min_cols
                maps[ij] = nowMap  ### 将处理后的这个映射图再返回回去！使得构建神经网络的时候根据新的映射关系来！！
                print("当前清零的这一列的标题是谁！", type(RowTitle), len(RowTitle))




        # print("目前是在model/builders/builders_utils.py文件中，中间各层隐藏层他们彼此之间的连接关系是！", maps)
        layer_inds = list(range(1, len(maps)))
        print('original dropout', dropout)
        print('当前所在的文件是：builders_utils.py， 此时，layer_inds, dropout, w_reg三项分别是谁！', layer_inds, dropout, w_reg)      ### 现在除了首个基因层，后面还剩五个通路层！
        w_regs = w_reg[1:]
        w_reg_outcomes = w_reg_outcomes[1:]
        dropouts = dropout[1:]

        ### 现在来对基因层他的输出结果中添加相应的GSEA分数      这个是在基因层他的后面添加相应的分数
        IFGeneAfter = True               ### 现在是尝试一下在他的后方进行消融
        if IFGeneAfter:
            # outcome = AddGSEAScoreAfter(maps[0], outcome, 'Gene')
            if RegLoss is None:
                RegLoss = RegSqureGSEAScoreAfter(maps[0], outcome, 'Gene')
            else:
                RegLoss = tf.add(RegLoss, RegSqureGSEAScoreAfter(maps[0], outcome, 'Gene'))
            print("现在相加之后的这个损失是谁！", RegLoss)
            # outcome = XiaoRong(maps[0], outcome, 'Gene')                ### 现在这个是用来进行消融的！




        ###---这块为5.11测试所加
        pathwayLength = [1387, 1066, 447, 147, 26]
        j = 0
        IFPathwayBefore = True  ### 这个决定着当前这个通路层要不要在他的输入部分加入相应的GSEA分数！
        ###------
        for i, mapp in enumerate(maps[0:-1]):
            w_reg = w_regs[i]
            w_reg_outcome = w_reg_outcomes[i]
            dropout = dropouts[1]
            names = mapp.index
            # print("当前这一层映射图中，他的索引的名字是谁！", type(mapp), len(names), np.array(names))
            # print("在当前这一层映射图中，他的列索引是谁！", np.array(mapp.columns))
            print("当前是处于pnet_prostate_paper-published_to_zenodo/model/builders/builders_utils.py文件当中，现在的这个映射图的索引情况是怎样的！", len(maps), len(list(maps[i].index)))
            ### 尝试着将这个映射关系保存一下试试！  将他保存到一个测试的csv文件中
            ###  在这一定一定要注意！！他的这个上下层之间的映射关系表每次运行的时候，他的这个映射表都是不一样的！也就是说每次运行的时候，这个神经网络的布局都是不一样的！
            testFilePath = '../CeShi/OtherTest/GenePathMapp/Layer' + str(i) + '.csv'
            mapPD = pd.DataFrame(mapp)
            mapPD.to_csv(testFilePath)
            print("目前是处于pnet_prostate_paper-published_to_zenodo/model/builders/builders_utils.py文件当中，此时这批映射关系数据保存成功了！")

            # names = list(mapp.index)
            OriMapp = mapp        ### 保留一下原始的映射关系图！
            mapp = mapp.values
            if shuffle_genes in ['all', 'pathways']:
                mapp = shuffle_genes_map(mapp)
            n_genes, n_pathways = mapp.shape
            print("此时经过精简处理之后的这个mapp的情况是！", mapp)

            ### 可以这样！直接在这让他的彼此连接关系是随机连接的！  即为想办法将现在mapp里面的数据内容进行随机打乱！
            if FCNFlag:
                ### 可以这样！直接在这让他的彼此连接关系是随机连接的！  即为想办法将现在mapp里面的数据内容进行随机打乱！
                for Imapp in range(len(mapp)):
                    for num in range(3):                  ### 现在来将当前这一行的对应关系进行打乱！  现在是连续打乱三次，加重他的打乱程度！
                        random.shuffle(mapp[Imapp])
                    # random.shuffle(mapp[Imapp])             ### 现在来将当前这一行的对应关系进行打乱！






            logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
            # print 'map # ones {}'.format(np.sum(mapp))
            print('layer {}, dropout  {} w_reg {}'.format(i, dropout, w_reg))
            layer_name = 'h{}'.format(i + 1)
            if sparse:
                ###  下面这个来进行测试一下梯度加权的，先给当前层的梯度权重初始化一个值，这个初始化值可以全部为1（因为是进行相乘的！）
                attentionWeights = np.ones(n_pathways)

                hidden_layer = SparseTF(n_pathways, mapp, activation=activation, W_regularizer=reg_l(w_reg),
                                        name=layer_name, kernel_initializer=kernel_initializer,
                                        use_bias=use_bias, attentionWeights=attentionWeights, **constraints)       #### 构建其中的一个隐藏层！   在这是需要构建一个稀疏的网络层           在这，通路层的数据流向走的是这里！

            else:
                hidden_layer = Dense(n_pathways, activation=activation, W_regularizer=reg_l(w_reg),
                                     name=layer_name, kernel_initializer=kernel_initializer, **constraints)    ### 这个就是照常操作，构建一个全连接的网络层！

            # outcome = hidden_layer(outcome)


            ### 下面，在数据流入当前这层神经网络之前，先将这批数据进行一波处理，向当前这层网络的输入数据中融入当前这层网络的GSEA分数！
            if gradients == 'OnlyGene':
                IFPathwayBefore = False
            if IFPathwayBefore:          ### 现在是只有第一层通路层嵌入知识
                print("现在开始走加GSEA分数的了！！！现在是在通路层的输入部分融入对应的GSEA分数！！")
                if gradients == 'All':
                    if j != 4:           ### 最后一层网络的激活输出值先不要加知识！
                        outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    # outcome = PathwayGSEALayer(name='PathGsea' + str(j), OriMapp=OriMapp, j=j)(outcome)               ## name, OriMapp, j
                elif gradients == 'AllPath':                        #### 现在来处理所有的通路层！
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    # outcome = PathwayGSEALayer(name='PathGsea' + str(j), OriMapp=OriMapp, j=j)(outcome)
                elif gradients == 'Path1' and j == 0:               ### 此时是第一层通路层
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    # outcome = PathwayGSEALayer(name='PathGsea' + str(j), OriMapp=OriMapp, j=j)(outcome)
                    IFPathwayBefore = False
                elif gradients == 'Path2' and j == 1:               ### 此时是第一层通路层
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    # outcome = PathwayGSEALayer(name='PathGsea' + str(j), OriMapp=OriMapp, j=j)(outcome)
                    IFPathwayBefore = False
                elif gradients == 'Path3' and j == 2:               ### 此时是第一层通路层
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    # outcome = PathwayGSEALayer(name='PathGsea' + str(j), OriMapp=OriMapp, j=j)(outcome)
                    IFPathwayBefore = False
                elif gradients == 'Path4' and j == 3:               ### 此时是第一层通路层
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    # outcome = PathwayGSEALayer(name='PathGsea' + str(j), OriMapp=OriMapp, j=j)(outcome)
                    IFPathwayBefore = False
                elif gradients == 'Path5' and j == 4:               ### 此时是第一层通路层   先试试最后一层网络不加只是
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    # outcome = PathwayGSEALayer(name='PathGsea' + str(j), OriMapp=OriMapp, j=j)(outcome)
                    IFPathwayBefore = False
                else:
                    print("当前是第几层通路层以及要消第几层！", j, gradients)


            input_Now = outcome                   ### 当前在 经过隐藏层处理之前应该保留当前隐藏层的输入数据，以便于后续梯度加权的时候进行计算！
            outcome = hidden_layer(input_Now)       ### 在经过上述选择好隐藏层之后，用数据流将当前的这个隐藏层给走了！

            ActFlag = False
            if j == 4 and ActFlag:           ### 此时就在原来的基础之上再加入一个激活网络层！  此时那个激活权重400就针对这个网络层进行操作，不再对28那个通路层了
                outcome = Dense(10, activation='linear', name='o_linearActHidder', W_regularizer=reg_l(w_reg_outcome))(outcome)  ### 这个即为当前的这个通路层的决策输出结果！



            ### 下面这个是来决定是否要将当前层的GSEA分数放到当前层的后面！
            IFAfter = False                          ### 尝试在后方进行消融
            if IFAfter:
                # outcome = AddGSEAScoreAfter(OriMapp, outcome, j)
                if RegLoss is None:
                    RegLoss = RegSqureGSEAScoreAfter(OriMapp, outcome, j)
                else:
                    RegLoss = tf.add(RegLoss, RegSqureGSEAScoreAfter(OriMapp, outcome, j))
                print("现在这个是通路层！相加之后的这个损失是谁！", RegLoss)
                # outcome = XiaoRong(OriMapp, outcome, j)  ### 现在这个是用来进行消融的！




            decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2),
                                     W_regularizer=reg_l(w_reg_outcome))(outcome)                   ### 这个即为当前的这个通路层的决策输出结果！

            # testing
            if batch_normal:
                decision_outcome = BatchNormalization()(decision_outcome)
            decision_outcome = Activation(activation=activation_decision, name='o{}'.format(i + 2))(decision_outcome)
            decision_outcomes.append(decision_outcome)

            if IFDropout and False:         ### 通过这一参数来决定当前这个网络是否要加这个Dropout机制！
                drop2 = Dropout(dropout, name='dropout_{}'.format(i + 1))
                outcome = drop2(outcome, training=dropout_testing)
                print("现在这个通路的Dropout应该是没有走进来！")

            feature_names['h{}'.format(i)] = names
            print("当前第%d层中，他所用的生物元素的数目！", i, len(names))
            j = j + 1
            print("目前所在的文件是：builders_utils.py，目前通路层中走了几个循环了！", j)
            # feature_names.append(names)

        ActAdd = False             #### 在这是在原来网络模型的基础之上在加入那个激活网络！
        if ActAdd:  ### 在这，就直接将激活网络加到他后面
            outcome = Dense(10, activation='linear', name='o_linearActHidder',
                            W_regularizer=reg_l(w_reg_outcome))(outcome)  ### 这个即为当前的这个通路层的决策输出结果！
            decision_outcome = Dense(1, activation='linear', name='o_linearAct',
                                     W_regularizer=reg_l(w_reg_outcome))(outcome)  ### 这个即为当前的这个通路层的决策输出结果！
            # testing
            if batch_normal:
                decision_outcome = BatchNormalization()(decision_outcome)
            decision_outcome = Activation(activation=activation_decision, name='oAct')(decision_outcome)
            decision_outcomes.append(decision_outcome)

        i = len(maps)
        feature_names['h{}'.format(i - 1)] = maps[-1].index
        # print("现在，最后添加进来的这个index是谁！", maps[-1].index)


        # feature_names.append(maps[-1].index)
    # print("最终要进行输出的这个特征名称是谁！", feature_names)        ### 现在我所定义1的这个特征名字即为每一层它所对应的各个生物元素
    print("目前所在的文件是：builders_utils.py  最终要进行输出的输出结果是谁！", outcome)
    print("现在的这个决策输出是谁！", decision_outcomes)
    return outcome, decision_outcomes, feature_names, RegLoss



# from data.pathways.pathway_loader import get_pathway_files
import itertools
import logging

import numpy as np
import pandas as pd

### 下面这块为测试部分
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Dense, Dropout, Activation, BatchNormalization, multiply, Layer
from keras.regularizers import l2
from keras.layers.core import Lambda

import random


# from data.pathways.pathway_loader import get_pathway_files
from data.pathways.reactome import ReactomeNetwork
from model.layers_custom import Diagonal, SparseTF



from keras import backend as K
from keras.layers import Activation
import tensorflow


from CeShi.OtherTest.ModelParamSave.ModelParam_Two import LayerEleGSEA, LayerEleRelationship, AfterLayerEleGSEA
import tensorflow as tf
from keras.layers import LeakyReLU




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
            GseaScore.append(random.uniform(0, 0.01))               ### 现在在这来指定这个基因的取值范围，它的取值范围是需要与所加知识的归一化范围保持一致！



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
        outcome = tf.add(outcome, attention_probs)
        return outcome                     ### 最终还是需要保证数据的取值范围为（-1，1）

    ## 下面这块是来引入GSEA这个知识的！！
    outcome = Lambda(constant)([outcome])
    outcome = tf.convert_to_tensor(outcome)
    print("最终经过融合之后的当前这一层的输出是什么！", outcome)
    return outcome









### 现在这个是对某一层网络层的输出值中添加属于当前层网络神经元的GSEA值！   现在这个mapp表示当前层与下一层之间的映射关系！outcome表示当前层的输出！
### 最终返回的是当前这一层网络输出经过融合处理之后的那个输出！！   他与上面的那个AddGSEAScore()函数不同，那个是将下一层网络的GSEA值拿过来进行处理，而这个是将当前层的GSEA值拿过来进行处理！
def AddGSEAScoreAfter(mapp, outcome, j):
    Gene = np.array(mapp.index)       ### 这个指代的是当前层网络中各个神经元所指代的生物学含义！   就是上一层！
    Pathway = np.array(mapp.columns)       ### 这个指代的是下一层网络中各个神经元所指代的生物学含义！    这两行其实主要是为了获取当前这层网络内部各个元素在此时他的分布顺序！

    ### 下面来获取当前这一层中神经元所代表的基因/通路他们所对应的GSEA数值！以及当前层中各个神经元节点所包含的上一层的神经元节点！
    LayerGSEA = AfterLayerEleGSEA()
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
        outcome = tf.add(outcome, attention_probs)
        # outcome = tf.multiply(outcome, attention_probs)
        return outcome                     ### 最终还是需要保证数据的取值范围为（-1，1）

    ## 下面这块是来引入GSEA这个知识的！！
    outcome = Lambda(constant)([outcome])
    outcome = tf.convert_to_tensor(outcome)
    print("最终经过融合之后的当前这一层的输出是什么！", outcome)
    return outcome




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
            GseaScore.append(random.uniform(0, 0.01))               ### 现在在这来指定这个基因的取值范围，它的取值范围是需要与所加知识的归一化范围保持一致！


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
        print("现在这两个数据的形状是怎样的！", type(attention_probs), type(outcome), outcome,
              attention_probs)
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

    gradients = 'All'                              ### AllPath   All    NoKno       OnlyGene     Path1     After
    IFGseaAfter = True                     ###  True  False 这个决定着当前的这个GSEA分数要不要在各层网络之后添加     这个标志是需要与上面的那个标志混合使用的！看看是只加网络后面还是只加网络前面抑或是网络的前后都加

    FCNFlag = False                   ### 这个标志意味着是否令当前的这个代码作为FCN算法的代码   网络的基本架构不变，主要就是各层之间的连接关系是随便练的！
    # shuffle_genes = 'all'            ### 现在就是让这个最开始的基因层连接随机的进行连！   现在这个标志就是打乱基因层的连接！




    ### 下面这个函数为自定义的一个修改函数，想要在此根据模型的梯度值来修改当前网络层的输出值，梯度值高的就强化，小的就弱化！
    def my_XiuGaiOutPut(inputs, outputs):

        # 计算梯度值         根据输入输出值来计算一下当前这个网络层的梯度情况！
        gradients = K.gradients(outputs, inputs)[0]       ##  这个是计算当前这一层的梯度情况！！
        print("所料不错的话，这的这个gradients应该是空！", outputs, inputs, gradients)
        if gradients == None:
            print("说明此时，当前的这个模型还是处于构建阶段，他传进来的输入输出以及梯度数据都为空", inputs, outputs)
        else:
            print("此时就是说明在实际训练阶段了，现在传进来的输入输出以及梯度值都是有具体数值的，分别为：", inputs, outputs, gradients)
            ### 这个是对当前输入的这个梯度进行归一化！！最后再加上K.epsilon()，是为了防止这个位置为0，要加上一个非常小的值！
            gradients /= (K.sqrt(K.mean(K.square(gradients))) + K.epsilon())
            print("在未经过ReLu处理之前的这个梯度形状是怎样的！", gradients.shape, gradients)
            ### 下面这个位置就是看哪个位置的梯度如果是负值的话，那么这个梯度直接按0进行处理，即当前这个位置的梯度直接清零
            grad_values_normalise = K.relu(gradients)
            # outputs *= grad_values_normalise               ### 这个x是作为当前网络层的输出
            inputs *= grad_values_normalise  ### 这个x是作为当前网络层的输出
        return inputs


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
    reg_l = l2
    constraints = {}
    if non_neg:
        from keras.constraints import nonneg
        constraints = {'kernel_constraint': nonneg()}
        # constraints= {'kernel_constraint': nonneg(), 'bias_constraint':nonneg() }
    if sparse:
        if shuffle_genes == 'all':
            ones_ratio = float(n_features) / np.prod([n_genes, n_features])                ###  这个np.prod([n_genes, n_features])就表示 让n_genes与 n_features彼此相乘
            logging.info('ones_ratio random {}'.format(ones_ratio))
            mapp = np.random.choice([0, 1], size=[n_features, n_genes], p=[1 - ones_ratio, ones_ratio])
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
        AddGSEAScoreGeneBefore(geneData, inputs)




    outcome = layer1(inputs)          ## 现在来得到第一层神经网络的输出        这个输出值必须得提前先得到，即使后面要来进行梯度加权！  因为算梯度的时候，必须知道这个位置的输入与输出值！


    print("算一下此时的这个梯度值的情况是怎样的！", K.gradients(outcome, inputs), K.gradients(outcome, inputs)==None)

    ### 下面尝试进行梯度加权！
    # outcome = WeightedOutput()(outcome)
    # print("现在这个梯度加权成功！！")



    # print("当前所传进来的这个WeightOutputLayers是什么！", WeightOutputLayers)
    # ### 下面这个判断就是说当前指定有第一层基因层来进行梯度加权了！
    # if WeightOutputLayers != None and 0 in WeightOutputLayers:
    #     # ### 其实现在有了第一层网络的输入与输出，完全可以计算当前这个网络的梯度值并随之进行修改网络的输入值！！（注意！！在这修改的是网络的输入值！而不是这层网络的输出！    这个在这里处理的是基因层！！）
    #     inputs = Lambda(lambda x: my_XiuGaiOutPut(*x))([inputs, outcome])           ### 在这必须用Lambda来搭建这个函数，而不能直接来调用这个函数！
    #     print("再修改之后的这个输入数据的情况是怎样的！", type(inputs), inputs)
    #     outcome = layer1(inputs)           ### 因为这个梯度注意力他修改的是输入，因此要将这个输入再次输入到这个网络中来计算输出结果！
    #     print("当前这一网络层的输出值已经被更改了！！")




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

    IFDropout = True       ### 决定着当前网络内部各层网络的输出结果是否加Dropout机制！
    if IFDropout:
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
        # print("目前是在model/builders/builders_utils.py文件中，中间各层隐藏层他们彼此之间的连接关系是！", maps)
        layer_inds = list(range(1, len(maps)))
        print('original dropout', dropout)
        print('当前所在的文件是：builders_utils.py， 此时，layer_inds, dropout, w_reg三项分别是谁！', layer_inds, dropout, w_reg)      ### 现在除了首个基因层，后面还剩五个通路层！
        w_regs = w_reg[1:]
        w_reg_outcomes = w_reg_outcomes[1:]
        dropouts = dropout[1:]

        ### 现在来对基因层他的输出结果中添加相应的GSEA分数      这个是在基因层他的后面添加相应的分数
        ### 现在是尝试一下在他的后方进行加知识！
        if IFGseaAfter:        ### 此标志意味着在网络层的后方来进行加知识
            print("现在是在基因层的后方来添加相应的知识！")
            # outcome = 0.5 * outcome  ### 原始的那个输入数据需要减半，与加入的那个GSEA分数进行配对！
            outcome = AddGSEAScoreAfter(maps[0], outcome, 'Gene')
            # attention_probs = Diagonal(n_genes, input_shape=(n_features,), activation='sigmoid', W_regularizer=l2(w_reg0),
            #                            name='attention')(outcome)  ### 在此加一注意力层  该注意力层的所用激活函数为sigmoid  输出得到各个输出结果的注意力权重
            # outcome = multiply([outcome, attention_probs], name='attention_mul')
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
            print("此时经过精简处理之后的这个mapp的情况是！", type(mapp), mapp)


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
                # outcome = AddGSEAScore(OriMapp, outcome, j)               ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                if gradients == 'All':
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                elif gradients == 'AllPath':                        #### 现在来处理所有的通路层！
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    print("测试的！当前是进入到所有的通路层里面了！", j)
                elif gradients == 'Path1' and j == 0:               ### 此时是第一层通路层
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    IFPathwayBefore = False
                elif gradients == 'Path2' and j == 1:               ### 此时是第一层通路层
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    IFPathwayBefore = False
                elif gradients == 'Path3' and j == 2:               ### 此时是第一层通路层
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    IFPathwayBefore = False
                elif gradients == 'Path4' and j == 3:               ### 此时是第一层通路层
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    IFPathwayBefore = False
                elif gradients == 'Path5' and j == 4:               ### 此时是第一层通路层
                    outcome = AddGSEAScore(OriMapp, outcome, j)  ### 输入的是当前层和上一层的映射图，当前这层网络的输入数据，当前这层网络的编号
                    IFPathwayBefore = False
                else:
                    print("当前是第几层通路层以及要消第几层！", j, gradients)


            input_Now = outcome                   ### 当前在 经过隐藏层处理之前应该保留当前隐藏层的输入数据，以便于后续梯度加权的时候进行计算！
            outcome = hidden_layer(input_Now)       ### 在经过上述选择好隐藏层之后，用数据流将当前的这个隐藏层给走了！


            ### 下面这个是来决定是否要将当前层的GSEA分数放到当前层的后面！
            if IFGseaAfter:        ### 此标志意味着在网络层的后方来进行加知识
                print("现在，当前的这层通路层在他的输出部分融入对应的GSEA分数")
                # outcome = 0.5 * outcome        ### 原始的那个输入数据需要减半，与加入的那个GSEA分数进行配对！
                outcome = AddGSEAScoreAfter(OriMapp, outcome, j)
                # attention_probs = Dense(n_pathways, activation='sigmoid', name='attention{}'.format(i + 1),
                #                         W_regularizer=l2(w_reg))(outcome)
                # outcome = multiply([outcome, attention_probs], name='attention_mul{}'.format(i + 1))

                # outcome = XiaoRong(OriMapp, outcome, j)  ### 现在这个是用来进行消融的！




            decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2),
                                     W_regularizer=reg_l(w_reg_outcome))(outcome)                   ### 这个即为当前的这个通路层的决策输出结果！

            # testing
            if batch_normal:
                decision_outcome = BatchNormalization()(decision_outcome)
            decision_outcome = Activation(activation=activation_decision, name='o{}'.format(i + 2))(decision_outcome)
            decision_outcomes.append(decision_outcome)

            if IFDropout:         ### 通过这一参数来决定当前这个网络是否要加这个Dropout机制！
                drop2 = Dropout(dropout, name='dropout_{}'.format(i + 1))
                outcome = drop2(outcome, training=dropout_testing)

            feature_names['h{}'.format(i)] = names
            print("当前第%d层中，他所用的生物元素的数目！", i, len(names))
            j = j + 1
            print("目前所在的文件是：builders_utils.py，目前通路层中走了几个循环了！", j)
            # feature_names.append(names)
        i = len(maps)
        feature_names['h{}'.format(i - 1)] = maps[-1].index
        # print("现在，最后添加进来的这个index是谁！", maps[-1].index)


        # feature_names.append(maps[-1].index)
    # print("最终要进行输出的这个特征名称是谁！", feature_names)        ### 现在我所定义1的这个特征名字即为每一层它所对应的各个生物元素
    print("目前所在的文件是：builders_utils.py  最终要进行输出的输出结果是谁！", outcome)
    print("现在的这个决策输出是谁！", decision_outcomes)
    return outcome, decision_outcomes, feature_names

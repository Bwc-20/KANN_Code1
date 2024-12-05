### 现在这块要想办法获取数据的ID进而找到这组数据所对应的那个复发复发的情况
import pandas as pd
import numpy as np
from config_path import *
base_path = BASE_PATH                   ### 现在的这个base_path是D:\DailyCode\P-Net\CodeModel\(Father)pnet_prostate_paper-published_to_zenodo\pnet_prostate_paper-published_to_zenodo

dataPath = base_path + '/_database/DataBaseTwo/prad_tcga_pan_can_atlas_2018_clinical_data.tsv'
dataPath = base_path + '/_database/DataBaseTwo/FuFa.xlsx'
NeedDataPath = base_path + '/_database/DataBaseTwo/NeedData/'
NeedDataFuFaPath = base_path + '/_database/DataBaseTwo/NeedData-FuFa/'

def ReadFuFaSample():
    # 下面是按照列属性读取的
    temp = pd.read_excel(dataPath)
    temp = temp.values      ### 现在这个就是那个二维数组
    OriSamplePath = base_path + '/_database/DataBaseTwo/NeedData/response_paper.csv'
    OriSample = pd.read_csv(OriSamplePath, usecols=['id', 'response'])
    OriSample = OriSample.values[:, 0]             ### 这个是原始样本的那个ID编号！
    NewData = []         ### 这个是来存储新整理后的有关复发样本的标签数据
    junFilter = 0
    NoF, Fu = 0, 0
    print("现在的这个原始数据是谁！", OriSample, type(OriSample[0]))
    ### 下面就是要想办法对他进行一波过滤，去除那些没有复发转移值的元素
    for ele in temp:
        if type(ele[1]) is float:        ### 这种情况下是没有值的！
            continue
        ele[0] = ele[0] + '-01'
        ### 下面就是要跟当前已有的csv样本数据挨个比对了！
        # for oriele in OriSample:
        #     print("现在这个数据是谁！", type(oriele), oriele, type(ele[0]), ele[0])
        #     if oriele == ele[0]:
        #         junFilter = junFilter + 1
        #         thisValue = [ele[0]]
        #         if ele[1][0] == '0':  ### 首个字母是0代表不复发
        #             thisValue.append(0)
        #             NoF = NoF + 1
        #         else:
        #             thisValue.append(1)
        #             Fu = Fu + 1
        #         NewData.append(thisValue)
        #     else:
        #         continue

        if ele[0] not in OriSample:
            print("当前的这批数据都是谁！", type(ele[0]), ele[0])
            continue
        else:              ### 此时说明当前复发数据有值而且也有对应的1原始的突变等数据
            junFilter = junFilter + 1
            thisValue = [ele[0]]
            if ele[1][0] == '0':     ### 首个字母是0代表不复发
                thisValue.append(0)
                NoF = NoF +1
            else:
                thisValue.append(1)
                Fu = Fu + 1
            NewData.append(thisValue)

    print("整理后的复发的样本标签是怎样的！", junFilter, NewData, NoF, Fu)

    ### 现在对得到的这个二维数组进行打乱
    ## 先保存response文件
    responseData = NewData
    responseData = np.array(responseData)
    df = pd.DataFrame(responseData)
    df.to_csv(NeedDataFuFaPath + 'response_paper.csv')

    ### 下面依次处理拷贝数扩增与拷贝数缺失
    CNA_Data = pd.read_csv(NeedDataPath + 'P1000_data_CNA_paper.csv')
    CNA_Data = CNA_Data.values
    CNA_DataCeshi = pd.read_csv(NeedDataPath + 'P1000_data_CNA_paper_Ceshi.csv')
    CNA_DataCeshi = CNA_DataCeshi.values
    Mut_Data = pd.read_csv(NeedDataPath + 'P1000_final_analysis_set_cross_important_only.csv')
    Mut_Data = Mut_Data.values

    NewCNA_Data, NewCNA_DataCeshi, NewMut_Data = [], [], []
    for nowEle in NewData:
        # nowEle[0]
        for i in range(len(CNA_Data)):
            if CNA_Data[i][0] != Mut_Data[i][0] or CNA_Data[i][0] != CNA_DataCeshi[i][0] or CNA_DataCeshi[i][0] != Mut_Data[i][0]:
                print("报错！！现在这个是有问题的！")
            if nowEle[0] == CNA_Data[i][0]:         ### 当前的这个样本是被选中的！
                NewCNA_Data.append(CNA_Data[i])
                NewCNA_DataCeshi.append(CNA_DataCeshi[i])
                NewMut_Data.append(Mut_Data[i])


    NewCNA_Data, NewCNA_DataCeshi, NewMut_Data = np.array(NewCNA_Data), np.array(NewCNA_DataCeshi), np.array(NewMut_Data)
    dfCNA, dfCNACeshi, dfMut = pd.DataFrame(NewCNA_Data), pd.DataFrame(NewCNA_DataCeshi), pd.DataFrame(NewMut_Data)
    dfCNA.to_csv(NeedDataFuFaPath + 'P1000_data_CNA_paper.csv')
    dfCNACeshi.to_csv(NeedDataFuFaPath + 'P1000_data_CNA_paper_Ceshi.csv')
    dfMut.to_csv(NeedDataFuFaPath + 'P1000_final_analysis_set_cross_important_only.csv')


    ### 下面来处理训练数据和测试数据
    ProcessData = NewData
    PosData, NegData = [], []
    for NowEle in ProcessData:
        if NowEle[1] == 0:
            NegData.append(NowEle)
        else:
            PosData.append(NowEle)
    # TrainData = np.append(np.array([PosData[0:int(0.8*len(PosData))]]), np.array([NegData[0:int(0.8*len(NegData))]]))
    # ValData = np.append(np.array([PosData[int(0.8*len(PosData)):int(0.9*len(PosData))]]), np.array([NegData[int(0.8*len(NegData)):int(0.9*len(NegData))]]))
    # TestData = np.append(np.array([PosData[int(0.9*len(PosData)):int(1*len(PosData))]]), np.array([NegData[int(0.9*len(NegData)):int(1*len(NegData))]]))

    TrainData = PosData[0:int(0.8*len(PosData))] + NegData[0:int(0.8*len(NegData))]
    ValData = PosData[int(0.8*len(PosData)):int(0.9*len(PosData))] + NegData[int(0.8*len(NegData)):int(0.9*len(NegData))]
    TestData = PosData[int(0.9*len(PosData)):int(1*len(PosData))] + NegData[int(0.9*len(NegData)):int(1*len(NegData))]

    TrainData, ValData, TestData = np.array(TrainData), np.array(ValData), np.array(TestData)
    dfTrain, dtVal, dfTest = pd.DataFrame(TrainData), pd.DataFrame(ValData), pd.DataFrame(TestData)
    dfTrain.to_csv(NeedDataFuFaPath + 'training_set_0.csv')
    dtVal.to_csv(NeedDataFuFaPath + 'validation_set.csv')
    dfTest.to_csv(NeedDataFuFaPath + 'test_set.csv')





    print("现在读取的数据是谁！", type(CNA_Data), CNA_Data)





def Ceshi_Two():
    OriPath = base_path + '/_database/prostate/processed/response_paper.csv'
    NowPath = NeedDataPath + 'response_paper.csv'
    OriSample = pd.read_csv(OriPath)
    NowSample = pd.read_csv(NowPath)
    OriSample, NowSample = OriSample.values, NowSample.values
    # print("现在的两个样本是怎样的：", NowSample)
    Same, NoS = 0, 0
    for nowele in NowSample:
        for oriele in OriSample:

            if nowele[0] == oriele[0]:
                if nowele[1] == oriele[1]:
                    print("此时的演变北京", nowele, oriele)
                    Same = Same + 1
                else:
                    NoS = NoS +1
    print("最终相同的样本数目和不同的样本数目！", Same, NoS)





def Ceshi():
    OriSamplePath = base_path + '/_database/DataBaseTwo/NeedData/response_paper.csv'
    OriSample = pd.read_csv(OriSamplePath, usecols=['id', 'response'])
    OriSample = OriSample.values
    print("原始数据样本的情况是怎样的！", OriSample[:, 0])





Ceshi_Two()
# ReadFuFaSample()
# Ceshi()
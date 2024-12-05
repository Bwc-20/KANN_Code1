import pandas as pd
from config_path import BASE_PATH, DATA_PATH
import numpy as np
import os
import csv

def Ceshi_One():
    filename = DATA_PATH + '/DataBaseTwo/PRAD/MCTP_clinical.txt'
    test1 = pd.read_table(filename)         ## SAMPLE_ID
    SAMPLE_ID = test1["SAMPLE_TYPE"]   # 根据标题来取值             SAMPLE_TYPE
    SAMPLE_ID = np.array(SAMPLE_ID)

    print("那么现在的这个样本编号是多少呢？", type(SAMPLE_ID), type(SAMPLE_ID[1]), type(SAMPLE_ID[-1]), SAMPLE_ID)
    data1 = 'VAMP7|ENSG00000124333.10'
    print("判断一下特殊符号是不是在这里面！", data1.split('|')[0])
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    result = np.concatenate((arr1, arr2))
    print(result)

    data2 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    data2 = np.array(data2)
    cols = ['cols1', 'cols2', 'cols3']
    rows = ['rows1', 'rows2']
    df = pd.DataFrame(data2, columns=cols, index=rows)
    df = df.T
    print("那么现在这个pd类型的数据是怎样的！", df)                ## _database/DataBaseTwo/NeedData/response_paper.csv
    df.to_csv(DATA_PATH + '/DataBaseTwo/NeedData/ceshi.csv')

    my_dict = {'a': 1, 'b': 2, 'c': 3}
    keys_list = list(my_dict.keys())
    print(keys_list)


### 下面来获取这批样本所对应的类别标签 并将这些样本的编号以及对应的类别标签进行整合保存到指定的文件中
def GetSampleResponce():
    ## 第一步：从各个数据库中来读取 各个样本的编号以及这些样本对应的类别
    FileClass = ['MCTP', 'prad', 'tcga', 'SU2C']    ### 在这SU2C这个文件类型不用来读取类别，因为这类文件中的每个样本都是转移性 Metastasis
    AllData = {}          ## 现在这个字典中存放各个样本编号（作为key）以及这个样本的类别（Value）      现在这个样本类别的取值中，0表示原发性Primary， 1表示转移性Metastasis
    Neg, Pos = 0, 0      ### 看一下当前的这个阳性以及阴性样本分别是多少个！
    for nowFile in FileClass:
        filePath = DATA_PATH + '/DataBaseTwo/PRAD/' + nowFile + '_clinical.txt'
        ReadData = pd.read_table(filePath)
        if nowFile == 'SU2C':    ###在这SU2C这个文件类型不用来读取类别，因为这类文件中的每个样本都是转移性 Metastasis
            SAMPLE_ID = np.array(ReadData["SAMPLE_ID"])
            SAMPLE_TYPE = np.full(len(SAMPLE_ID), 'Metastasis')
        else:
            SAMPLE_ID, SAMPLE_TYPE = np.array(ReadData["SAMPLE_ID"]), np.array(ReadData["SAMPLE_TYPE"])
        for i in range(len(SAMPLE_ID)):
            if type(SAMPLE_TYPE[i]) == float:
                continue     ### 此时就说明后面的类别缺失，那么这个样本也就不再要了
            if SAMPLE_ID[i] in AllData:     ### 目前这块没走进过，说明各个数据文件中每个样本都是独立的！
                print("错误警告！！说明当前这个文件类别中得这个样本在其他的文件类别中也出现过了！")
            if SAMPLE_TYPE[i] == 'Primary':
                AllData[SAMPLE_ID[i]] = 0
                Neg = Neg + 1
            else:
                AllData[SAMPLE_ID[i]] = 1         ### 转移性的数据类别就是1
                Pos = Pos + 1
    print("目前所构造的这个字典的情况是怎样的！", Neg, Pos, len(AllData), AllData)

    ## 第二步：现在将读取的这些类别信息进行存储！
    savePath = DATA_PATH + '\\DataBaseTwo\\NeedData\\response_paper.csv'              ### 跟原来的那个数据集中的对应的数据文件的文件名保持一致！
    # 1. 创建文件对象
    f = open(savePath, 'w', newline='', encoding='utf-8')  ### 这个就是最终要写入的这个文件
    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)
    RowHeader = ["id", "response"]
    csv_writer.writerow(RowHeader)  ## 将这个标题写到对应的文件中！
    for ele in AllData:
        line = [ele, AllData[ele]]
        csv_writer.writerow(line)             ## 将当前这行数据写入指定文件中


### 下面是对各个数据库中的CNV文件进行处理，进行汇总并构造出与源数据集类似的形式！
def GetCNVFile():
    ## 第一步：从各个数据库中来读取 各个样本他们所涉及的基因的CNV情况！  每个数据库中的CNV文件中有许多位置的取值为NA，没有值，像这种情况感觉是可以为其填充0的，因为源数据集中，他最终是将突变数据与CNV数据两者取并集的，因此说有些基因是CNV没有的，像那种他就是填充0了
    ## 看看那些含有空值的基因删除之后，整体基因与候选基因交集的数目会发生怎样变化，变化不大的话就删了，变化大的话，那这些基因就全都留
    def Ceshi_One():
        ## 下面来看一下CNV数据文件中所涉及的基因有哪些！
        CNV_file = DATA_PATH + '/DataBaseTwo/PRAD/MCTP_cna.txt'
        ReadData = pd.read_table(CNV_file)
        # nowCols1 = np.array(ReadData.columns)[1:len(np.array(ReadData.columns))]
        print("现在的读取的这个数据文件是谁！", np.array(ReadData.columns)[1:len(np.array(ReadData.columns))])
        dataValue = ReadData.values
        # print("现在读取的这个文件是谁！", type(dataValue), type(dataValue[-1]), type(dataValue[-1][0]), type(dataValue[0][1]), np.isnan(dataValue[-1][0]), np.isnan(dataValue[0][1]), dataValue[-1][1:len(dataValue[-1])])
        # print("下面来判断一下数组中是否含有nan", True in np.isnan(dataValue[-1][1:len(dataValue[-1])]))


        ### 下面这个是原始的数据样本
        Temp = pd.read_csv(DATA_PATH + '/DataBaseTwo/NeedData/response_paper.csv')
        OriSamples = np.array(Temp['id'])            ### 现在这个就是原始能用的样本 一共1063个

        ## 下面这个是原始数据集的编码基因          _database/genes/HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt
        CodeGenesFile = DATA_PATH + '/genes/HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt'
        coding_genes_df = pd.read_csv(CodeGenesFile, sep='\t', header=None)
        coding_genes_df.columns = ['chr', 'start', 'end', 'name']
        coding_genes = set(coding_genes_df['name'].unique())             ### 现在这个是编码基因

        ## 下面这个是原始的候选基因             _database/genes/tcga_prostate_expressed_genes_and_cancer_genes.csv
        SelectGenesFile = DATA_PATH + '/genes/tcga_prostate_expressed_genes_and_cancer_genes.csv'
        df = pd.read_csv(SelectGenesFile, header=0)
        selected_genes = list(df['genes'])

        ### 编码基因与候选基因的交集
        OriGenes = set.intersection(coding_genes, set(selected_genes))
        print("目前这个交集是谁！", len(OriGenes), OriGenes)

        print("现在原始的数据样本是怎样的！", len(OriSamples), OriSamples)
        ### 下面分别来计算一下这四个数据集中样本的数量、基因的数量，这四个基因分别与候选基因重叠了多少个！以及样本重叠情况
        FileClass = ['MCTP', 'prad', 'tcga', 'SU2C']    ### 在这，四个CNV数据集之间，他们的样本是没有重叠的！
        Samples, Genes = [], []          ## 现在这个字典中存放各个样本编号（作为key）以及这个样本的类别（Value）      现在这个样本类别的取值中，0表示原发性Primary， 1表示转移性Metastasis
        CNVGenes = []            ### 这里面存放四个CNV文件中，各个基因的交集
        for nowFile in FileClass:
            filePath = DATA_PATH + '/DataBaseTwo/PRAD/' + nowFile + '_cna.txt'
            ReadData = pd.read_table(filePath)
            nowSample = np.array(ReadData.columns)[1:len(np.array(ReadData.columns))]
            nowGene = np.array(ReadData['Hugo_Symbol'])           ### 现在这个就是当前这个CNV文件中的基因
            for i in range(len(nowGene)):
                if '|' in nowGene[i]:
                    nowGene[i] = nowGene[i].split('|')[0]

            if nowFile == 'tcga':
                nowSample = nowSample[1:len(nowSample)]
            if nowFile == 'prad':
                nowSample = nowSample[2:len(nowSample)]
            Samples = Samples + nowSample.tolist()           ### 最终所有样本均汇聚于此
            ## 接下来判断一下当前这个基因与候选基因之间的重叠关系！
            intersectGenes = set.intersection(OriGenes, set(nowGene))      ### 看当前这个数据文件中的基因重叠了多少个
            print("当前这个数据文件中CNV所涉及的基因，他重叠了多少个！", len(intersectGenes), len(nowGene), nowGene)

            ## 下面的这个操作主要是为了下面求CNV文件数据服务的！
            if len(CNVGenes) == 0:
                CNVGenes = set(nowGene)
            else:
                CNVGenes = set.intersection(CNVGenes, set(nowGene))

        intersectSamples = set.intersection(set(OriSamples), set(Samples))
        print("最终这个样本重叠了多少个！", len(intersectSamples), len(Samples), len(OriSamples), len(CNVGenes), set(Samples)-intersectSamples)
        print("上天啊！最终这四个文件中的样本组合在一块是怎样的！", Samples)


        ### 先按照严格的方法计算一下！
        ## 四个CNV文件中的数据，基因取交集(这样！他的这个交集跟候选基因以及编码基因来取)， 取完交集之后，再看一下剩下的基因中还有没有取值为空的那些基因！
        OriGenes = OriGenes                ### 现在开始来看当前这个CNV文件中的数据，如果它里面的基因不在这个原始基因中的话，那么这个基因就直接去掉！
        print("现在原始的这个基因是谁！", OriGenes)
        FileClass = ['MCTP', 'prad', 'tcga', 'SU2C']    ### 在这，四个CNV数据集之间，他们的样本是没有重叠的！
        Genes = []         ## 现在这个字典中存放各个样本编号（作为key）以及这个样本的类别（Value）      现在这个样本类别的取值中，0表示原发性Primary， 1表示转移性Metastasis
        for nowFile in FileClass:
            nowGenesDict = {}          ## 这个来存放当前的这个CNV文件中的每个基因他的CNV数据
            filePath = DATA_PATH + '/DataBaseTwo/PRAD/' + nowFile + '_cna.txt'
            ReadData = pd.read_table(filePath)
            ReadData = np.array(ReadData)
            for ele in ReadData:
                ## 第一，分隔基因与CNV数据
                nowgene = ele[0]        ### 第一个数据一定是基因
                if nowFile == 'prad':
                    nowCNV = ele[3:len(ele)]           ### 前三位不是样本的CNV数据
                elif nowFile == 'tcga':
                    nowCNV = ele[2:len(ele)]             ### 前两位不是样本的CNV数据
                else:
                    nowCNV = ele[1:len(ele)]          ### 前一位不是样本的CNV数据
                nowCNV = np.array(nowCNV, dtype=float)     ### 之所以添加这句代码，主要是为了下面判断数组中是否含有nan数据服务的！否则的话下面的那个判断数组中是否含有nan数据的代码走不通！

                ## 第二，对当前的这个基因进行再一波的处理
                if '|' in nowgene:
                    nowgene = nowgene.split('|')[0]

                ## 第三，对CNV数据进行过滤处理，   先按照严格的来，只有当这个基因他在编码基因与选定的基因里，同时他里面不含有nan的情况下，这个基因的CNV数据才会被保留！
                # if nowgene in OriGenes and np.isnan(nowCNV).any() == False:            ### 此时当前这个样本是需要保留的
                # if nowgene in OriGenes:
                ### 先按照这种严格的方法进行处理，就是说这个基因所对应的CNV数据中但凡含有一个nan，那么这个基因就不要了！   下面是含有三个判定条件，看看当前的这个基因如果在四个文件中都出现过，而且出现在候选基因以及编码基因中，而且他的CNV数据中没有nan，此时，这个数据就保留
                ### 对于严格情况下，各个文件中所含有的基因数目分别是8325 9216 9216 9128， 对于非严格意义下，各个文件中所含有的基因数目分别为9216 9216 9216 9216
                if nowgene in OriGenes and nowgene in CNVGenes and np.isnan(nowCNV).any() == False:              ### 先按照这种严格的方法进行处理，就是说这个基因所对应的CNV数据中但凡含有一个nan，那么这个基因就不要了！
                    nowGenesDict[nowgene] = nowCNV


            Genes.append(nowGenesDict)               ### 当前这个类别他的CNV基因数据

        print("当前所构造出来的这个各个类别的CNV数据中分别是有多少个合格的基因！", len(Genes[0]), len(Genes[1]), len(Genes[2]), len(Genes[3]))
        ### 第四，将这四个数据文件中的基因数据取交集合并
        ## 因为此时是严格条件下，各个文件中合适的基因的数目分别是：8325 9216 9216 9128    因此第一个文件MCTP文件中的基因数目最少！
        FinalGeneCNV = []          ### 他是要作为二维数组的样子的，里面存放每个基因所对应的那些样本的CNV数据
        genes1, genes2, genes3, genes4 = list(Genes[0]), list(Genes[1]), list(Genes[2]), list(Genes[3])
        FinalGenes = set.intersection(*[set(genes1), set(genes2), set(genes3), set(genes4)])
        FinalGenes = list(FinalGenes)
        print("最终这四个文件中基因的交集是怎样的！", type(FinalGenes), len(FinalGenes), FinalGenes)
        FinalSamples = Samples               ### 这里面分别来存放最终所构造的这个CNV文件的列标题与行标题
        ## 依次遍历基因数最少的那个文件，读取1其中的每个基因，并将这个基因所对应的四个文件中的CNV数据进行合并！
        for ele in FinalGenes:
            if ele not in Genes[0]:
                print("当前这个基因不在第一个文件中", ele)
            if ele not in Genes[1]:
                print("当前这个基因不在第二个文件中", ele)
            if ele not in Genes[2]:
                print("当前这个基因不在第三个文件中", ele)
            if ele not in Genes[3]:
                print("当前这个基因不在第四个文件中", ele)
            resultCNV = np.concatenate((Genes[0][ele], Genes[1][ele]))              ### 将这两个基因所对应的CNV数据进行合并
            resultCNV = np.concatenate((resultCNV, Genes[2][ele]))
            resultCNV = np.concatenate((resultCNV, Genes[3][ele]))
            FinalGeneCNV.append(resultCNV)        ### 将当前基因所对应的四个文件中的CNV数据统一进行合并，之后将合并后的CNV数据再进行写入到对应的二维数组中
        FinalGeneCNV = np.array(FinalGeneCNV)


        ### 第五，合并后的数据开始进行保存，只是他的行列需要转置
        cols = FinalSamples         ### 现在的列标题应该是每个的样本
        rows = FinalGenes           ### 现在的行标题应该是每个的基因
        df = pd.DataFrame(FinalGeneCNV, columns=cols, index=rows)
        df = df.T
        print("那么现在这个pd类型的数据是怎样的！", df)  ## _database/DataBaseTwo/NeedData/response_paper.csv
        df.to_csv(DATA_PATH + '/DataBaseTwo/NeedData/P1000_data_CNA_paper.csv')         ### 现在这个CNV的数据文件便构造完成了 他是 1053 rows x 8255 columns








    ### 遍历各个CNV文件，先将这四个数据集进行合并成一个，之后分别按照基因以及样本来对其进行删除
    def Censhi_Two():
        FileClass = ['MCTP', 'prad', 'tcga', 'SU2C']    ### 在这，四个CNV数据集之间，他们的样本是没有重叠的！
        AllData = []          ## 现在这个字典中存放各个样本编号（作为key）以及这个样本的类别（Value）      现在这个样本类别的取值中，0表示原发性Primary， 1表示转移性Metastasis
        for nowFile in FileClass:
            filePath = DATA_PATH + '/DataBaseTwo/PRAD/' + nowFile + '_cna.txt'
            ReadData = pd.read_table(filePath)
            nowCols = np.array(ReadData.columns)[1:len(np.array(ReadData.columns))]
            if len(AllData) > 0:
                intersect = set.intersection(set(AllData), set(nowCols))
                AllData = intersect
                print("他们现在交集的长度是多少！", len(intersect), len(nowCols))
            else:
                AllData = nowCols
            # AllData = AllData + nowCols.tolist()
        print("那么最终这个样本的长度是多少呢？", len(AllData))




    Ceshi_One()


### 现在下面这歌函数来构造对应的突变数据文件！
def GetMut_importantFile():
    ## 第一步：从各个数据库中来读取 各个样本的编号以及这些样本所含有的基因，如果这个样本中有这个基因的话就说明该基因在这个样本内突变，否则就是不突变
    ## 并构造一下每个样本它内部所包含的突变基因都是谁！
    FileClass = ['MCTP', 'prad', 'tcga', 'SU2C']    ### 在这SU2C这个文件类型不用来读取类别，因为这类文件中的每个样本都是转移性 Metastasis
    AllData = {}          ## 现在这个字典中存放各个样本编号（作为key）以及这个样本所含有的突变基因（Value）
    MutGenes = []      ### 现在是来测试一下，看看所有样本中他们的突变基因组合在一起是有哪些？有多少个！
    for nowFile in FileClass:
        filePath = DATA_PATH + '/DataBaseTwo/PRAD/' + nowFile + '.txt'
        ReadData = pd.read_table(filePath)
        Genes, Samples = np.array(ReadData["Hugo_Symbol"]), np.array(ReadData["Tumor_Sample_Barcode"])
        for i in range(len(Genes)):
            nowgene, nowsample = Genes[i], Samples[i]
            if nowgene not in MutGenes:
                MutGenes.append(nowgene)
            if nowsample not in AllData:      ## 此时说明当前这个样本是第一次出现！  那么它所对应的value就应该是一个空的列表！
                AllData[nowsample] = []
            else:
                AllData[nowsample].append(nowgene)
    print("那么现在最终所构造的这个突变的样本情况，以及突变基因情况：", len(AllData), len(MutGenes))                  ### 现在一共是涉及1061个样本，含有 18461个不同的突变基因！

    ## 要不这样：最终突变文件中的基因由：当前突变基因MutGenes、候选基因、编码基因三者取并集组成！
    ## 第二步，根据已经构造好的CNV数据中的行列标题（样本与基因）以他的样本与基因来作为突变自己的
    Temp = pd.read_csv(DATA_PATH + '/DataBaseTwo/NeedData/P1000_data_CNA_paper_Ceshi.csv')             ### 这个数据文件与P1000_data_CNA_paper.csv这个文件里面的内容其实是一摸一样，区别在于他的第一行第一列那个标题是id，不再是空，方面下面求取样本那一列
    rows, cols = np.array(Temp['id']), np.array(Temp.columns)       ## 两者分别代表样本与基因
    cols = cols[1:len(cols)]
    print("现在这行列标题分别是谁！", len(rows), len(cols), rows)

    ## 第三步，根据构造好的行列标题，来填充具体的突变数据
    FinalValue = []
    for nowsample in rows:
        Mutgene = AllData[nowsample]    ### 看看当前这个样本他的突变基因有哪些！
        MutValue = []
        for nowgene in cols:
            if nowgene in Mutgene:     ### 此时说明当前的这个基因在此样本中发生突变了！  那么这个基因位置就要填充1，否则就是0
                MutValue.append(1)
            else:
                MutValue.append(0)
        FinalValue.append(np.array(MutValue))        ### 当前这个样本里面的各个基因都已经处理好了！
    FinalValue = np.array(FinalValue)
    ### 第四步，构造好的突变数据进行保存为csv文件
    df = pd.DataFrame(FinalValue, columns=cols, index=rows)
    print("那么现在这个pd类型的数据是怎样的！", df)  ## _database/DataBaseTwo/NeedData/response_paper.csv
    df.to_csv(DATA_PATH + '/DataBaseTwo/NeedData/P1000_final_analysis_set_cross_important_only.csv')  ### 现在这个CNV的数据文件便构造完成了 他是 1053 rows x 8255 columns





### 现在就是将整体数据随机的划分为训练集、验证集与测试集！
def SplitTrain_Valit_Test():
    ### 在这里面response_paper.csv 这个文件不用动  他是作为总的，大的类别标签对应表   现在来读取目前数据集中实际用了哪些样本并将它们进行分隔
    Temp = pd.read_csv(DATA_PATH + '/DataBaseTwo/NeedData/P1000_data_CNA_paper_Ceshi.csv')             ### 这个数据文件与P1000_data_CNA_paper.csv这个文件里面的内容其实是一摸一样，区别在于他的第一行第一列那个标题是id，不再是空，方面下面求取样本那一列
    rows, cols = np.array(Temp['id']), np.array(Temp.columns)       ## 两者分别代表样本与基因
    cols = cols[1:len(cols)]
    print("现在所构造的这个样本是谁！", len(rows), rows)
    OriSampleLabel = pd.read_csv(DATA_PATH + '/DataBaseTwo/NeedData/response_paper.csv')          ### 这是原始的数据样本
    OriSample, OriLabel = np.array(OriSampleLabel['id']), np.array(OriSampleLabel['response'])     ## 两者分别代表原始的样本编号与对应的标签
    ## 现在来构造样本标签字典
    SampleDict = {}
    for i in range(len(OriSample)):
        if OriSample[i] in rows:           ### 表明当前的这个样本确实是被选中了！
            SampleDict[OriSample[i]] = OriLabel[i]
    print("那么最终所购的这个字典的样子是什么样的！", len(SampleDict), SampleDict)
    ## 下面来将原始的数据打乱并完成分隔
    np.random.shuffle(rows)     # 打乱
    NewNum = int(0.10 * len(rows))  ### 验证集与测试集分别是多少数据
    ValSample, TestSample, TrainSample = rows[0:NewNum], rows[NewNum:2*NewNum], rows[2*NewNum:len(rows)]     ## 1:1:8

    print("现在这三个文件的长度分别是多少！", len(ValSample), len(TestSample), len(TrainSample))
    pdValue = []
    AllData = [ValSample, TestSample, TrainSample]
    for nowData in AllData:
        Value, Index = [], []
        for i in range(len(nowData)):
            Index.append(i)          ### 这个是代表最终所构建的这个样本的索引
            nowValue = [nowData[i], SampleDict[nowData[i]]]              ### 现在来求出当前的这个样本的编号以及这个样本所对应的类别标签
            nowValue = np.array(nowValue)
            Value.append(nowValue)
        Value = np.array(Value)
        ## 下面开始进行保存！
        df = pd.DataFrame(Value, columns=['id', 'response'], index=Index)
        pdValue.append(df)        ### 最终所要进行保存的那些数据
        print("现在所构造的这个样本的长度是怎样的！", len(Value), len(Index))

    ## 现在开始进行保存！
    pdValue[0].to_csv(DATA_PATH + '/DataBaseTwo/NeedData/validation_set.csv')
    pdValue[1].to_csv(DATA_PATH + '/DataBaseTwo/NeedData/test_set.csv')
    pdValue[2].to_csv(DATA_PATH + '/DataBaseTwo/NeedData/training_set_0.csv')



### 下面是来进行分析并测试新数据的
def dataAnalysis_Process():
    ### 第一组分析新数据
    def Analysis_One():
        Temp = pd.read_csv(DATA_PATH + '/DataBaseTwo/NeedData/validation_set.csv')
        label = np.array(Temp['response'])
        print("当前验证集中阳性样本数目及占比：", sum(label), sum(label)/len(label))
        Temp = pd.read_csv(DATA_PATH + '/DataBaseTwo/NeedData/test_set.csv')
        label = np.array(Temp['response'])
        print("当前测试集中阳性样本数目及占比：", sum(label), sum(label)/len(label))
        Temp = pd.read_csv(DATA_PATH + '/DataBaseTwo/NeedData/training_set_0.csv')
        label = np.array(Temp['response'])
        print("当前训练集中阳性样本数目及占比：", sum(label), sum(label)/len(label))

        Temp = pd.read_csv(DATA_PATH + '/DataBaseTwo/NeedData/P1000_data_CNA_paper_Ceshi.csv')             ### 这个数据文件与P1000_data_CNA_paper.csv这个文件里面的内容其实是一摸一样，区别在于他的第一行第一列那个标题是id，不再是空，方面下面求取样本那一列
        cols = np.array(Temp.columns)       ## 两者分别代表样本与基因
        cols = cols[1:len(cols)]            ### 这个就是新数据中所用的基因

        ### 下面来看一下旧数据中那些基因与他的匹配程度
        from CeShi.OtherTest.ModelParamSave.ModelParam_Two import LayerEleGSEA, LayerEleRelationship
        LayerGSEA = LayerEleGSEA()
        GSEADict = LayerGSEA.getLayer0()            ### 这个就表示基因层的GSEA分数
        OriGenes = GSEADict.keys()

        intersect = set.intersection(set(OriGenes), set(cols))
        print("当前基因与旧基因之间的重复元素有多少个！", len(intersect), len(cols))


    ### 第二组分析新数据，下面这个函数主要是来看看新数据中所用的各层通路与旧数据中所用的各层通路是不是完全一致
    def Analysis_Two():
        Path = BASE_PATH + '/DataBaseTwo/NeedData/validation_set.csv'



    ### 下面这个来保证输入的这个训练数据他的分布是极其均衡的！
    # def
























# Ceshi_One()

# GetSampleResponce()
# GetMut_importantFile()
# GetCNVFile()
# SplitTrain_Valit_Test()
dataAnalysis_Process()
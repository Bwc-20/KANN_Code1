import os
import urllib.request
from os.path import join, basename, dirname

processed_dir = 'processed'
data_dir = 'raw_data'

current_dir = dirname(__file__)

processed_dir = join(current_dir, processed_dir)
data_dir = join(current_dir, data_dir)


def download_data():
    print ('downloading data files')
    # P1000 data
    ## 下载进行训练和测试所需要用的那些病人的数据   就包含了每个病人的编号，癌症状态信息，突变数目等
    file2 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM6_ESM.xlsx'       ### 这个文件中是补充表4：使用MutSig2CV的1013个样本中显著突变的基因（SMGs）列表
    file1 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM4_ESM.txt'
    file3 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM10_ESM.txt'       ### 补充表10：拷贝数调用矩阵
    file4 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM10_ESM.txt'
    file5 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0078-z/MediaObjects/41588_2018_78_MOESM5_ESM.xlsx'

    # Met500 files 'https://www.nature.com/articles/nature23306'

    links = [file1, file2, file3, file4, file5]

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for link in links:
        print(('downloading file {}'.format(link)))
        filename = join(data_dir, basename(link))
        with open(filename, 'wb') as f:
            f.write(urllib.request.urlopen(link).read())
            f.close()


download_data()

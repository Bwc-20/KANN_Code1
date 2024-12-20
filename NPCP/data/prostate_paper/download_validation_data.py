import os
import urllib.request
from urllib.request import urlopen
from os.path import join, basename, dirname, exists


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


current_dir = dirname(__file__)

processed_dir = 'processed'
data_dir = 'external_validation'

processed_dir = join(current_dir, processed_dir)
data_dir = join(current_dir, data_dir)

if not exists(data_dir):
    os.makedirs(data_dir)


def download_data_MET500():
    saving_dir = join(data_dir, 'Met500')
    if not exists(saving_dir):
        os.makedirs(saving_dir)

    print ('downloading data files')
    # sub_dir = 'Met500'
    # file1= 'https://met500.path.med.umich.edu/met500_download_datasets/cnv_v4.csv'
    file1 = 'https://met500.path.med.umich.edu/met500_download_datasets/somatic_v4.csv'
    # file2 = 'https://www.dropbox.com/s/62fqw2zgc6ayxvg/Met500_cnv.txt?dl=0'
    # file3 = 'https://www.dropbox.com/s/htcx4f09k231l5m/samples.txt?dl=0'
    file2 = 'https://www.dropbox.com/s/62fqw2zgc6ayxvg/Met500_cnv.txt'
    file3 = 'https://www.dropbox.com/s/htcx4f09k231l5m/samples.txt'

    links = [file1, file2, file3]

    for link in links:
        print(('downloading file {}'.format(link)))
        filename = join(saving_dir, basename(link))
        # print("测试一下，看看目前的这个文件名是什么！", saving_dir, filename)
        # filename = 'D:/DailyCode/P-Net/CodeModel/pnet_prostate_paper-published_to_zenodo/pnet_prostate_paper-published_to_zenodo/data/prostate_paper\external_validation\Met500\somatic_v4.csv'
        # filename.replace("\", "/"")
        if filename[-4] != '.':
            filename = filename[0: len(filename)-5]
        print(("替换后的文件名时！", filename))
        with open(filename, 'wb') as f:
            f.write(urllib.request.urlopen(link).read())
            f.close()


def download_data_PRAD():
    pass
    # https://www.nature.com/articles/nature20788#MOESM323
    file1 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fnature20788/MediaObjects/41586_2017_BFnature20788_MOESM324_ESM.zip'
    print("目前是走到第几个文件了！")
    file2 = 'https://static-content.springer.com/esm/art%3A10.1038%2Fnature20788/MediaObjects/41586_2017_BFnature20788_MOESM325_ESM.zip'
    links = [file1, file2]
    for link in links:
        print(('downloading file {}'.format(link)))
        filename = join(data_dir, basename(link))
        print("来测试一下，现在的这个文件名字是谁！", filename, data_dir)
        with open(filename, 'wb') as f:
            f.write(urllib.request.urlopen(link).read())
            f.close()


download_data_MET500()
download_data_PRAD()
print('Done')

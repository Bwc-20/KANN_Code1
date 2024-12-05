from os.path import join, realpath, dirname

BASE_PATH = dirname(realpath(__file__))               ### 此时这个给定的是当前大项目所在的路径：D:\DailyCode\P-Net\CodeModel\(Father)pnet_prostate_paper-published_to_zenodo\pnet_prostate_paper-published_to_zenodo
# print("测试一下这个BASE_PATH是谁", BASE_PATH)
DATA_PATH = join(BASE_PATH, '_database')              ### 这个是指项目的数据集所在的路径：D:\DailyCode\P-Net\CodeModel\(Father)pnet_prostate_paper-published_to_zenodo\pnet_prostate_paper-published_to_zenodo\_database
# print("测试一下这个DATA_PATH是谁", DATA_PATH)
GENE_PATH = join(DATA_PATH, 'genes')                  ### D:\DailyCode\P-Net\CodeModel\(Father)pnet_prostate_paper-published_to_zenodo\pnet_prostate_paper-published_to_zenodo\_database\genes
PATHWAY_PATH = join(DATA_PATH, 'pathways')
REACTOM_PATHWAY_PATH = join(PATHWAY_PATH, 'Reactome')   ## D:\DailyCode\P-Net\CodeModel\(Father)pnet_prostate_paper-published_to_zenodo\pnet_prostate_paper-published_to_zenodo\_database\pathways\Reactome
PROSTATE_DATA_PATH = join(DATA_PATH, 'prostate')
RUN_PATH = join(BASE_PATH, 'train')
LOG_PATH = join(BASE_PATH, '_logs')
PROSTATE_LOG_PATH = join(LOG_PATH, 'p1000')
PARAMS_PATH = join(RUN_PATH, 'params')
POSTATE_PARAMS_PATH = join(PARAMS_PATH, 'P1000')
PLOTS_PATH = join(BASE_PATH, '_plots')                ### 此时这个给定的是当前大项目所在的路径：D:\DailyCode\P-Net\CodeModel\(Father)pnet_prostate_paper-published_to_zenodo\pnet_prostate_paper-published_to_zenodo\_plots


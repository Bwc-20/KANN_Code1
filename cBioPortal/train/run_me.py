
import sys
from os.path import join, dirname, realpath
current_dir = dirname(realpath(__file__))
sys.path.insert(0, dirname(current_dir))
import os
import imp
import logging
import random
import timeit
import datetime
import numpy as np
import tensorflow as tf
from utils.logs import set_logging, DebugFolder
from config_path import PROSTATE_LOG_PATH, POSTATE_PARAMS_PATH
from pipeline.train_validate import TrainValidatePipeline
from pipeline.one_split import OneSplitPipeline
from pipeline.crossvalidation_pipeline import CrossvalidationPipeline
from pipeline.LeaveOneOut_pipeline import LeaveOneOutPipeline

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

random_seed = 234
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_random_seed(random_seed)

timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())

## 下面为计算训练的间隔时间（花了多少分，多少秒）
def elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

params_file_list = []

# pnet        下面这几行分别是P-net网络的参数设置文件，每次选择其中一个进行运行！
# params_file_list.append('./pnet/onsplit_average_reg_10_tanh_large_testing')      ### 这个是添加进参数训练时所配置的相关参数设置文件
# params_file_list.append('./pnet/onsplit_average_reg_10_tanh_large_testing_inner')
### 下面的这个文件主要是用来导入模型所需的相关的参数
params_file_list.append('./pnet/crossvalidation_average_reg_10_tanh')              ### train/params/P1000/external_validation/pnet_validation.py     这个文件主要是用来导入模型所需的相关的参数



# params_file_list.append('./external_validation/pnet_validation')                   ### 这个处理的是train/params/P1000/external_validation/pnet_validation.py文件，进行训练验证
#
# # other ML models
# params_file_list.append('./compare/onsplit_ML_test')
# params_file_list.append('./compare/crossvalidation_ML_test')
#
# # dense
# params_file_list.append('./dense/onesplit_number_samples_dense_sameweights')
# params_file_list.append('./dense/onsplit_dense')
#
# # number_samples
# params_file_list.append('./number_samples/crossvalidation_average_reg_10')
## params_file_list.append('./number_samples/crossvalidation_average_reg_10_tanh')
# params_file_list.append('./number_samples/crossvalidation_number_samples_dense_sameweights')

# # external_validation
# params_file_list.append('./external_validation/pnet_validation')
#
# #reviews------------------------------------
# #LOOCV
# params_file_list.append('./review/LOOCV_reg_10_tanh')
# #ge
# params_file_list.append('./review/onsplit_average_reg_10_tanh_large_testing_ge')
# #fusion
# params_file_list.append('./review/fusion/onsplit_average_reg_10_tanh_large_testing_TMB')
# params_file_list.append('./review/fusion/onsplit_average_reg_10_tanh_large_testing_fusion')
# params_file_list.append('./review/fusion/onsplit_average_reg_10_tanh_large_testing_fusion_zero')
# params_file_list.append('./review/fusion/onsplit_average_reg_10_tanh_large_testing_inner_fusion_genes')
#
# #single copy
# params_file_list.append('./review/9single_copy/onsplit_average_reg_10_tanh_large_testing_single_copy')
# params_file_list.append('./review/9single_copy/crossvalidation_average_reg_10_tanh_single_copy')
#
# #custom arch
# params_file_list.append('./review/10custom_arch/onsplit_kegg')
#
# #learning rate
# params_file_list.append('./review/learning_rate/onsplit_average_reg_10_tanh_large_testing_inner_LR')

# hotspot
# params_file_list.append('./review/9hotspot/onsplit_average_reg_10_tanh_large_testing_hotspot')
# params_file_list.append('./review/9hotspot/onsplit_average_reg_10_tanh_large_testing_count')

# cancer genes
# params_file_list.append('./review/onsplit_average_reg_10_tanh_large_testing')
# params_file_list.append('./review/onsplit_average_reg_10_cancer_genes_testing')
# params_file_list.append('./review/crossvalidation_average_reg_10_tanh_cancer_genes')

# review 2 (second iteration of reviews)
# params_file_list.append('./review/cnv_burden_training/onsplit_average_reg_10_tanh_large_testing_TMB2')
# params_file_list.append('./review/cnv_burden_training/onsplit_average_reg_10_tanh_large_testing_account_zero2')
# params_file_list.append('./review/cnv_burden_training/onsplit_average_reg_10_tanh_large_testing_TMB_cnv')
# params_file_list.append('./review/cnv_burden_training/onsplit_average_reg_10_tanh_large_testing_cnv_burden2')

for params_file in params_file_list:
    log_dir = join(PROSTATE_LOG_PATH, params_file)      ### 指定要记录的日志所在的位置
    log_dir = log_dir
    set_logging(log_dir)        ## 记录一下当前的训练情况，把它写到指定的日志记录里面
    params_file = join(POSTATE_PARAMS_PATH, params_file)      ###指定一下当前模型训练时参数所在的那个文件！
    logging.info('random seed %d' % random_seed)
    params_file_full = params_file + '.py'
    print("测试一下，目前所指定的训练时的参数文件是谁！", params_file_full)
    params = imp.load_source(params_file, params_file_full)         ### 这个函数的作用在于将 params_file_full 这个文件中所实现的功能给整体导入到params_file中，使其作为一个模块而存在！

    DebugFolder(log_dir)
    if params.pipeline['type'] == 'one_split':
        print("目前是one_split走进来了！")
        pipeline = OneSplitPipeline(task=params.task, data_params=params.data, model_params=params.models,
                                    pre_params=params.pre, feature_params=params.features,
                                    pipeline_params=params.pipeline,
                                    exp_name=log_dir)

    elif params.pipeline['type'] == 'crossvalidation':         ### 这块是来进行五折交叉验证的，具体的传入训练数据来进行拟合模型，并根据测试数据获取测试结果（预测分数）
        print("目前是交叉熵验证走进来了！")
        print("现在是在 run_me.py文件中，目前来测试一下当前的这个params.models是谁", params.models)
        pipeline = CrossvalidationPipeline(task=params.task, data_params=params.data, feature_params=params.features,
                                           model_params=params.models, pre_params=params.pre,
                                           pipeline_params=params.pipeline, exp_name=log_dir)

    elif params.pipeline['type'] == 'Train_Validate':          ### 根据输入数据进行组合，之后输入模型进行训练，而且在训练的时候使用双模型根据两个模型的输出结果进行组合进而得到最终的预测分数
        print("目前是训练验证走进来了！")
        pipeline = TrainValidatePipeline(data_params=params.data, model_params=params.models, pre_params=params.pre,
                                         feature_params=params.features, pipeline_params=params.pipeline,
                                         exp_name=log_dir)

    elif params.pipeline['type'] == 'LOOCV':
        print("目前是LeaveOneOutPipeline走进来了！")
        pipeline = LeaveOneOutPipeline(task=params.task, data_params=params.data, feature_params=params.features,
                                       model_params=params.models, pre_params=params.pre,
                                       pipeline_params=params.pipeline, exp_name=log_dir)
    start = timeit.default_timer()
    print("直接开始运行了")
    pipeline.run()    ### 设定好参数情况，现在开始真正的运行！
    stop = timeit.default_timer()
    mins, secs = elapsed_time(start, stop)
    logging.info('Elapsed Time: {}m {}s'.format(mins, secs))

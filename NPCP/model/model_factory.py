import logging

from sklearn import svm, linear_model
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import Ridge, ElasticNet, Lasso, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from model import nn


### 本文件主要是来构造当前的P-Net模型以及其他的对标的算法


# get a model object from a dictionary     从字典中获得一个模型对象
# the params is in the format of {'type': 'model_type', 'params' {}}
# an example is params = {'type': 'svr', 'parmas': {'C': 0.025} }
### 下面这个函数根据输入的模型信息来确定生成当前设计的模型还是其他的对比算法！

def construct_model(model_params_dict):
    model_type = model_params_dict['type']
    p = model_params_dict['params']
    print("现在这个是在model_factory.py文件中的，这个模型的参数以及这个模型的类型是谁！", p, model_type)
    # logging.info ('model type: ', str(model_type))
    # logging.info('model paramters: {}'.format(p))

    if model_type == 'svr':
        model = svm.SVR(max_iter=5000, **p)

    if model_type == 'knn':
        model = KNeighborsClassifier(**p)

    if model_type == 'svc':
        model = svm.SVC(max_iter=5000, **p)

    if model_type == 'linear_svc':
        model = LinearSVC(max_iter=5000, **p)

    if model_type == 'multinomial':
        model = MultinomialNB(**p)

    if model_type == 'nearest_centroid':
        model = NearestCentroid(**p)

    if model_type == 'bernoulli':
        model = BernoulliNB(**p)

    if model_type == 'sgd':
        model = SGDClassifier(**p)                 ### 这个是随机梯度下降分类器！   SGDClassifier是一系列采用了梯度下降来求解参数的算法的集合，例如（SVM, logistic regression)等； 而sklearn中，LogisticRegression的实现方法是基于“liblinear”, “newton-cg”, “lbfgs” and “sag”这些库来实现的，当数据集特别大的时候，推荐使用SGDClassifier中的逻辑回归
        ### 所以说眼下的这个SGDClassifier() 其实是一个算法的集合！   它里面包含了SVM以及逻辑回归，而且这两类算法都是使用SGD算法来进行实现的！

    if model_type == 'gaussian_process':
        model = GaussianProcessClassifier(**p)

    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(**p)

    if model_type == 'random_forest':
        model = RandomForestClassifier(**p)

    if model_type == 'adaboost':
        model = AdaBoostClassifier(**p)

    if model_type == 'svr':
        model = svm.SVR(max_iter=5000, **p)
    # elif model_type == 'dt':
    #     # from sklearn.tree import DecisionTreeClassifier
    #     # model = DecisionTreeClassifier(**p)
    #     model = ModelWrapper(model)
    # elif model_type == 'rf':
    #     # from sklearn.ensemble import RandomForestClassifier
    #     model = RandomForestClassifier(**p)
    #     model = ModelWrapper(model)

    if model_type == 'ridge_classifier':
        model = RidgeClassifier(**p)

    elif model_type == 'ridge':
        model = Ridge(**p)


    elif model_type == 'elastic':
        model = ElasticNet(**p)
    elif model_type == 'lasso':
        model = Lasso(**p)
    elif model_type == 'randomforest':
        model = DecisionTreeRegressor(**p)

    elif model_type == 'extratrees':
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(**p)
        # print model

    elif model_type == 'randomizedLR':
        from sklearn.linear_model import RandomizedLogisticRegression
        model = RandomizedLogisticRegression(**p)

    elif model_type == 'AdaBoostDecisionTree':
        DT_params = params['DT_params']
        model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(**DT_params), **p)
    elif model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(**p)
    elif model_type == 'ranksvm':
        model = RankSVMKernel()
    elif model_type == 'logistic':
        logging.info('现在为model_factory.py文件，目前所要求的模型类别为：逻辑回归，model class {}'.format(model_type))
        model = linear_model.LogisticRegression()

    elif model_type == 'nn':
        model = nn.Model(**p)

    elif model_type == 'nn':
        model = nn.Model(**p)

    return model



### 下面就是根据传过来的模型参数来构造模型（如果模型的参数达标那么就进行构造！）
def get_model(params):
    if type(params['params']) == dict:
        model = construct_model(params)
    else:
        model = params['params']
    return model

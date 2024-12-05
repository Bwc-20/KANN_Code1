import pickle
import logging
import os
import time

# import model.coef_weights_utils
# from coef_weights_utils import coef_weights_utils.get_permutation_weights

### 下面为测试部分
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'


from keras.models import Sequential
from matplotlib import pyplot as plt




def save_model(model, filename):
    print('saving model in', filename)
    f = file(filename + '.pkl', 'wb')
    import sys
    sys.setrecursionlimit(100000)
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def load_model(file_name):
    f = file(file_name + '.pkl', 'rb')
    # theano.config.reoptimize_unpickled_function = False
    start = time.time()
    model = pickle.load(f)
    end = time.time()
    elapsed_time = end - start
    return model


def print_model(model, level=1):
    for i, l in enumerate(model.layers):
        indent = '  ' * level + '-'
        if type(l) == Sequential:
            logging.info('{} {} {} {}'.format(indent, i, l.name, l.output_shape))
            print_model(l, level + 1)
        else:
            logging.info('{} {} {} {}'.format(indent, i, l.name, l.output_shape))


def get_layers(model, level=1):
    layers = []
    for i, l in enumerate(model.layers):

        # indent = '  ' * level + '-'
        if type(l) == Sequential:
            layers.extend(get_layers(l, level + 1))
        else:
            layers.append(l)

    return layers


### 下面这个并非是主函数，我在这之所以这样写是为了方便两个文件彼此互相调用！
if __name__ == '__main__':
    from model.coef_weights_utils import get_gradient_weights, get_permutation_weights, get_weights_linear_model, \
        get_gradient_weights_with_repeated_output, get_weights_gradient_outcome, \
        get_deep_explain_scores, get_shap_scores, get_skf_weights
import numpy as np





### 选择一种可解释性方法来计算输入数据（主要是输入的训练数据）的重要性
def get_coef_importance(model, X_train, y_train, target, feature_importance, detailed=True, **kwargs):
    if feature_importance.startswith('skf'):
        coef_ = get_skf_weights(model, X_train, y_train, feature_importance)
        # pass
    elif feature_importance == 'loss_gradient':
        coef_ = get_gradient_weights(model, X_train, y_train, signed=False, detailed=detailed,
                                     normalize=True)  # use total loss
    elif feature_importance == 'loss_gradient_signed':
        coef_ = get_gradient_weights(model, X_train, y_train, signed=True, detailed=detailed,
                                     normalize=True)  # use total loss
    elif feature_importance == 'gradient_outcome':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=False, signed=False)
    elif feature_importance == 'gradient_outcome_signed':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target=target, detailed=detailed,
                                             multiply_by_input=False, signed=True)
    elif feature_importance == 'gradient_outcome*input':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=True, signed=False)
    elif feature_importance == 'gradient_outcome*input_signed':
        coef_ = get_weights_gradient_outcome(model, X_train, y_train, target, multiply_by_input=True, signed=True)

    elif feature_importance.startswith('deepexplain'):
        method = feature_importance.split('_')[1]
        coef_ = get_deep_explain_scores(model, X_train, y_train, target, method_name=method, detailed=detailed,
                                        **kwargs)

    elif feature_importance.startswith('shap'):
        method = feature_importance.split('_')[1]
        coef_ = get_shap_scores(model, X_train, y_train, target, method_name=method, detailed=detailed)


    elif feature_importance == 'gradient_with_repeated_outputs':
        coef_ = get_gradient_weights_with_repeated_output(model, X_train, y_train, target)
    elif feature_importance == 'permutation':
        coef_ = get_permutation_weights(model, X_train, y_train)
    elif feature_importance == 'linear':
        coef_ = get_weights_linear_model(model, X_train, y_train)
    elif feature_importance == 'one_to_one':
        weights = model.layers[1].get_weights()
        switch_layer_weights = weights[0]
        coef_ = np.abs(switch_layer_weights)
    else:
        coef_ = None
    return coef_


### 一堆模型顺序向量，上一个模型的输出作为当前这个模型的输入！
def apply_models(models, inputs):
    output = inputs
    for m in models:
        output = m(output)

    return output




def plot_channels(history, channels, filename, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plt.figure()
    for k in channels:
        v = history[k]
        plt.plot(v)
    plt.legend(channels)
    filename = os.path.join(folder_name, filename)
    plt.savefig(filename)
    plt.close()


def plot_history(history, folder_name):
    keys = list(history.keys())

    losses = [x for x in keys if ('_loss' in x) and (x != 'val_loss')]
    val_losses = [x for x in losses if 'val_' in x]
    train_losses = [x for x in losses if ('val_' not in x) and (x != 'loss')]
    # train_losses = [x.replace('val_', '') for x in val_losses ]

    monitors = [x for x in keys if 'loss' not in x]
    val_monitors = [x for x in monitors if 'val_' in x]
    train_monitors = [x for x in monitors if ('val_' not in x) and (x != 'loss') and (x != 'lr')]
    # train_monitors= [x.replace('val_', '') for x in val_monitors]

    monitors.sort()
    val_monitors.sort()
    train_monitors.sort()

    train_losses.sort()
    val_losses.sort()

    print("目前是model_utils.py文件，现在的这个验证损失是谁！", val_losses)
    print("目前是model_utils.py文件，现在的这个训练损失是谁！", train_losses)           ### 现在的这个训练损失是谁！ ['o1_loss', 'o2_loss', 'o3_loss', 'o4_loss', 'o5_loss', 'o6_loss']
    print("目前是model_utils.py文件，现在的这个monitors是谁！", monitors)              ### 现在的这个monitors是谁！ ['lr', 'o1_f1', 'o2_f1', 'o3_f1', 'o4_f1', 'o5_f1', 'o6_f1'

    plot_channels(history, val_monitors, 'val_monitors', folder_name)
    plot_channels(history, train_monitors, 'train_monitors', folder_name)
    # plot_channels(history, ['val_loss', 'loss'], 'loss')
    for v, t in zip(val_monitors, train_monitors):
        plot_channels(history, [v, t], t, folder_name)

    plot_channels(history, val_losses, 'validation_loss', folder_name)
    plot_channels(history, train_losses, 'training_loss', folder_name)

    if 'val_loss' in keys:
        plot_channels(history, ['val_loss', 'loss'], 'loss', folder_name)
    else:
        plot_channels(history, ['loss'], 'loss', folder_name)

    for v, t in zip(val_losses, train_losses):
        plot_channels(history, [v, t], t, folder_name)
    pass

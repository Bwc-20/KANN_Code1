### 这个是在2023.6.8将当前文件进行保存的。   此文件中核心是在写：根据当前的层的输出值来计算相应的梯度值，之后利用梯度值的大小来修改该层的输出值！
### 现在主要是在走GradientProcessingCallback(Callback)  这个类，现在主要是对当前的输出值无法进行修改！！

import logging
import warnings
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras import backend as K
# from tensorflow.keras import backend as K
from keras.callbacks import LambdaCallback
import tensorflow as tf
from keras.layers import Lambda
import keras
from keras.layers.core import Reshape



### 下面这个函数是新一个求网络梯度的一个函数，求出网络梯度值后并对其进行处理
class GradientModifier(Callback):

    def __init__(self, layer_name, X_train):
        super(GradientModifier, self).__init__()
        self.layer_name = layer_name
        # self.validation_data = X_train
        # self.BaiData = X_train

    def on_batch_end(self, batch, logs={}):

        # x, y = self.validation_data[:2]
        layer = self.model.get_layer(self.layer_name)
        gradients = K.gradients(self.model.total_loss, layer.output)[0]
        modify_func = K.function([self.model.input], [gradients])

        # layer_output, = modify_func([self.BaiData[0]])
        print("现在这个是来进行测试的！！当前的这个验证数据是谁！", type(self.validation_data), self.validation_data)
        layer_output, = modify_func([self.validation_data[0]])
        print("当前这个验证函数的输出结果是什么！", self.validation_data.shape, layer_output.shape[-1])
        for i in range(layer_output.shape[-1]):
            if layer_output[0, i] > 0:  # 梯度为正，强化节点
                layer_output[:, i] = layer_output[:, i] * 2
                print("现在是在model/callbacks_custom.py文件中，梯度为正的神经元节点的输出值将会被强化！")
            elif layer_output[0, i] < 0:  # 梯度为负，将输出置为0
                layer_output[:, i] = 0
                print("现在是在model/callbacks_custom.py文件中，梯度为负的神经元节点的输出值直接被清0！")
            else:  # 梯度为0，不做处理
                continue
        K.set_value(layer.output, layer_output)





# 自定义回调函数     现在这个回调函数是2023.6.6创建！！用以计算神经网络训练过程中神经元的梯度情况！
def adjust_output_by_gradient(model):
    def get_gradient_values(grads):
        """
        返回一个函数，用于获取各个神经元的梯度值
        """

        def get_grad_values(layer_outputs):
            return grads

        return get_grad_values

    # 获取梯度值并处理输出
    def on_batch_end(batch, logs={}):
        X_batch = logs['inputs']
        y_batch = logs['targets']

        # 获取各层梯度值
        get_gradient = K.function([model.input, model.output],
                                  K.gradients(model.output, model.layers[-1].output))
        grad_values = get_gradient([X_batch, y_batch])[0]

        # 调整输出值
        for layer in model.layers:
            if hasattr(layer, 'h0'):
                layer_output = layer.output
                # 如果梯度为负，则将输出清零
                neg_grad_mask = K.cast(grad_values < 0., K.floatx())
                layer_output *= neg_grad_mask

                # 对梯度值进行归一化处理
                grad_values /= (K.sqrt(K.mean(K.square(grad_values))) + K.epsilon())

                # 将梯度值与输出值相乘，以调整输出值
                layer_output *= grad_values

    return LambdaCallback(on_batch_end=get_gradient_values)





### 这个是来求模型中具体某一层网络的梯度情况！
def get_gradient_layer(model, X, y, layer, normalize=True):
    grad = model.optimizer.get_gradients(model.total_loss, layer)     ### 获取指定层的梯度情况！
    gradients = layer * grad  # gradient tensors

    # special case of repeated outputs (e.g. output for each hidden layer)
    if type(y) == list:
        n = len(y)
    else:
        n = 1
    input_tensors = [model.inputs[0],  # input data
                     # model.sample_weights[0],  # how much to weight each sample by
                     # model.targets[0:n],  # labels
                     # model.targets[0],  # labels
                     # K.learning_phase(),  # train or test mode
                     ]

    # how much to weight each sample by
    for i in range(n):
        input_tensors.append(model.sample_weights[i])
    # labels
    for i in range(n):
        input_tensors.append(model.targets[i])
    # train or test mode
    input_tensors.append(K.learning_phase())
    # normalize
    if normalize:
        gradients /= (K.sqrt(K.mean(K.square(gradients))) + 1e-5)

    get_gradients = K.function(inputs=input_tensors, outputs=[gradients])
    # https: // github.com / fchollet / keras / issues / 2226
    # print 'y_train', y_train.shape

    nb_sample = X.shape[0]

    inputs = [X,  # X
              # sample_weights,  # sample weights
              # y,  # y
              # 0  # learning phase in TEST mode
              ]

    for i in range(n):
        inputs.append(np.ones(nb_sample))

    if n > 1:
        for i in range(n):
            inputs.append(y[i].reshape((nb_sample, 1)))
    else:
        inputs.append(y.reshape(nb_sample, 1))

    inputs.append(0)  # learning phase in TEST mode
    gradients = get_gradients(inputs)
    print("看看此时这样是否还会出错！！", gradients)
    gradients = get_gradients(inputs)[0][0]
    return gradients




# 定义一个Lambda层用于修改输出
def multiply_output(x, multiplier):
    return tf.multiply(x, multiplier)

###  第三种求梯度并进行处理的方法！
class GradientProcessingCallback(Callback):
    def __init__(self, layer_name, X_train, y_train):
        super(GradientProcessingCallback, self).__init__()
        self.layer_name = layer_name
        self.layer_index = 0
        self.X = X_train
        self.Y = y_train

    def on_batch_end(self, batch, logs={}):
        layer_output = self.model.get_layer(self.layer_name).output
        # gradients = K.gradients(self.model.total_loss, layer_output)[0]
        gradients = K.gradients(self.model.total_loss, layer_output)


        ### 另一种的梯度计算方法
        this_Layer = self.model.get_layer(self.layer_name).get_output_at(0)
        gradients = self.model.optimizer.get_gradients(self.model.total_loss, this_Layer)[0]  ### 获取指定层的梯度情况！
        # l.get_output_at(0)

        # gradients = get_gradient_layer(self.model, self.X, self.Y, this_Layer, normalize=True)



        print("那么此时的这个梯度情况是怎样的！", type(gradients), gradients)
        print("看一下此时这个梯度值与网络输出值的形状是否相同！", layer_output.shape == self.model.get_layer(self.layer_name).output.shape, gradients.shape == layer_output.shape, gradients.shape, layer_output.shape, gradients, layer_output)
        gradients /= (K.sqrt(K.mean(K.square(gradients))) + K.epsilon())
        print("现在看一下在经过计算之后的这个梯度情况！这个梯度值与网络输出值的形状是否相同！", layer_output.shape == self.model.get_layer(self.layer_name).output.shape, gradients.shape == layer_output.shape, gradients.shape, layer_output.shape, gradients,
              layer_output)

        # outputs, _ = self.model.predict_on_batch(self.model.input)
        # processed_outputs = outputs * K.relu(gradients)

        ### 在这修改一下，在求得梯度之后，换种新的方法来对网络的输出值进行处理！
        # # 如果梯度为负，则将输出清零
        # neg_grad_mask = K.cast(gradients < 0., K.floatx())
        # layer_output *= neg_grad_mask
        # 对梯度值进行处理 如果梯度为负，则将输出清零
        grad_values_normalise = K.relu(gradients)      ### 此时，这个gradients已经是经过归一化之后的，现在再进行Relu操作，将负梯度置为0
        # 将梯度值与输出值相乘，以调整输出值


        # grad_values_normalise = Reshape((1, 9229))(grad_values_normalise)  ### 修改一下当前梯度的形状！
        # print("现在重塑形状之后的这个情况是怎样的！", grad_values_normalise.shape)
        # tf.squeeze(grad_values_normalise, 0)     ### 先删掉他的第一维
        # print("现在，删掉第一维之后的情况是怎样的！", grad_values_normalise.shape, grad_values_normalise)
        # tf.expand_dims(grad_values_normalise, 0)     ### 再为他增加第一维
        # print("之后，再增加第一维之后的情况是怎样的！", grad_values_normalise.shape, grad_values_normalise)
        # print("此时是grad_values_normalise数据，他当前tile函数之前这个数据的维度是怎样的！", grad_values_normalise.shape, grad_values_normalise)
        # grad_values_normalise = tf.tile(grad_values_normalise, [tf.shape(self.model.get_layer(self.layer_name).output)[0], 1])  ### 此时就说明将weights这个数据分别在各个维度上进行扩充，其中第一维扩充的倍数为tf.shape(outcome)[0]，第二维扩充的倍数为1（相当于在这第二维不变）



        print("修改形状之后梯度形状是否一样！！", layer_output.shape == self.model.get_layer(self.layer_name).output.shape, layer_output.shape == grad_values_normalise.shape, self.model.get_layer(self.layer_name).output.shape == grad_values_normalise.shape)



        layer_output *= grad_values_normalise

        # for i in range(len(grad_values_normalise)):
        #     layer_output[:, i] = layer_output[:, i]*grad_values_normalise[i]

        # layer_output = multiply_output(layer_output, grad_values_normalise)





        # # Compute the element-wise product of the layer output and the normalised gradient values
        # product = layer_output * grad_values_normalise
        # # Copy the original layer output tensor and reshape it to the shape of the product tensor
        # # layer_output_copy = K.reshape(K.copy(layer_output), K.int_shape(product))
        # print("现在这几项数据的数据类型是怎样的！", type(K.int_shape(product)), type(tf.identity(layer_output)))
        # print("当前的这几项元组情况是怎样的！！", K.int_shape(product), tf.identity(layer_output))
        # # product_shape = K.int_shape(product)
        # # product_shape = tf.cast(product_shape, dtype=tf.float32)
        # layer_output_copy = tf.reshape(tf.identity(layer_output), K.int_shape(product))
        # # layer_output_copy = K.reshape(K.tensorflow_backend.tf.identity(layer_output), K.int_shape(product))
        # # Perform the element-wise multiplication in place
        # K.update(layer_output_copy, layer_output_copy * grad_values_normalise)







        processed_outputs = layer_output

        print("修改形状之后梯度形状是否一样！！", layer_output.shape == self.model.get_layer(self.layer_name).output.shape, layer_output.shape == grad_values_normalise.shape, self.model.get_layer(self.layer_name).output.shape == processed_outputs.shape)


        # # 构建一个新模型，该模型的输出值为修改后的输出值
        # layer_input = self.model.get_layer(self.layer_name).input
        # temp_model = keras.engine.Layer(inputs=layer_input, outputs=processed_outputs)
        #
        # # 将新模型的权重设置为原模型的权重
        # temp_model.set_weights(self.model.get_weights())
        # # 将原模型的输出值替换为新模型的输出值
        # self.model.layers[self.layer_name].output = temp_model.output




        print("现在这个模型的输出值的情况是怎样的！", self.model.get_layer(self.layer_name).output.shape, self.model.get_layer(self.layer_name).output)
        print("现在求出来的这个梯度值是怎样的！", processed_outputs.shape, processed_outputs)
        print("测试一下目前这两个数据的形状是否相同！！", layer_output.shape == processed_outputs.shape, self.model.get_layer(self.layer_name).output.shape == processed_outputs.shape)

        processed_outputs = np.array(processed_outputs)
        # # print("现在的这个数据的情况是怎样的！", type(processed_outputs), processed_outputs)
        # # output_var = K.variable(processed_outputs)
        # # K.assign(self.model.get_layer(self.layer_name).output, output_var)
        #
        # # processed_outputs = processed_outputs.astype(layer_output.dtype)
        # # processed_outputs = np.zeros_like(layer_output, dtype=layer_output.dtype)
        # processed_outputs = np.zeros_like(np.atleast_1d(layer_output))
        # processed_outputs[:] = layer_output * grad_values_normalise
        #
        #
        # K.set_value(self.model.get_layer(self.layer_name).output, processed_outputs)








        # ### 因为这一层网络的输出结果不能进行修改，因此在该网络之后又构建了一个网络对其输出值进行修改
        # output_multiplier_layer = Lambda(lambda x: multiply_output(x, grad_values_normalise), name='modify_layer_output')(layer_output)
        # self.model.layers[self.layer_index + 1] = output_multiplier_layer
        # # self.model.layers.insert(self.layer_index + 1, output_multiplier_layer)   # 插入一个Lambda层，用于将当前层的输出与参数相乘
        print("当前这个网络模型的大体情况！", len(self.model.layers))



        # print("当前这一层网络的索引号是谁！", self.model.get_layer(self.layer_name).index)
        # print("现在这个模型的输出值的情况是怎样的！", self.model.get_layer(self.layer_name).output)
        # print("现在求出来的这个梯度值是怎样的！", grad_values_normalise)
        # K.set_value(self.model.get_layer(self.layer_name).output, processed_outputs)
        # self.model.get_layer(self.layer_name).output *= grad_values_normalise


        self.model.get_layer(self.layer_name).output = processed_outputs







##******** 现在下面的这个是23.6.8晚20点创建，是一种克服神经网络输出赋值的方法，现在这个方法时会报错：AttributeError: 'list' object has no attribute 'dtype'    因此试一下另一种方法
###  第三种求梯度并进行处理的方法！
class GradientProcessingCallback_Two(Callback):
    def __init__(self, layer_name, X_train, y_train):
        super(GradientProcessingCallback_Two, self).__init__()
        self.layer_name = layer_name
        self.layer_index = 0
        self.X = X_train
        self.Y = y_train

    def on_batch_end(self, batch, logs={}):
        layer_output = self.model.get_layer(self.layer_name).output
        # gradients = K.gradients(self.model.total_loss, layer_output)[0]
        gradients = K.gradients(self.model.total_loss, layer_output)


        ### 另一种的梯度计算方法
        this_Layer = self.model.get_layer(self.layer_name).get_output_at(0)
        gradients = self.model.optimizer.get_gradients(self.model.total_loss, this_Layer)[0]  ### 获取指定层的梯度情况！
        # l.get_output_at(0)

        # gradients = get_gradient_layer(self.model, self.X, self.Y, this_Layer, normalize=True)



        # print("那么此时的这个梯度情况是怎样的！", type(gradients), gradients)
        # print("看一下此时这个梯度值与网络输出值的形状是否相同！", layer_output.shape == self.model.get_layer(self.layer_name).output.shape, gradients.shape == layer_output.shape, gradients.shape, layer_output.shape, gradients, layer_output)
        # gradients /= (K.sqrt(K.mean(K.square(gradients))) + K.epsilon())
        # print("现在看一下在经过计算之后的这个梯度情况！这个梯度值与网络输出值的形状是否相同！", layer_output.shape == self.model.get_layer(self.layer_name).output.shape, gradients.shape == layer_output.shape, gradients.shape, layer_output.shape, gradients,
        #       layer_output)
        #
        # ### 在这修改一下，在求得梯度之后，换种新的方法来对网络的输出值进行处理！
        # # # 如果梯度为负，则将输出清零
        # # neg_grad_mask = K.cast(gradients < 0., K.floatx())
        # # layer_output *= neg_grad_mask
        # # 对梯度值进行处理 如果梯度为负，则将输出清零
        # grad_values_normalise = K.relu(gradients)      ### 此时，这个gradients已经是经过归一化之后的，现在再进行Relu操作，将负梯度置为0
        # # 将梯度值与输出值相乘，以调整输出值
        #
        #
        #
        # print("修改形状之后梯度形状是否一样！！", layer_output.shape == self.model.get_layer(self.layer_name).output.shape, layer_output.shape == grad_values_normalise.shape, self.model.get_layer(self.layer_name).output.shape == grad_values_normalise.shape)
        #
        #
        #
        # ### 这块这几个是新加的！！
        # grad_fun = K.function([self.model.inputs[0], K.learning_phase()], [this_Layer, gradients])
        # layer_output_value, gradients_value = grad_fun([self.X, 0])
        # processed_layer_output = layer_output_value * gradients_value
        # processed_outputs = [processed_layer_output]
        # grad_layer_fun = K.function([self.model.inputs[0], K.learning_phase()], processed_outputs)
        # self.model.get_layer(self.layer_name).output = processed_outputs


        ### 又一种的梯度计算方法
        # 创建计算梯度和处理输出的函数
        # get_output_and_gradients = K.function(
        #     inputs=[self.model.input],
        #     outputs=[self.model.get_layer(self.layer_name).output, K.gradients(self.model.total_loss, this_Layer)[0]]
        # )

        # o1_sample_weights = tf.placeholder_with_default(input=[1.0], shape=[None], name='o1_sample_weights')
        # sample_weights_value = np.ones((len(self.X),))






        # get_output_and_gradients = K.function(
        #     inputs=[self.model.input, self.model.targets, K.learning_phase(), self.model.sample_weights],
        #     outputs=[self.model.get_layer(self.layer_name).output, K.gradients(self.model.total_loss, this_Layer)[0]]
        # )
        #
        # # 调用函数获取指定层的输出和梯度
        # print("目前这个输入输出数据的形状是怎样的！", self.X.shape, self.Y.shape)
        # # self.model.sample_weights = None
        # # self.model.targets = None
        # # inputs = [self.X, self.Y, 0, self.model.sample_weights]                    ## [self.X, None, 0, self.model.sample_weights]
        # # inputs = [np.array(x) if isinstance(x, list) else x for x in inputs]
        # # 计算样本权重
        # sample_weights = compute_sample_weight(class_weight='balanced', y=self.Y)
        #
        # # sample_weights = np.ones((self.X.shape[0],))
        #
        # y_test = np.array(self.Y)
        # sample_weights = np.array(sample_weights)
        # # sample_weights = np.reshape(sample_weights, (self.X.shape[0], 1))
        #
        # for i in range(len(self.X)):
        #     self.X[i] = np.array(self.X[i])
        #     self.Y[i] = np.array(self.Y[i])
        #     sample_weights[i] = np.array(sample_weights[i])
        #
        # print("那么现在的这个权重是怎样的！", sample_weights)
        # print("目前的这个数据类型是怎样的！！", type(y_test), type(sample_weights), type(self.X), type(self.X[0]), type(self.Y[0]), type(sample_weights[0]))
        # # inputs = tuple([self.X, None, 0, sample_weights])
        #
        # Result = get_output_and_gradients([self.X, self.Y, 0, sample_weights])
        # outputs, gradients = Result[0], Result[1]




        # 创建计算梯度和处理输出的函数                  self.model.optimizer.get_gradients(self.model.total_loss, this_Layer)[0]
        # get_output_and_gradients = K.function(
        #     inputs=[self.model.input, self.model.targets],
        #     outputs=[self.model.get_layer(self.layer_name).output, K.gradients(self.model.total_loss, this_Layer)[0]]
        # )



        self.Y = self.Y.astype('float32')
        self.X = self.X.astype('float32')
        get_output_and_gradients = K.function(
            inputs=[self.model.input, self.model.targets],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.optimizer.get_gradients(self.model.total_loss, this_Layer)[0]]
        )

        # 调用函数获取指定层的输出和梯度
        outputs, gradients = get_output_and_gradients([self.X, self.Y])
        # outputs, gradients = get_output_and_gradients([self.X, self.Y])[0], get_output_and_gradients([self.X, self.Y])[1]




        # 处理梯度并与原输出相乘得到处理后的输出
        gradients /= (K.sqrt(K.mean(K.square(gradients))) + K.epsilon())
        grad_values_normalise = K.relu(gradients)
        outputs *= grad_values_normalise

        # 手动赋值给指定层的输出
        self.model.get_layer(self.layer_name).output = outputs




















# http://alexadam.ca/ml/2018/08/03/early-stopping.html
class FixedEarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitors: quantities to be monitored.
        min_deltas: minimum change in the monitored quantities
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        modes: list of {auto, min, max}. In `min` mode,
            training will stop when the quantities
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baselines: Baseline values for the monitored quantities to reach.
            Training will stop if the model doesn't show improvement
            for at least one of the baselines.
    """



    """自定义回调
当一个被监测的数量停止改进时，停止训练。
    # 参数
        monitors: 要监测的数量。
        min_deltas：被监测数量的最小变化。
            的最小变化，也就是说，绝对变化小于min_delta，将被视为改进。
            小于min_delta的绝对变化，将被视为没有改善。
            变化。
        patience（耐心）：没有改进的历时数。
            之后，训练将被停止。
        verbose：粗略的模式。
        modes：{自动、最小、最大}的列表。在 最小 模式下。
            训练将在所监测的数量
            训练将停止；在`最大`模式下，训练将在监测的数量不再减少时停止。
            模式下，训练将在监测到的数量停止减少时停止；在`max`模式下，训练将在监测到的数量
            在`min`模式下，训练将在监测的数量停止减少时停止；在`max`模式下，训练将在监测的数量停止增加时停止。
            模式下，方向是自动推断出来的
            在 "自动 "模式下，方向是由监测到的数量的名称自动推断出来的。
        基线。监测数量要达到的基线值。
            如果模型没有显示出改善，训练将停止。
            至少有一个基线没有显示出改进。"""




    def __init__(self,
                 monitors=['val_loss'],
                 min_deltas=[0],
                 patience=0,
                 verbose=0,
                 modes=['auto'],
                 baselines=[None]):
        super(FixedEarlyStopping, self).__init__()

        self.monitors = monitors
        self.baselines = baselines
        self.patience = patience
        self.verbose = verbose
        self.min_deltas = min_deltas
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_ops = []

        for i, mode in enumerate(modes):
            if mode not in ['auto', 'min', 'max']:
                warnings.warn('EarlyStopping mode %s is unknown, '
                              'fallback to auto mode.' % mode,
                              RuntimeWarning)
                modes[i] = 'auto'

        for i, mode in enumerate(modes):
            if mode == 'min':
                self.monitor_ops.append(np.less)
            elif mode == 'max':
                # self.monitor_ops.append(np.greater)
                self.monitor_ops.append(np.greater_equal)
            else:
                if 'acc' in self.monitors[i]:
                    self.monitor_ops.append(np.greater)

                else:
                    self.monitor_ops.append(np.less)

        for i, monitor_op in enumerate(self.monitor_ops):
            if monitor_op == np.greater:
                self.min_deltas[i] *= 1
            else:
                self.min_deltas[i] *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.waits = []
        self.stopped_epoch = 0
        self.bests = []

        for i, baseline in enumerate(self.baselines):
            if baseline is not None:
                self.bests.append(baseline)
            else:
                self.bests.append(np.Inf if self.monitor_ops[i] == np.less else -np.Inf)

            self.waits.append(0)

    def on_epoch_end(self, epoch, logs=None):
        reset_all_waits = False
        for i, monitor in enumerate(self.monitors):
            current = logs.get(monitor)

            if current is None:
                warnings.warn(
                    'Early stopping conditioned on metric `%s` '
                    'which is not available. Available metrics are: %s' %
                    (monitor, ','.join(list(logs.keys()))), RuntimeWarning
                )
                return

            if self.monitor_ops[i](current - self.min_deltas[i], self.bests[i]):
                self.bests[i] = current
                self.waits[i] = 0
                reset_all_waits = True
            else:
                self.waits[i] += 1

        if reset_all_waits:
            for i in range(len(self.waits)):
                self.waits[i] = 0

            return

        num_sat = 0
        for wait in self.waits:
            if wait >= self.patience:
                num_sat += 1

        if num_sat == len(self.waits):
            self.stopped_epoch = epoch
            self.model.stop_training = True

        print((self.waits))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(('Epoch %05d: early stopping' % (self.stopped_epoch + 1)))


class GradientCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, gradient_function, x_train, y_train, max_epoch, feature_names=None, monitor='val_loss',
                 verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=10):
        super(GradientCheckpoint, self).__init__()
        self.monitor = monitor
        self.feature_names = feature_names
        self.gradient_function = gradient_function
        self.x_train = x_train
        self.y_train = y_train
        self.verbose = verbose
        n = len(self.feature_names)
        self.history = [[] for i in range(n)]
        self.max_epoch = max_epoch
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        is_last_epoch = (self.max_epoch - epoch - 1) < self.period
        # is_last_epoch = self.max_epoch == epoch
        print(('is_last_epoch', is_last_epoch))
        if (self.epochs_since_last_save >= self.period) or (epoch == 0):
            self.epochs_since_last_save = 0
            logging.info('getting gradient')
            coef_ = self.gradient_function(self.model, self.x_train, self.y_train)
            if type(coef_) != list:
                coef_ = [coef_]
            # for i, c in enumerate(coef_):
            #     df = pd.DataFrame(c)
            #     logging.info('saving gradient epoch {} layer {}'.format(epoch, i))
            #     f= '{} epoch {} layer {} .csv'.format(self.filepath , str(epoch), str(i) )
            #     df.to_csv(f)
            i = 0

            for c, names in zip(coef_, self.feature_names):
                print(i)
                print((c.shape))
                print((len(names)))
                df = pd.DataFrame(c.ravel(), index=names, columns=[str(epoch)])
                self.history[i].append(df)
                # logging.info('saving gradient epoch {} layer {}'.format(epoch, i))
                # f= '{} epoch {} layer {} .csv'.format(self.filepath , str(epoch), str(i) )
                # df.to_csv(f)
                i += 1

            # save
            if is_last_epoch:
                logging.info('saving gradient')
                for i, h in enumerate(self.history):
                    df = pd.concat(h, axis=1)
                    f = '{} layer {} .csv'.format(self.filepath, str(i))
                    df.to_csv(f)

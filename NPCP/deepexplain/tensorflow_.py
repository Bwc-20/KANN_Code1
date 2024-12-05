



import sys
import warnings
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from skimage.util import view_as_windows
# from tensorflow.python.framework import ops
import ops
# from tensorflow.python.ops import nn_grad, math_grad
from tensorflow.python.ops import nn_grad, math_grad

SUPPORTED_ACTIVATIONS = ['Relu', 'Elu', 'Sigmoid', 'Tanh', 'Softplus']

UNSUPPORTED_ACTIVATIONS = [
    'CRelu', 'Relu6', 'Softsign'
]

_ENABLED_METHOD_CLASS = None
_GRAD_OVERRIDE_CHECKFLAG = 0


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------


def activation(type):
    """
    Returns Tensorflow's activation op, given its type
    :param type: string
    :return: op
    """
    if type not in SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % type)
    f = getattr(tf.nn, type.lower())
    return f


def original_grad(op, grad):
    """
    Return original Tensorflow gradient for an op
    :param op: op
    :param grad: Tensor
    :return: Tensor
    """
    if op.type not in SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % op.type)
    opname = '_%sGrad' % op.type
    if hasattr(nn_grad, opname):
        f = getattr(nn_grad, opname)
    else:
        f = getattr(math_grad, opname)
    return f(op, grad)


# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS BASE CLASSES
# -----------------------------------------------------------------------------


class AttributionMethod(object):
    """
    Attribution method base class
    """

    def __init__(self, T, X, inputs, xs, session, keras_learning_phase=None):
        self.T = T
        self.inputs = inputs
        self.X = X
        self.xs = xs
        self.session = session
        self.keras_learning_phase = keras_learning_phase
        # print (self.X)
        # print (self.inputs)
        self.has_multiple_inputs = type(self.X) is list or type(self.X) is tuple
        # self.has_multiple_inputs= False
        # self.has_multiple_inputs = type(self.inputs) is list or type(self.inputs) is tuple
        # print ('Model with multiple inputs: ', self.has_multiple_inputs, self.inputs)

    def session_run(self, T, xs):
        feed_dict = {}
        print("路漫漫其修远兮！测试一下，看看目前AttributionMethod这个类走进来没！")         ### 现在T就是所传进来的梯度信息
        if self.has_multiple_inputs:
            print('现在是在deepexplain/tensorflow_.py文件下,   has_multiple_inputs')
            # if len(xs) != len(self.X):
            if len(xs) != len(self.inputs):
                raise RuntimeError('List of input tensors and input data have different lengths (%s and %s)'
                                   # % (str(len(xs)), str(len(self.X))))
                                   % (str(len(xs)), str(len(self.inputs))))
            # for k, v in zip(self.X, xs):
            for k, v in zip(self.inputs, xs):
                feed_dict[k] = np.float32(v)
        else:
            # feed_dict[self.X] = xs
            feed_dict[self.inputs] = xs

        if self.keras_learning_phase is not None:
            feed_dict[self.keras_learning_phase] = 0
        # print ('debugging')
        # print ('# of keys in feeding dict {}'.format( len(feed_dict.keys())))
        # print (self.has_multiple_inputs, T,feed_dict )
        # print ('feed_dict', feed_dict )
        for key, value in feed_dict.items():
            if type(value) == np.ndarray:
                print("目前是在tensorflow_.py这个文件中，目前这些key和value的情况！", key, type(value), value.shape, value.dtype)

        # print("现在来测试一下传进来的梯度以及feed_dict信息！", feed_dict)
        return self.session.run(T, feed_dict)

    def _set_check_baseline(self):
        xss = self.xs
        # xss= self.session_run(self.X, self.xs)
        print('xss {}, xs {}'.format(xss.shape, self.xs.shape))
        if self.baseline is None:
            if self.has_multiple_inputs:
                self.baseline = [np.zeros((1,) + xi.shape[1:]) for xi in xss]
            else:
                self.baseline = np.zeros((1,) + xss.shape[1:])
        else:
            if self.has_multiple_inputs:
                for i, xi in enumerate(self.xs):
                    if self.baseline[i].shape == xss[i].shape[1:]:
                        self.baseline[i] = np.expand_dims(self.baseline[i], 0)
                    else:
                        raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                           % (self.baseline[i].shape, self.xs[i].shape[1:]))
            else:
                if self.baseline.shape == xss.shape[1:]:
                    self.baseline = np.expand_dims(self.baseline, 0)
                else:
                    raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                       % (self.baseline.shape, self.xs.shape[1:]))


class GradientBasedMethod(AttributionMethod):
    """
    Base class for gradient-based attribution methods        基于梯度的归因方法的基类
    """

    def get_symbolic_attribution(self):
        print('现在是在deepexplain/tensorflow_.py文件下,   hello from symbolic attribution')
        # gradients= K.gradients(self.T, self.X)
        # grad = K.function(inputs=self.inputs, outputs=gradients)
        gradients = [g for g in tf.gradients(self.T, self.X)]
        return gradients

    def run(self):
        attributions = self.get_symbolic_attribution()
        print("目前是在 tensorflow_.py 文件中，现在用tensorflow求出来的梯度属性信息是！", attributions)     ### 现在所求出来的这个attributions就是输入以及中间隐藏层中的各个神经元节点他们各自的重要性分数！
        results = self.session_run(attributions, self.xs)
        print("目前是在 tensorflow_.py 文件中， 测试部分！白文超必胜！")
        return results[0] if not self.has_multiple_inputs else results

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return original_grad(op, grad)


class PerturbationBasedMethod(AttributionMethod):
    """
       Base class for perturbation-based attribution methods    基于扰动的归因方法的基类
       """

    def __init__(self, T, X, inputs, xs, session, keras_learning_phase):
        super(PerturbationBasedMethod, self).__init__(T, X, inputs, xs, session, keras_learning_phase)
        self.base_activation = None

    def _run_input(self, x):
        return self.session_run(self.T, x)

    def _run_original(self):
        return self._run_input(self.xs)

    def run(self):
        raise RuntimeError('Abstract: cannot run PerturbationBasedMethod')


# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS
# -----------------------------------------------------------------------------
"""
Returns zero attributions. For testing only.
"""


class DummyZero(GradientBasedMethod):

    def get_symbolic_attribution(self, ):
        return tf.gradients(self.T, self.X)

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        input = op.inputs[0]
        return tf.zeros_like(input)


"""
Saliency maps
https://arxiv.org/abs/1312.6034
"""


class Saliency(GradientBasedMethod):

    def get_symbolic_attribution(self):       ## 获取符号化属性！
        return [tf.abs(g) for g in tf.gradients(self.T, self.X)]


"""
Gradient * Input
https://arxiv.org/pdf/1704.02685.pdf - https://arxiv.org/abs/1611.07270
"""


### 下面的这个类继承自GradientBasedMethod这个类（而GradientBasedMethod又继承自AttributionMethod这个类）
### 下面这个类的核心思想还是DeepLIFT算法的思想，就是这个位置的乘数乘上这个位置的输入即为这个位置的重要性分数   只是在这他是用梯度信息来作为具体位置的乘数！
### 在这的各个位置的梯度直接用tensorflow求出来！
class GradientXInput(GradientBasedMethod):

    def get_symbolic_attribution(self):
        print('现在是在deepexplain/tensorflow_.py文件下,   hello from GradientXInput')
        # gradients =  [self.X*g for g in K.gradients(self.T, self.X)]
        gradients = [self.X * g for g in tf.gradients(self.T, self.X)]          ### 在这里，g就是用对应的tensorflow方法求出来的各个位置的梯度信息，而最终求出来的这个gradients就是最终各个位点的重要性分数（用此位置的输入乘以此位置的梯度）
        # gradients =   K.gradients(self.T, self.X)

        # gradients = tf.gradients(self.T, self.X)
        print("现在是在deepexplain/tensorflow_.py文件下,  现在来测试一下这个self.T, self.X, 以及gradients的相关信息：", self.T, self.X, gradients)    ### 现在来求出各层他的梯度信息！
        #
        # # grad = K.function(inputs=self.inputs, outputs=gradients)
        # ret = K.function(inputs=[self.inputs], outputs=gradients)
        # print('ret', ret)
        # if self.has_multiple_inputs:
        #     ret =[g * x for g, x in zip(grad,self.X)]
        # else:
        #     ret= [self.X]

        # gradients= [g * x for g, x in zip(
        #     tf.gradients(self.T, self.X),
        #     self.X if self.has_multiple_inputs else [self.X])]

        return gradients


"""
Integrated Gradients
https://arxiv.org/pdf/1703.01365.pdf
"""


class IntegratedGradients(GradientBasedMethod):

    def __init__(self, T, X, input, xs, session, keras_learning_phase, steps=10, baseline=None):
        super(IntegratedGradients, self).__init__(T, X, input, xs, session, keras_learning_phase)
        self.steps = steps
        self.baseline = baseline

    def run(self):
        # Check user baseline or set default one
        self._set_check_baseline()

        attributions = self.get_symbolic_attribution()
        gradient = None
        for alpha in list(np.linspace(1. / self.steps, 1.0, self.steps)):
            xs_mod = [b + (xs - b) * alpha for xs, b in zip(self.xs, self.baseline)] if self.has_multiple_inputs \
                else self.baseline + (self.xs - self.baseline) * alpha
            _attr = self.session_run(attributions, xs_mod)
            # print ('attributions',attributions)
            _attr = self.session_run(attributions, self.xs)
            xss = self.session_run(self.X, self.xs)
            if gradient is None:
                gradient = _attr
            else:
                gradient = [g + a for g, a in zip(gradient, _attr)]

        # layer_baseline = self.baseline
        layer_baseline = self.session_run(self.X, self.baseline)
        # xss = self.xs
        xss = self.session_run(self.X, self.xs)
        if self.has_multiple_inputs:
            results = [g * (x - b) / self.steps for g, x, b in zip(gradient, xss, layer_baseline)]
        else:
            print('现在是在deepexplain/tensorflow_.py文件下,  self.xs {}, self.baseline  {}, gradient {} {}'.format(xss.shape, layer_baseline.shape, len(gradient),
                                                                         gradient[0].shape))
            results = [g * (x - b) / self.steps for g, x, b in zip(gradient, [xss], [layer_baseline])]

        # results = [g * (x - b) / self.steps for g, x, b in zip(
        #     gradient,
        #     self.xs if self.has_multiple_inputs else [self.xs],
        #     self.baseline if self.has_multiple_inputs else [self.baseline])]

        return results[0] if not self.has_multiple_inputs else results


"""
Layer-wise Relevance Propagation with epsilon rule
http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
"""


class EpsilonLRP(GradientBasedMethod):
    eps = None

    def __init__(self, T, X, inputs, xs, session, keras_learning_phase, epsilon=1e-4):
        super(EpsilonLRP, self).__init__(T, X, inputs, xs, session, keras_learning_phase)
        assert epsilon > 0.0, 'LRP epsilon must be greater than zero'
        global eps
        eps = epsilon

    def get_symbolic_attribution(self):
        # debugging
        x = self.X if self.has_multiple_inputs else [self.X]
        print(x)
        import keras.backend as K
        grad = tf.gradients(K.mean(self.T), self.X)
        print(grad)
        z = list(zip(grad, x))
        print(z)
        ret = [g * x for g, x in z]
        return ret

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        return grad * output / (input + eps *
                                tf.where(input >= 0, tf.ones_like(input), -1 * tf.ones_like(input)))


"""
DeepLIFT
This reformulation only considers the "Rescale" rule
https://arxiv.org/abs/1704.02685
"""


### 在这，我使用Rescale规则进行实现DeepLIft算法！
class DeepLIFTRescale(GradientBasedMethod):
    _deeplift_ref = {}

    def __init__(self, T, X, inputs, xs, session, keras_learning_phase, baseline=None):
        super(DeepLIFTRescale, self).__init__(T, X, inputs, xs, session, keras_learning_phase)
        self.baseline = baseline
        # self.baseline_layer = baseline

    def get_symbolic_attribution(self):
        # layer_baseline =  self.baseline
        layer_baseline = self.session_run(self.X, self.baseline)          ### DeepLIFT算法的基本策略！先找一个基本结果作为参照！  用这个layer_baseline来作为网络中每个位置的基本输入，参照结果！
        if self.has_multiple_inputs:
            ret = [g * (x - b) for g, x, b in zip(tf.gradients(self.T, self.X), self.X, layer_baseline)]
        else:
            ret = [g * (x - b) for g, x, b in zip(tf.gradients(self.T, self.X), [self.X], [layer_baseline])]      ### 在这g * (x - b) 就是用梯度乘上输入的变化量   最后求得的这个ret就是各个位置的重要性贡献分数
        return ret
        # [g * (x - b) for g, x, b in zip(
        # tf.gradients(self.T, self.X),
        # self.X if self.has_multiple_inputs else [self.X],
        # self.baseline if self.has_multiple_inputs else [self.baseline])]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        ref_input = cls._deeplift_ref[op.name]          ### 来构建参照的输入输出！
        ref_output = activation(op.type)(ref_input)
        delta_out = output - ref_output
        delta_in = input - ref_input
        instant_grad = activation(op.type)(0.5 * (ref_input + input))
        return tf.where(tf.abs(delta_in) > 1e-5, grad * delta_out / delta_in,
                        original_grad(instant_grad.op, grad))         ## 在这这个tf.where(condition, x=None, y=None, name=None) 函数  该函数的作用是根据condition,返回相对应的x或y,返回值是一个tf.bool类型的Tensor

    def run(self):
        # Check user baseline or set default one
        self._set_check_baseline()

        # Init references with a forward pass
        self._init_references()

        # Run the default run
        return super(DeepLIFTRescale, self).run()

    def _init_references(self):
        sys.stdout.flush()
        self._deeplift_ref.clear()
        ops = []
        g = self.session.graph
        # get subgraph starting from the target node down
        subgraph = tf.graph_util.extract_sub_graph(g.as_graph_def(), [self.T.name.split(':')[0]])

        for n in subgraph.node:
            op = g.get_operation_by_name(n.name)
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in SUPPORTED_ACTIVATIONS:
                    ops.append(op)
                    print("现在是位于tensorflow_.py文件，这个操作的名字为：", op.name)

        ins = [o.inputs[0] for o in ops]
        print('现在是处于tensorflow_.py文件中，现在各个的输入情况  ins', ins)
        YR = self.session_run(ins, self.baseline)
        for (r, op) in zip(YR, ops):
            self._deeplift_ref[op.name] = r
        sys.stdout.flush()


"""
Occlusion method
Generalization of the grey-box method presented in https://arxiv.org/pdf/1311.2901.pdf
This method performs a systematic perturbation of contiguous hyperpatches in the input,
replacing each patch with a user-defined value (by default 0).

window_shape : integer or tuple of length xs_ndim
Defines the shape of the elementary n-dimensional orthotope the rolling window view.
If an integer is given, the shape will be a hypercube of sidelength given by its value.

step : integer or tuple of length xs_ndim
Indicates step size at which extraction shall be performed.
If integer is given, then the step is uniform in all dimensions.
"""


class Occlusion(PerturbationBasedMethod):

    def __init__(self, T, X, inputs, xs, session, keras_learning_phase, window_shape=None, step=None):
        super(Occlusion, self).__init__(T, X, inputs, xs, session, keras_learning_phase)
        if self.has_multiple_inputs:
            raise RuntimeError('Multiple inputs not yet supported for perturbation methods')

        input_shape = xs[0].shape
        # input_shape = xs.shape
        if window_shape is not None:
            assert len(window_shape) == len(input_shape), \
                'window_shape must have length of input (%d)' % len(input_shape)
            self.window_shape = tuple(window_shape)
        else:
            self.window_shape = (1,) * len(input_shape)

        if step is not None:
            assert isinstance(step, int) or len(step) == len(input_shape), \
                'step must be integer or tuple with the length of input (%d)' % len(input_shape)
            self.step = step
        else:
            self.step = 1
        self.replace_value = 0.0
        print('现在是在deepexplain/tensorflow_.py文件下,  Input shape: %s; window_shape %s; step %s' % (input_shape, self.window_shape, self.step))

    def run(self):
        self._run_original()

        input_shape = self.xs.shape[1:]
        batch_size = self.xs.shape[0]
        total_dim = np.asscalar(np.prod(input_shape))

        # Create mask
        index_matrix = np.arange(total_dim).reshape(input_shape)
        idx_patches = view_as_windows(index_matrix, self.window_shape, self.step).reshape((-1,) + self.window_shape)
        heatmap = np.zeros_like(self.xs, dtype=np.float32).reshape((-1), total_dim)
        w = np.zeros_like(heatmap)

        # Compute original output
        eval0 = self._run_original()
        num_patches = len(idx_patches)
        # Start perturbation loop
        for i, p in enumerate(idx_patches):
            print('{}/{}'.format(i, num_patches))
            mask = np.ones(input_shape).flatten()
            mask[p.flatten()] = self.replace_value
            masked_xs = mask.reshape((1,) + input_shape) * self.xs
            delta = eval0 - self._run_input(masked_xs)
            delta_aggregated = np.sum(delta.reshape((batch_size, -1)), -1, keepdims=True)
            heatmap[:, p.flatten()] += delta_aggregated
            w[:, p.flatten()] += p.size

        attribution = np.reshape(heatmap / w, self.xs.shape)
        if np.isnan(attribution).any():
            warnings.warn('Attributions generated by Occlusion method contain nans, '
                          'probably because window_shape and step do not allow to cover the all input.')
        return attribution


# -----------------------------------------------------------------------------
# END ATTRIBUTION METHODS
# -----------------------------------------------------------------------------


attribution_methods = OrderedDict({
    'zero': (DummyZero, 0),
    'saliency': (Saliency, 1),
    'grad*input': (GradientXInput, 2),
    'intgrad': (IntegratedGradients, 3),
    'elrp': (EpsilonLRP, 4),
    'deeplift': (DeepLIFTRescale, 5),
    'occlusion': (Occlusion, 6)
})


# @ops.RegisterGradient("DeepExplainGrad")
@tf.RegisterGradient("DeepExplainGrad")
def deepexplain_grad(op, grad):
    global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
    _GRAD_OVERRIDE_CHECKFLAG = 1
    if _ENABLED_METHOD_CLASS is not None \
            and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
        return _ENABLED_METHOD_CLASS.nonlinearity_grad_override(op, grad)
    else:
        return original_grad(op, grad)


class DeepExplain(object):

    def __init__(self, graph=None, session=tf.get_default_session()):
        self.method = None
        self.batch_size = None
        self.session = session
        self.graph = session.graph if graph is None else graph       ### 在这创建tensorflow的数据流图！
        print('graph', self.graph)
        # op = session.graph.get_operations()
        # for m in op:
        #     print (m.values())

        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self.get_override_map())
        self.keras_phase_placeholder = None
        self.context_on = False
        if self.session is None:
            raise RuntimeError('DeepExplain: could not retrieve a session. Use DeepExplain(session=your_session).')

    def __enter__(self):
        # Override gradient of all ops created in context   重写上下文中创建的所有操作的梯度
        self.graph_context.__enter__()
        self.override_context.__enter__()
        self.context_on = True
        return self

    def __exit__(self, type, value, traceback):
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)
        self.context_on = False

    def explain(self, method, T, X, inputs, xs, **kwargs):
        print('现在是在deepexplain/tensorflow_.py文件下,  hello from deep explain')
        if not self.context_on:
            raise RuntimeError('Explain can be called only within a DeepExplain context.')
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
        print("正气公心！加油文超！现在是在tensorflow_.py文件中，目前传进来的这个可解释方法是！", method)
        self.method = method
        if self.method in attribution_methods:
            method_class, method_flag = attribution_methods[self.method]
        else:
            raise RuntimeError('Method must be in %s' % list(attribution_methods.keys()))
        print('现在是在deepexplain/tensorflow_.py文件下,  DeepExplain: running "%s" explanation method (%d)' % (self.method, method_flag))
        self._check_ops()
        _GRAD_OVERRIDE_CHECKFLAG = 0

        _ENABLED_METHOD_CLASS = method_class
        print("现在是在deepexplain/tensorflow_.py文件下, 测一下，现在的这个_ENABLED_METHOD_CLASS是谁！", _ENABLED_METHOD_CLASS)
        method = _ENABLED_METHOD_CLASS(T, X, inputs, xs, self.session, self.keras_phase_placeholder, **kwargs)
        print("现在是在deepexplain/tensorflow_.py文件下, 测一下，现在的这个method是谁！", method)
        result = method.run()
        # print("现在是在deepexplain/tensorflow_.py文件下, 测试一下，目前的这个可解释性算法的输出结果是！", result)        ### 这个是对当前这一层的解释结果！
        if issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod) and _GRAD_OVERRIDE_CHECKFLAG == 0:
            warnings.warn('DeepExplain detected you are trying to use an attribution method that requires '
                          'gradient override but the original gradient was used instead. You might have forgot to '
                          '(re)create your graph within the DeepExlain context. Results are not reliable!')
            """
            覆盖上下文中创建的所有操作的梯度
            DeepExplain检测到你正试图使用一个需要覆盖梯度的归属方法 
            梯度覆盖，但却使用了原始梯度。你可能忘记了 
            (在DeepExlain上下文中（重新）创建你的图形。结果不可靠!
            """
        _ENABLED_METHOD_CLASS = None
        _GRAD_OVERRIDE_CHECKFLAG = 0
        self.keras_phase_placeholder = None
        return result

    @staticmethod
    def get_override_map():
        return dict((a, 'DeepExplainGrad') for a in SUPPORTED_ACTIVATIONS)

    def _check_ops(self):
        """
        Heuristically check if any op is in the list of unsupported activation functions.
        This does not cover all cases where explanation methods would fail, and must be improved in the future.
        Also, check if the placeholder named 'keras_learning_phase' exists in the graph. This is used by Keras
         and needs to be passed in feed_dict.     启发式地检查任何操作是否在不支持的激活函数列表中。这并不包括所有解释方法会失败的情况，未来必须加以改进。另外，检查图中是否存在名为 "keras_learning_phase "的占位符。这是由Keras使用的 使用的，需要在feed_dict中传递。
        :return:
        """
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in UNSUPPORTED_ACTIVATIONS:
                    warnings.warn('Detected unsupported activation (%s). '
                                  'This might lead to unexpected or wrong results.' % op.type)
            elif 'keras_learning_phase' in op.name:
                self.keras_phase_placeholder = op.outputs[0]

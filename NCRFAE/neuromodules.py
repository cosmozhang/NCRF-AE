'''
Cosmo Zhang @ Purude
neuromodules.py
neuro layer toolkit in theano
'''

import theano
import theano.tensor as T
import numpy as np

class LogisticRegression(object):
    """
    Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """
        Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)

        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.hard_sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)+0.01
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class ConvPoolLayer(object):
    """2D Pool Layer of a convolutional network"""

    def __init__(self, rng, input, input_shape, filter_shape=(3, 2, 2), poolsize=(2, 2), activation=T.tanh):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.tensor3
        :param input: symbolic input tensor, of shape input_shape

        :type input_shape: tuple or list of length 3
        :param input_shape: (input height, input width)

        :type filter_shape: tuple or list of length 3
        :param filter_shape: (number of filters, filter height, filter width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        self.input = input

        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.poolsize = poolsize
        self.activation = activation
        # there are "filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[1:]) /np.prod(poolsize))
        # initialize weights with random weights
        if self.activation=="none" or self.activation=="relu":
            self.W = theano.shared(np.asarray(rng.uniform(low=-0.01, high=0.01, size=filter_shape), dtype=theano.config.floatX), borrow=True, name="W_conv")
        else:
            self.W = theano.shared(np.asarray(rng.uniform(low=-np.sqrt(6. / (fan_in + fan_out)), high=np.sqrt(6. / (fan_in + fan_out)), size=filter_shape), dtype=theano.config.floatX), borrow=True, name="W_conv")

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")

        # convolve input feature maps with filters
        conv_out = theano.tensor.signal.conv.conv2d(input=self.input, filters=self.W, filter_shape=self.filter_shape)
        # print conv_out.tag.test_value.shape
        # print self.input_shape[0] - self.filter_shape[1] + 1, self.input_shape[1] - self.filter_shape[2] + 1

        conv_out_next = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # print conv_out_next.tag.test_value.shape

        self.output = T.signal.pool.pool_2d(input=conv_out_next, ds=self.poolsize, ignore_border=True).flatten(2)
        # print self.output.tag.test_value.shape

        pool_outsize = (self.input_shape[0] - self.filter_shape[1] + 1)/self.poolsize[0]*(self.input_shape[1] - self.filter_shape[2] + 1)/self.poolsize[1]
        self.out_dim = self.filter_shape[0]*pool_outsize
        # print (self.input_shape[0] - self.filter_shape[1] + 1)/self.poolsize[0], (self.input_shape[1] - self.filter_shape[2] + 1)/self.poolsize[1]
        # print 'in conv', self.out_dim

        self.params = [self.W, self.b]

class DropoutLayer(object):
    def __init__(self, rng, input, dropout_rate=0.5):
        """
        input: output of last layer
        """
        self.input = input
        self.dropout_rate = dropout_rate

        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

        if self.dropout_rate > 0:
            # p=1-p because 1's indicate keep and p is prob of dropping
            mask = srng.binomial(n=1, p = 1-self.dropout_rate, size=self.input.shape)
            # The cast is important because
            # int * float32 = float64 which pulls things off the gpu
            self.output = self.input * T.cast(mask, theano.config.floatX)
        else:
            self.output = input

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, dropout_rate = 0.5, W=None, b=None, activation=T.tanh):
        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.hard_sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        output = (
            lin_output if activation is None
            else activation(lin_output))

        srng = T.shared_randomstreams.RandomStreams(
                rng.randint(999999))
        # p=1-p because 1's indicate keep and p is prob of dropping
        mask = srng.binomial(n=1, p = 1-dropout_rate, size=output.shape)
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        self.output = output * T.cast(mask, theano.config.floatX)

        # parameters of the model
        self.params = [self.W, self.b]

class ElmanRnnLayer(object):

    def __init__(self, rng, inputSeq, n_in, n_out, Wx = None, Wh = None, b = None, activation = T.nnet.hard_sigmoid, bptt_truncate = -1, retrun_final_only = False, batch_mode = False):

        if Wx is None:
            Wx = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.hard_sigmoid:
                Wx *= 4

            Wx = theano.shared(value=Wx, name='Wx', borrow=True)
        self.Wx = Wx

        if Wh is None:
            Wh = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_out + n_out)),
                    high=np.sqrt(6. / (n_out + n_out)),
                    size=(n_out, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.hard_sigmoid:
                Wh *= 4

            Wh = theano.shared(value=Wh, name='Wh', borrow=True)
        self.Wh = Wh

        if b is None:
            b_values = np.zeros((n_out), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='bh', borrow=True)
        self.b = b

        h0 = theano.shared(np.zeros(n_out, dtype=theano.config.floatX))

        if batch_mode:
            self.h0 = T.alloc(h0, inputSeq.shape[1], inputSeq.shape[2], n_out)
        else:
            self.h0 = h0

        def recurrence(x_t, h_tm1):
            h_t = activation(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)  # hidden layer
            return h_t

        self.bptt_truncate = bptt_truncate
        hSeq, _ = theano.scan(fn=recurrence,
                                # s_t is computed by h_t, so no inistial valuse
                                # is in need
                                sequences=inputSeq,
                                outputs_info=[self.h0],
                                # no non_sequences is needed here
                                truncate_gradient=self.bptt_truncate)

        if retrun_final_only:
            self.output = hSeq[-1]
        else:
            self.output = hSeq

        self.params = [self.Wx, self.Wh, self.h0, self.b]
        # self.params = [self.Wx, self.Wh, self.b]

class LstmLayer(object):

    def __init__(self, rng, inputSeq, n_in, n_out, U = None, W = None, b = None, activation = T.tanh, bptt_truncate = -1, retrun_final_only = False, batch_mode = False):

        # LSTM parameters
        if U is None:
            U = np.random.uniform(-np.sqrt(6./n_out), np.sqrt(6./n_out), (4, n_in, n_out))
            if activation == T.nnet.hard_sigmoid:
                U *= 4
        self.U = theano.shared(name = 'U', value = U.astype(theano.config.floatX))

        if W is None:
            W = np.random.uniform(-np.sqrt(6./n_out), np.sqrt(6./n_out), (4, n_out, n_out))
            if activation == T.nnet.hard_sigmoid:
                W *= 4
        self.W = theano.shared(name = 'W', value = W.astype(theano.config.floatX))

        # b: bias
        if b is None:
            b = np.zeros((4, n_out))
        self.b = theano.shared(name = 'b', value = b.astype(theano.config.floatX))

        self.bptt_truncate = bptt_truncate

        def recurrence(x_t, h_t_prev, c_t_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # LSTM Layer 1
            i_t = T.nnet.hard_sigmoid(T.dot(x_t, self.U[0]) + T.dot(h_t_prev, self.W[0]) + self.b[0])
            f_t = T.nnet.hard_sigmoid(T.dot(x_t, self.U[1]) + T.dot(h_t_prev, self.W[1]) + self.b[1])
            o_t = T.nnet.hard_sigmoid(T.dot(x_t, self.U[2]) + T.dot(h_t_prev, self.W[2]) + self.b[2])
            g_t = activation(T.dot(x_t, self.U[3]) + T.dot(h_t_prev, self.W[3]) + self.b[3])

            c_t = c_t_prev * f_t + g_t * i_t

            h_t = T.tanh(c_t) * o_t

            return [h_t, c_t]

        h0 = theano.shared(np.zeros(n_out, dtype=theano.config.floatX))
        c0 = theano.shared(np.zeros(n_out, dtype=theano.config.floatX))

        if batch_mode:
            self.h0 = T.alloc(h0, inputSeq.shape[1], inputSeq.shape[2], n_out)
            self.c0 = T.alloc(c0, inputSeq.shape[1], inputSeq.shape[2], n_out)
        else:
            self.h0 = h0
            self.c0 = c0

        [hSeq, cSeq], _ = theano.scan(
            fn = recurrence,
            sequences=inputSeq,
            outputs_info=[self.h0, self.c0],
            truncate_gradient=self.bptt_truncate)

        if retrun_final_only:
            self.output = hSeq[-1]
        else:
            self.output = hSeq

        # bundle
        self.params = [self.U, self.W, h0, c0, self.b]
        # self.params = [self.U, self.W, self.b]

class DropoutLstmLayer(object):

    def __init__(self, rng, inputSeq, n_in, n_out, dropout_rate = 0.5, U = None, W = None, b = None, activation = T.tanh, bptt_truncate = -1, retrun_final_only = False):

        # LSTM parameters
        if U is None:
            U = np.random.uniform(-np.sqrt(6./n_out), np.sqrt(6./n_out), (4, n_in, n_out))
            if activation == T.nnet.hard_sigmoid:
                U *= 4
        self.U = theano.shared(name = 'Uh', value = U.astype(theano.config.floatX))

        if W is None:
            W = np.random.uniform(-np.sqrt(6./n_out), np.sqrt(6./n_out), (4, n_out, n_out))
            if activation == T.nnet.hard_sigmoid:
                W *= 4
        self.W = theano.shared(name = 'Wh', value = W.astype(theano.config.floatX))

        # b: bias
        if b is None:
            b = np.zeros((4, n_out))
        self.b = theano.shared(name = 'bh', value = b.astype(theano.config.floatX))

        self.bptt_truncate = bptt_truncate
        srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        dropoutSeq = T.cast(srng.binomial(n=1, p = 1-dropout_rate, size=(inputSeq.shape[0], inputSeq.shape[1], n_out)), theano.config.floatX)

        def recurrence(x_t, mask_t, h_t_prev, c_t_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # LSTM Layer 1
            # x_t is a matrix
            i_t = T.nnet.hard_sigmoid(T.dot(x_t, self.U[0]) + T.dot(h_t_prev, self.W[0]) + self.b[0])
            f_t = T.nnet.hard_sigmoid(T.dot(x_t, self.U[1]) + T.dot(h_t_prev, self.W[1]) + self.b[1])
            o_t = T.nnet.hard_sigmoid(T.dot(x_t, self.U[2]) + T.dot(h_t_prev, self.W[2]) + self.b[2])
            g_t = activation(T.dot(x_t, self.U[3]) + T.dot(h_t_prev, self.W[3]) + self.b[3])

            c_t = c_t_prev * f_t + g_t * i_t

            output = T.tanh(c_t) * o_t

            h_t = output * mask_t

            return [h_t, c_t]

        self.h0 = T.zeros((inputSeq.shape[1], n_out), dtype=theano.config.floatX)
        self.c0 = T.zeros((inputSeq.shape[1], n_out), dtype=theano.config.floatX)

        [hSeq, cSeq], _ = theano.scan(
            fn = recurrence,
            sequences=[inputSeq, dropoutSeq],
            outputs_info=[self.h0, self.c0],
            truncate_gradient=self.bptt_truncate)
        # hSeq is a 3d tensor
        if retrun_final_only:
            self.output = hSeq[-1]
        else:
            self.output = hSeq

        # bundle
        self.params = [self.U, self.W, self.b]

class GruLayer(object):

    def __init__(self, rng, inputSeq, n_in, n_out, U = None, W = None, b = None, activation = T.tanh, bptt_truncate = -1, retrun_final_only = False):


        # GRU parameters
        if U is None:
            U = np.random.uniform(-np.sqrt(6./n_out), np.sqrt(6./n_out), (3, de * cs, n_out))
            if activation == T.nnet.hard_sigmoid:
                U *= 4
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))

        if W is None:
            W = np.random.uniform(-np.sqrt(6./n_out), np.sqrt(6./n_out), (3, n_out, n_out))
            if activation == T.nnet.hard_sigmoid:
                W *= 4
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))

        b = np.zeros((3, n_out))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))

        self.bptt_truncate = bptt_truncate

        def recurrence(x_t, h_t_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # GRU Layer 1
            z_t = T.nnet.hard_sigmoid(T.dot(x_t, self.U[0]) + T.dot(h_t_prev, self.W[0]) + self.b[0])
            r_t = T.nnet.hard_sigmoid(T.dot(x_t, self.U[1]) + T.dot(h_t_prev, self.W[1]) + self.b[1])
            c_t = activation(T.dot(x_t, self.U[2]) + T.dot(h_t_prev * r_t, self.W[2]) + self.b[2])
            h_t = (T.ones_like(z_t) - z_t) * c_t + z_t * h_t_prev

            return h_t

        self.h0 = theano.shared(np.zeros(n_out, dtype=theano.config.floatX))

        hSeq, _ = theano.scan(
            fn = recurrence,
            sequences = inputSeq,
            outputs_info = [self.h0],
            truncate_gradient = self.bptt_truncate)

        if retrun_final_only:
            self.output = hSeq[-1]
        else:
            self.output = hSeq

        # bundle
        self.params = [self.U, self.W, self.h0, self.b]




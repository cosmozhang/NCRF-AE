'''
Cosmo Zhang @ Purude
nnoptimizer.py
neuro optimizer toolkit in theano
'''

from collections import OrderedDict
import theano
import theano.tensor as T
import numpy as np

def SGD(cost, params, learning_rate = 0.002):

    updates = OrderedDict()
    grads = T.grad(cost, params)

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    return updates

def MomentumSGD(cost, params, learning_rate = 0.002, momentum=0.9):
    updates = SGD(cost, params, learning_rate)
    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param]
        updates[velocity] = x - param
        updates[param] = x

    return updates

def NesterovMomentumSGD(cost, params, learning_rate = 0.002, momentum=0.9):
    updates = SGD(cost, params, learning_rate)
    for param in params:
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
        x = momentum * velocity + updates[param] - param
        updates[velocity] = x
        updates[param] = momentum * x + updates[param]

    return updates

def Adam(cost, params, learning_rate=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = OrderedDict()
    grads = T.grad(cost, params)
    i = theano.shared(np.asarray(0., dtype=theano.config.floatX))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = learning_rate * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)

        updates[m] = m_t
        updates[v] = v_t
        updates[p] = p_t
    updates[i] = i_t

    return updates

def AdaGrad(cost, params, learning_rate=1.0, epsilon=1e-6):
    updates = OrderedDict()
    grads = T.grad(cost, params)
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates

def RmsProp(cost, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    updates = OrderedDict()
    grads = T.grad(cost, params)
    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates

def Adadelta(cost, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):

    updates = OrderedDict()
    grads = T.grad(cost, params)

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        # accu: accumulate gradient magnitudes
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        # delta_accu: accumulate update magnitudes (recursively!)
        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

        # update accu (as in rmsprop)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = (grad * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        updates[param] = param - learning_rate * update

        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates

def Adamax(cost, params, learning_rate=0.002, beta1=0.9,
           beta2=0.999, epsilon=1e-8):

    updates = OrderedDict()
    grads = T.grad(cost, params)
    t_prev = theano.shared(np.asarray(0., dtype=theano.config.floatX))

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate/(one-beta1**t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        u_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (one-beta1)*g_t
        u_t = T.maximum(beta2*u_prev, abs(g_t))
        step = a_t*m_t/(u_t + epsilon)

        updates[m_prev] = m_t
        updates[u_prev] = u_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates

def EGD(cost, params, learning_rate = 0.33, constraint = 1.0):

    updates = OrderedDict()

    grads = T.grad(cost, params)
    U = T.constant(constraint)

    #first half of params
    rw_pos = T.exp(-learning_rate * U * grads[0])
    rb_pos = T.exp(-learning_rate * U * grads[1])

    #second half
    rw_neg = 1/rw_pos
    rb_neg = 1/rb_pos

    rs = [rw_pos, rb_pos, rw_neg, rb_neg]

    partition = T.sum(params[0]*rs[0]) + T.sum(params[1]*rs[1]) + T.sum(params[2]*rs[2]) + T.sum(params[3]*rs[3])

    for param, r in zip(params, rs):
        updates[param] = U*param*r/partition

    return updates

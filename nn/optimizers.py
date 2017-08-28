from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np

dtype = theano.config.floatX
if theano.config.floatX == 'float32':
    floatX = np.float32
    intX = np.int32
else:
    floatX = np.float64
    intX = np.int64


def rmsprop(cost, params, learning_rate, rho=0.9, epsilon=1e-6):
    updates = list()

    for param in params:
        accu = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=dtype),
                broadcastable=param.broadcastable)

        grad = T.grad(cost, param)
        accu_new = rho * accu + (1 - rho) * grad ** 2

        updates.append((accu, accu_new))
        unclipped_grad = grad / T.sqrt(accu_new + epsilon)
        #unclipped_l2norm = T.sqrt(T.sum(T.sqr(unclipped)))
        #clipped_grad = unclipped * (1.0 / unclipped_l2norm)
        updates.append((param, param - (learning_rate * unclipped_grad)))

    return updates

def sgd_clipped(cost, params, learning_rate):
    grads = [T.grad(cost, param) for param in params]
    gradl2s = [T.sum(T.sqr(grad)) for grad in grads]
    return [(param, param - learning_rate * grad * (1.0 / gradl2)) for param,grad,gradl2 in zip(params, grads, gradl2s)]

def sgd(cost, params, learning_rate):
    return [(param, param - learning_rate * T.grad(cost, param)) for param in params]

def momentum(cost, params, learning_rate, momentum=0.9, type='nesterov'):
    assert type in ['std', 'nesterov'], 'Possible momentum types: `std`, `nesterov`'
    assert 0 <= momentum < 1
    updates = list()

    for param in params:
        # this is the "momentum" part: it is shared across updates
        velocity = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=dtype),
                broadcastable=param.broadcastable)

        update = param - learning_rate * T.grad(cost, param)
        if type == 'nesterov': # nesterov
            x = momentum * velocity + update - param
            updates.append((velocity, x))
            updates.append((param, momentum * x + update))
        else: # standard
            x = momentum * velocity + update
            updates.append((velocity, x - param))
            updates.append((param, x))

    return updates

def adam(cost, params, learning_rate=0.0002, b1=0.1, b2=0.001, e=1e-8):
    lr = learning_rate
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
        updates.append((i, i_t))
    return updates


if __name__ == '__main__':
    import gru as rnn

    rng = np.random.RandomState(42)

    n_in, n_out = 10, 1

    X, y, output, params = rnn.generate_rnn(n_in, n_out, 50)
    output = output[-1, :]

    # minimize binary crossentropy
    xent = -y * T.log(output) - (1 - y) * T.log(1 - output)
    cost = xent.mean()

    # minimize mean squared error (don't get a nan error this way)
    # cost = 0.5 * ((y - output) ** 2).sum()

    lr = T.scalar(name='lr', dtype=dtype)

    # grads = [T.grad(cost=cost, wrt=p) for p in params]
    updates = rmsprop(cost, params, lr)

    t_sets = 10
    X_datas = [np.asarray(rng.rand(20, n_in) > 0.5, dtype=dtype) for _ in range(t_sets)]
    y_datas = [np.asarray(rng.rand(1, n_out) > 0.5, dtype=dtype) for _ in range(t_sets)]

    train = theano.function([X, y, lr], [cost], updates=updates)
    test = theano.function([X], [output])

    n_train = 1000
    l = 0.1

    cost = sum([train(X_data, y_data, 0)[0] for X_data, y_data in zip(X_datas, y_datas)])
    print('Before training:', cost)

    for i in range(n_train):
        for X_data, y_data in zip(X_datas, y_datas):
            train(X_data, y_data, l)

        if (i+1) % 10 == 0:
            cost = sum([train(X_data, y_data, 0)[0] for X_data, y_data in zip(X_datas, y_datas)])
            print('%d:' % (i+1), cost)

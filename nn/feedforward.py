#!/usr/bin/env python
import numpy as np
from optimizers import rmsprop
import theano
import theano.tensor as T
import json


__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    intX = np.int32
    floatX = np.float32
else:
    intX = np.int64
    floatX = np.float64

def _get_weights(name, shape1, shape2, init='nestrov'):
    if init == 'rand':
        x = np.random.rand(shape1, shape2) 
    elif init == 'nestrov':
        x = np.random.uniform(-np.sqrt(1. / shape2), np.sqrt(1. / shape2), (shape1, shape2))
    else:
        raise NotImplementedError("don't know how to initialize the weight matrix")
    return theano.shared(floatX(x), name)

def _get_zeros(name, shape1):
    x = 0.0 * np.random.rand(shape1,) 
    return theano.shared(floatX(x), name)


class FeedForward(object):
    def __init__(self, n_in, n_hidden, reg, saved_weights = None):
        self.n_out = 2
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.reg = reg
        self.params = []
        self.reg_params = []
        self._update = rmsprop
        if saved_weights is None:
            if self.n_hidden > 0:
                w_in_hidden = _get_weights('W_i_h', self.n_in, self.n_hidden)
                b_hidden = _get_zeros('B_h', n_hidden)
                w_hidden_out = _get_weights('W_h_o', self.n_hidden, self.n_out)
                b_out = _get_zeros('B_o', self.n_out)
                self.params += [w_in_hidden, b_hidden, w_hidden_out, b_out]
                self.reg_params += [w_in_hidden,  w_hidden_out]
            else:
                w_in_out= _get_weights('W_i_o', self.n_in, self.n_out)
                b_out = _get_zeros('B_o', self.n_out)
                self.params += [w_in_out, b_out]
                self.reg_params += [w_in_out]
        else:
            _params = [floatX(np.asarray(i)) for i in json.loads(open(saved_weights, 'r').read())]
            if len(_params) == 2:
                w_in_out = theano.shared(_params[0], name = 'W_i_o')
                b_out = theano.shared(_params[1], name = 'B_o')
                self.params += [w_in_out, b_out]
                self.reg_params += [w_in_out]
            elif len(_params) == 4:
                w_in_hidden = theano.shared(_params[0], 'W_i_h')
                b_hidden = theano.shared(_params[1], 'B_h')
                w_hidden_out = theano.shared(_params[2], 'W_h_o')
                b_out = theano.shared(_params[3], 'B_o')
                self.params += [w_in_hidden, b_hidden, w_hidden_out, b_out]
                self.reg_params += [w_in_out]
            else:
                raise "Unknown number of params"
        self._make_graph()

    def save_weights(self, save_path):
        _params = json.dumps([i.tolist() for i in self.get_params_t()])
        f = open(save_path, 'w')
        f.write(_params)
        f.flush()
        f.close()
        return _params

    def _make_graph(self,):
        lr = T.scalar('lr', dtype=theano.config.floatX)
        X = T.fmatrix('X') #(batch_size, n_in)
        Y = T.ivector('Y') #(batch_size,)
        if self.n_hidden > 0:
            w_in_hidden = self.params[0]
            b_hidden = self.params[1]
            w_hidden_out = self.params[2]
            b_out = self.params[3]
            h = T.tanh(X.dot(w_in_hidden) + b_hidden)
            py_given_x = T.nnet.softmax(h.dot(w_hidden_out) + b_out)
            #xent = -T.mean(T.log(py_given_x[T.arange(Y.shape[0]), Y]))
            #py_given_x = T.nnet.sigmoid(h.dot(w_hidden_out) + b_out)
            xent = -T.mean(T.log(py_given_x[T.arange(Y.shape[0]), Y]))
        else:
            w_in_out = self.params[0]
            b_out = self.params[1]
            py_given_x = T.nnet.softmax(X.dot(w_in_out) + b_out)
            xent = -T.mean(T.log(py_given_x[T.arange(Y.shape[0]), Y]))
            #py_given_x = T.nnet.sigmoid(X.dot(w_in_out) + b_out)
        reg_loss = 0.0
        p1 = T.switch(T.gt(py_given_x[:,1], 0.5), 1, 0)
        matches = T.sum(T.switch(T.eq(p1, Y), 1, 0)) / Y.shape[0]
        for rp in self.reg_params:
            reg_loss += T.sum(T.sqr(rp))
        loss = xent + self.reg * reg_loss
        self.get_matches = theano.function(inputs = [X,Y], outputs = matches)
        self.get_model_loss = theano.function(inputs = [X, Y], outputs = xent)
        self.do_update = theano.function(inputs = [X, Y, lr], 
                outputs= [loss, xent], 
                updates = self._update(loss, self.params, lr))
        self.get_params_t = theano.function(inputs = [], outputs = [T.as_tensor_variable(p) for p in self.params])


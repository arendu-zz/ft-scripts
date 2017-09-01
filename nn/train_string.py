#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import numpy as np
import codecs
import theano
from feedforward import FeedForward
import time
import argparse
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

if theano.config.floatX == 'float32':
    intX = np.int32
    floatX = np.float32
else:
    intX = np.int64
    floatX = np.float64

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('-t', action='store' , dest='training_data', required = True, help='prefix of the training data features')
    opt.add_argument('-f', action='store' , dest='feat_set', required = True, help="comma seperated feature set eg. 1,2,3")
    opt.add_argument('-r', action='store', dest='reg', type=float, default = 0.0)
    opt.add_argument('--nh', action='store', dest='hidden_size', type=int, default=0)
    opt.add_argument('--save', action='store', dest='save_params', required = True)
    options = opt.parse_args()
    print 'reading data...'
    DATA_Y = np.array([int(line.strip()) for line in codecs.open(options.training_data + '0', 'r', 'utf-8').readlines()], dtype=intX)
    f_list = []
    for f in options.feat_set.split(','):
        _f =np.array([[float(i) for i in line.split()] for line in codecs.open(options.training_data + f, 'r',  'utf-8').readlines()])
        f_list.append(_f)
    DATA = floatX(np.concatenate(f_list, axis=1))
    data_rows = len(DATA)
    assert data_rows == DATA_Y.shape[0]

    shuffle_ids = np.random.choice(xrange(data_rows), data_rows, False)
    #test_ids = shuffle_ids[:int(0.25 * data_rows)]
    dev_ids = shuffle_ids[:int(0.3 * data_rows)]
    train_ids = shuffle_ids[int(0.3 * data_rows):]
    batch_size = 128 
    print 'initializing network...'
    ff = FeedForward(n_in = len(DATA[0]), n_hidden = options.hidden_size, reg =  options.reg)
    start_time = time.time()
    dev_batches = np.array_split(dev_ids, int(len(dev_ids)/batch_size))
    dev_ave_loss = 0.
    dev_ave_acc = 0.
    for dev_batch_ids in dev_batches:
        X = DATA[dev_batch_ids]
        Y = DATA_Y[dev_batch_ids]
        dev_batch_loss = ff.get_model_loss(X, Y)
        dev_acc = ff.get_matches(X, Y)
        if np.isnan(dev_batch_loss):
            raise Exception("DEV loss is nan")
        dev_ave_loss += ((1.0 / len(dev_batches)) * dev_batch_loss)
        dev_ave_acc += ((1.0 / len(dev_batches)) * dev_acc)

    print 'initial dev loss', dev_ave_loss , 'initial dev accuracy', dev_ave_acc
    print 'training...'
    prev_dev_loss = np.inf
    prev_dev_acc = 0.
    for epoch_idx in xrange(50):
        lr = 0.001
        ave_loss = 0.
        dev_ave_loss = 0.
        dev_ave_acc = 0.
        train_ids = np.random.choice(train_ids, len(train_ids), False)
        train_batches = np.array_split(train_ids, int(len(train_ids)/batch_size))
        #loop over training examples
        for batch_ids in train_batches:
            X = DATA[batch_ids]
            Y = DATA_Y[batch_ids]
            _, batch_loss = ff.do_update(X, Y, lr)
            if np.isnan(batch_loss):
                raise Exception("TRAIN loss is nan")
            ave_loss += ((1.0 / len(train_batches)) * batch_loss)

        #loop over dev examples
        for dev_batch_ids in dev_batches:
            X = DATA[dev_batch_ids]
            Y = DATA_Y[dev_batch_ids]
            dev_batch_loss = ff.get_model_loss(X, Y)
            dev_acc = ff.get_matches(X, Y)
            if np.isnan(dev_batch_loss):
                raise Exception("DEV loss is nan")
            dev_ave_loss += ((1.0 / len(dev_batches)) * dev_batch_loss)
            dev_ave_acc += ((1.0 / len(dev_batches)) * dev_acc)

        dev_ave_acc += ((1.0 / len(dev_batches)) * dev_acc)
        imp = 'worse' if (dev_ave_loss > prev_dev_loss)  else 'continue'
        print epoch_idx,'train loss:', '%.4f'%ave_loss, 'dev loss:', '%.4f'%dev_ave_loss, 'dev accuracy:', '%.4f'%dev_ave_acc, imp
        prev_dev_loss = dev_ave_loss
        prev_dev_acc = dev_ave_acc
        if imp == 'continue':
            ff.save_weights(options.save_params) 
        else:
            break;
    print time.time() - start_time, 'seconds'

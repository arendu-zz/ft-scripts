#!/usr/bin/env python
__author__ = 'arenduchintala'
import numpy as np
import argparse
import json
from embed_utils import CombinedEmbeddings
import editdistance as ed

NULL = 'Null'

class FeedForward(object):
    def __init__(self, dim,  f_type, saved_weights, ET):
        self.et = ET #embedd tools
        self.dim = dim
        self.f_type = sorted([int(f) for f in f_type.split(',')])
        print self.f_type
        _params = [np.float32(np.asarray(i)) for i in json.loads(open(saved_weights, 'r').read())]
        if len(_params) == 2:
            self.n_hidden = 0
            self.n_in = _params[0].shape[0]
            self.w_in_out = _params[0]
            self.b_out = _params[1]
        elif len(_params) == 4:
            self.n_hidden = _params[1].shape[0]
            self.n_in = _params[0].shape[0]
            self.w_in_hidden = _params[0]
            self.b_hidden = _params[1]
            self.w_hidden_out = _params[2]
            self.b_out = _params[3]
        else:
            raise "Unknown number of params"
        print 'loaded params', self.n_in, self.n_hidden, 2

    def edit_sim(self, w1, w2):
        if w1 == 'Null' or w2 == 'Null':
            return 0.0
        else:
            return 1.0 - ((float(ed.eval(w1, w2))) / max(len(w1), len(w2)))

    def vecs(self, w):
        if w == NULL:
            w_vec = np.ones(self.dim)
            w_norm = 1.0
        else:
            w_vec, w_norm = self.et.get_vec(w, 1)
        return w_vec, w_norm

    def get_feats(self, w1, w2):
        #w1, w2 = (w1, w2) if np.random.rand() < 0.5 else (w2, w1)
        final_f = []
        if 1 in self.f_type:
            w1_isNull = 1.0 if w1 == NULL else 0.0
            w2_isNull = 1.0 if  w2 == NULL else 0.0
            f1 = np.array([w1_isNull, w2_isNull])
            final_f.append(f1)
        else:
            pass

        if 2 in self.f_type: 
            w1_vec, w1_norm = self.vecs(w1)
            w2_vec, w2_norm = self.vecs(w2)
            cs = w1_vec.dot(w2_vec) / (w1_norm * w2_norm)
            f2 = np.array([cs])
            final_f.append(f2)
        else:
            pass
        
        if 3 in self.f_type:
            es = self.edit_sim(w1, w2) 
            f3 = np.array([es])
            final_f.append(f3)
        else:
            pass

        if 4 in self.f_type:
            l1 = len(w1) if w1 != NULL else 0
            l2 = len(w2) if w2 != NULL else 0
            f4 = np.array([np.abs(float(l1 - l2)/10.0), float(l1)/10.0, float(l2)/10.0]) 
            final_f.append(f4)
        else:
            pass

        if 5 in self.f_type: 
            prod_w1_w2 = w1_vec * w2_vec
            prod_w1_w2_norm = np.linalg.norm(prod_w1_w2)
            prod_w1_w2 = prod_w1_w2 / prod_w1_w2_norm
            f5 = prod_w1_w2 
            final_f.append(f5)
        else:
            pass
        if 6 in self.f_type:
            f6 = np.concatenate((w1_vec, w2_vec), axis = 0)
            final_f.append(f6)
        else:
            pass
        #return np.concatenate((f1, f2, f3, f4, f5, f6), axis = 0)
        return np.concatenate(final_f, axis = 0)


    def score(self, a, b):
        assert isinstance(a, basestring)
        assert isinstance(b, basestring)
        w1 = a if len(a) >= 1 else NULL
        w2 = b if len(b) >= 1 else NULL
        f = self.get_feats(w1, w2)
        if self.n_hidden > 0:
            h = np.tanh(f.dot(self.w_in_hidden) + self.b_hidden)
            o = np.exp(h.dot(self.w_hidden_out) + self.b_out)
            po = o / np.sum(o)
            return po[1] # if its a good alignment po[0] will be close to one to po[1] is the "cost"
        else:
            o = np.exp(f.dot(self.w_in_out) + self.b_out)
            po = o / np.sum(o)
            return po[1] # if its a good alignment po[0] will be close to one to po[1] is the "cost"

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    opt.add_argument('--word-vec', action='store', dest='word_vec_file', default = None)
    opt.add_argument('--ngram-vec', action='store', dest='ngram_vec_file', required = True)
    opt.add_argument('--dim', action='store', dest='dim', required = True, type = int)
    opt.add_argument('--minn', action='store', dest='minn', type= int, required = True)
    opt.add_argument('--maxn', action='store', dest='maxn', type= int, required = True)
    opt.add_argument('--params', action='store', dest='params', required = True)
    opt.add_argument('--ftypes', action='store', dest='f_types', required = True)
    options = opt.parse_args()
    ET = CombinedEmbeddings(options.word_vec_file, options.dim, options.ngram_vec_file, options.minn, options.maxn)
    ff = FeedForward(options.dim, options.f_types, options.params, ET)
    print '\n----------------DIFF-------------------\n'
    print 'travel, tourist', ff.score(u'travel', u'tourist')
    print 'traveler, tourist', ff.score(u'traveler', u'tourist')
    print 'kids, children:',ff.score('kids', 'children')
    print 'kids, boys:',ff.score('kids', 'boys')
    print 'kids, computers:',ff.score('kids', 'computers')
    print 'loving, like:', ff.score('loving', 'like')
    print 'hello, hey',ff.score('hello', 'hey')
    print 'book,wirte', ff.score('book', 'write')
    print 'nice, good', ff.score('nice', 'good')
    print 'nice, bad', ff.score('nice', 'bad')

    print '\n----------------TYPOS-------------------\n'
    print 'chaildrens,  children:', ff.score('chaildrens', 'children')
    print 'keds, children:', ff.score('keds', 'children')
    print 'keds, kids:', ff.score('keds', 'kids')
    print 'loveing, like', ff.score('loveing', 'like')
    print 'helo, hey:', ff.score('helo', 'hey')
    print 'bok, write', ff.score('bok', 'write')

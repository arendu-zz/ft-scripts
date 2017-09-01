#!/usr/bin/env python
__author__ = 'arenduchintala'
import numpy as np
import time
import editdistance as ed
import codecs
import argparse
import json
from normalization import strip_accents, normalize

EPS = '<eps>'
UNK = 'UNK'

def ntsf(x,k):
    #normalized tunable sigmoid function
    assert -0.99 <= k <= 0.99
    return (x - k*x) / (k - 2*x*k + 1.0)


class FeedForward(object):
    def __init__(self, ngram_dict_path, f_type, saved_weights):
        self.ngram_dict = self.load_ngram_dict(ngram_dict_path)
        self.f_type = sorted([int(f) for f in f_type.split(',')])
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
        print 'loaded model with param-size:', [p.shape for p in _params] 

    def load_ngram_dict(self, _path):
        _d = {}
        for line in codecs.open(_path, 'r', 'utf-8').readlines():
            w,i = line.strip().split()
            _d[w] = int(i)
        return _d
    
    def get_ngrams(self, _w, gram_size=2):
        if _w != EPS:
            w = '<' + strip_accents(normalize(_w.lower())) + '>'
            grams = set([w[i:i + n] for n in xrange(gram_size, gram_size + 1) for i in xrange(len(w)) if w[i:i + n] != '>'])
        else:
            grams = set([])
        return grams

    def get_feats(self, w1, w2):
        w1_grams = self.get_ngrams(w1)
        w2_grams = self.get_ngrams(w2)
        """
        for g1 in w1_grams:
            if g1 not in self.ngram_dict:
                print 'Not seen', g1
        for g1 in w2_grams:
            if g1 not in self.ngram_dict:
                print 'Not seen', g1
        """
        final_f = []
        if 1 in self.f_type:
            diff_gram_idxs = [self.ngram_dict.get(g, self.ngram_dict[UNK]) for g in w1_grams.symmetric_difference(w2_grams)]
            diff_np = np.zeros(len(self.ngram_dict))
            diff_np[diff_gram_idxs] = 1
            final_f.append(diff_np)
        if 2 in self.f_type:
            if w1 == EPS or w2 == EPS:
                final_f.append(np.zeros(len(self.ngram_dict)))
            else:
                intersection_grams_idxs = [self.ngram_dict.get(g, self.ngram_dict[UNK]) for g in w1_grams.intersection(w2_grams)]
                intersection_np = np.zeros(len(self.ngram_dict))
                intersection_np[intersection_grams_idxs] = 1
                final_f.append(intersection_np)
        if 3 in self.f_type:
            if w1 == EPS or w2 == EPS:
                final_f.append(diff_np)
            else:
                union_grams_idx = diff_gram_idxs + intersection_grams_idxs
                union_np = np.zeros(len(self.ngram_dict))
                union_np[union_grams_idx] = 1
                final_f.append(union_np)
        return np.concatenate(final_f, axis = 0)
    
    def eval(self,a,b):
        return self.score(a,b)

    def score(self, a, b):
        assert isinstance(a, basestring)
        assert isinstance(b, basestring)
        w1 = a if len(a) >= 1 else EPS
        w2 = b if len(b) >= 1 else EPS
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

def examples_compute(f):
    f.eval(u'travel', u'tourist')
    f.eval(u'traveler', u'tourist')
    f.eval('kids', 'children')
    f.eval('kids', 'boys')
    f.eval('kids', 'computers')
    f.eval('loving', 'like')
    f.eval('hello', 'hey')
    f.eval('book', 'write')
    f.eval('nice', 'good')
    f.eval('nice', 'bad')
    f.eval('running', 'run')
    f.eval('janked', 'janking')
    f.eval('cried', 'crying')
    f.eval('too', 'boo')
    f.eval('too', 'toos')
    f.eval('individual', 'induviduals')
    f.eval('individual', EPS)
    f.eval('very', EPS)
    f.eval('the', EPS)
    f.eval('a', EPS)
    f.eval('in', EPS)
    f.eval('really', EPS)
    f.eval('much', EPS)
    f.eval('city', EPS)

    f.eval('chaildrens', 'children')
    f.eval('keds', 'children')
    f.eval('keds', 'kids')
    f.eval('loveing', 'loving')
    f.eval('helo', 'hey')
    f.eval('bok', 'write')

def examples(ff, k = 0):
    global ntsf
    print '\n----------------DIFF-------------------\n'
    print 'namibia, familia', ntsf(ff.eval(u'namibia', u'familia'), k)
    print 'travel, tourist', ntsf(ff.eval(u'travel', u'tourist'), k)
    print 'traveler, tourist', ntsf(ff.eval(u'traveler', u'tourist'), k)
    print 'kids, children:',ntsf(ff.eval('kids', 'children'), k)
    print 'kids, boys:',ntsf(ff.eval('kids', 'boys'), k)
    print 'kids, computers:',ntsf(ff.eval('kids', 'computers'), k)
    print 'loving, like:', ntsf(ff.eval('loving', 'like'), k)
    print 'hello, hey',ntsf(ff.eval('hello', 'hey'), k)
    print 'book,wirte', ntsf(ff.eval('book', 'write'), k)
    print 'nice, good', ntsf(ff.eval('nice', 'good'), k)
    print 'nice, bad', ntsf(ff.eval('nice', 'bad'), k)
    print 'run, running', ntsf(ff.eval('running', 'run'), k)
    print 'janking, janks', ntsf(ff.eval('janked', 'janking'), k)
    print 'crying, cried', ntsf(ff.eval('cried', 'crying'), k)
    print 'too, boos', ntsf(ff.eval('too', 'boo'), k)
    print 'too, toos', ntsf(ff.eval('too', 'toos'), k)
    print 'ok, yes', ntsf(ff.eval('ok', 'yes'), k)
    print 'ok, no', ntsf(ff.eval('ok', 'no'), k)
    print 'ya, yes', ntsf(ff.eval('ya', 'yes'), k)
    print 'ya, no', ntsf(ff.eval('ya', 'no'), k)
    print 'thanks, cheese', ntsf(ff.eval('thanks', 'cheese'), k)
    print 'individual, induviduals', ntsf(ff.eval('individual', 'induviduals'), k)
    print 'individual, EPS', ntsf(ff.eval('individual', EPS), k)
    print 'very, EPS', ntsf(ff.eval('very', EPS), k)
    print 'the, EPS', ntsf(ff.eval('the', EPS), k)
    print 'a, EPS', ntsf(ff.eval('a', EPS), k)
    print 'in, EPS', ntsf(ff.eval('in', EPS), k)
    print 'not, EPS', ntsf(ff.eval('not', EPS), k)
    print 'EPS, not', ntsf(ff.eval(EPS, 'not'), k)
    print 'no, EPS', ntsf(ff.eval('no', EPS), k)
    print 'EPS, no', ntsf(ff.eval(EPS, 'no'), k)
    print 'really, EPS', ntsf(ff.eval('really', EPS), k)
    print 'much, EPS', ntsf(ff.eval('much', EPS), k)
    print 'city, EPS', ntsf(ff.eval('city', EPS), k)

    print '\n----------------TYPOS-------------------\n'
    print 'chaildrens,  children:', ntsf(ff.eval('chaildrens', 'children'), k)
    print 'keds, children:', ntsf(ff.eval('keds', 'children'), k)
    print 'keds, kids:', ntsf(ff.eval('keds', 'kids'), k)
    print 'loveing, loving', ntsf(ff.eval('loveing', 'loving'), k)
    print 'helo, hey:', ntsf(ff.eval('helo', 'hey'), k)
    print 'bok, write', ntsf(ff.eval('bok', 'write'), k)

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    opt.add_argument('--fdict', action='store', dest='ngram_dict_path', default = './distance_metrics/en-ppdb-ver2.0-m-phrasal.filtered.out.sub.sim.uniq.feats.fdict')
    opt.add_argument('--params', action='store', dest='params', default='./distance_metrics/en-ppdb-ver2.0-m-phrasal.filtered.out.sub.sim.uniq.feats.f.1,2,3.0.0.001.params')
    opt.add_argument('--ftypes', action='store', dest='f_types', default='1,2,3')
    options = opt.parse_args()
    ff = FeedForward(options.ngram_dict_path, options.f_types, options.params)
    for _k in [0]: #, 0.5, 0.99, -0.5, -0.99]:
        print '*******************', _k, '************************'
        examples(ff, _k)
    #examples(ed)
    for func_name, func in zip(['ff', 'ed'],[ff, ed]):
        start_time = time.time()
        for _ in xrange(1000):
            examples_compute(func)
        print func_name, 'time', time.time() - start_time


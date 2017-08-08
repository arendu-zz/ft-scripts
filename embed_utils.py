#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import argparse
import numpy as np
import codecs

np.set_printoptions(precision = 4, linewidth = np.inf)

class Embeddings(object):
    def __init__(self, _file, _dim, _type, _with_header = True):
        self.type = _type
        self.mat = None
        self.w2idx = {} 
        self.idx2w = {}
        self.mat = {}
        self.dim = _dim
        self.load(_file, _with_header)

    def load(self, v_file, _with_header):
        sys.stderr.write('reading:' + v_file + '\n')
        line_idx = 0
        for line in codecs.open(v_file, 'r', 'utf-8').readlines():
            if line_idx == 0 and _with_header:
                i_ = line.split()
                assert self.dim == int(i_[1]) #check if dim init matches what is in the file
                #self.mat = np.zeros((self.vocab_size, self.dim), dtype=np.float32)
            else:
                i_ = line.split()
                w_ = i_[0]
                w_idx = self.add2w(w_)
                v_ = np.array(i_[1:], dtype=np.float32)
                self.mat[w_idx] = v_
            line_idx += 1

    def add2w(self, w):
        w_idx = self.w2idx.get(w, len(self.w2idx))
        self.w2idx[w] = w_idx
        self.idx2w[w_idx] = w
        return w_idx

    #def add2mat(self,v_str, v_idx):
    #    v_ = np.array(v_str, dtype=np.float32)
    #    self.mat[v_idx, :] = v_

class NgramEmbeddings(Embeddings):
    def __init__(self, _file, _dim, _type, _minn, _maxn):
        Embeddings.__init__(self, _file, _dim, _type, _with_header = False)
        self.minn = _minn
        self.maxn = _maxn

class CombinedEmbeddings(object):
    def __init__(self, w2v_file, dim, ngram2v_file, minn, maxn):
        if w2v_file is not None:
            self.word_vectors = Embeddings(w2v_file, dim, _type = 'words')
        else:
            self.word_vectors = None
        if ngram2v_file is not None:
            self.ngram_vectors = NgramEmbeddings(ngram2v_file, dim, _type = 'ngrams', _minn = minn, _maxn = maxn)
        else:
            self.ngram_vectors = None
        self.dim = dim
        if self.word_vectors is not None and self.ngram_vectors is not None:
            assert self.ngram_vectors.dim == self.word_vectors.dim
        assert self.word_vectors is not None or self.ngram_vectors is not None

    def compute_word_vector(self, w, full_word = 1):
        #w = unicode(w)
        final_vec = np.zeros(self.dim, dtype=np.float32)
        final_vec_num = 0
        if full_word == 1 and self.word_vectors is not None:
            w_idx = self.word_vectors.w2idx.get(w, -1)
            if w_idx >= 0:
                final_vec += self.word_vectors.mat[w_idx] 
                final_vec_num += 1
        else:
            pass
        if self.ngram_vectors is not None:
            w_decorated = '<' + w + '>'
            ngrams = [w_decorated[i:i + n] for n in xrange(self.ngram_vectors.minn, self.ngram_vectors.maxn + 1) for i in xrange(len(w_decorated)) if w_decorated[i:i + n] not in ['<', '>']]
            for g in ngrams:
                g_idx = self.ngram_vectors.w2idx.get(g, -1)
                if g_idx >= 0:
                    final_vec += self.ngram_vectors.mat[g_idx] #[self.ngram_vectors.w2idx[g],:]
                    final_vec_num += 1
                else:
                    sys.stderr.write('dropped:' + g+ '\n')
                    final_vec_num += 1
        else:
            pass
        return final_vec * (1.0 / final_vec_num)

    def get_vec(self, w, full_word):
        w_vec = self.compute_word_vector(w, full_word)
        w_norm = np.linalg.norm(w_vec)
        return w_vec, w_norm

    def cosine_sim(self, w1, w2, full_word = 1):
        assert isinstance(w1, unicode) and isinstance(w2, unicode)
        w1_vec, w1_norm = self.get_vec(w1, full_word) 
        w2_vec, w2_norm = self.get_vec(w2, full_word) 
        return w1_vec.dot(w2_vec) / (w1_norm * w2_norm)


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write ngrams from a corpus to stdout")
    opt.add_argument('--word-vec', action='store', dest='word_vec_file', required = True)
    opt.add_argument('--dim', action='store', dest='dim', required = True, type = int)
    opt.add_argument('--ngram-vec', action='store', dest='ngram_vec_file', required = True)
    opt.add_argument('--minn', action='store', dest='minn', type= int, required = True)
    opt.add_argument('--maxn', action='store', dest='maxn', type= int, required = True)
    options = opt.parse_args()
    ET = CombinedEmbeddings(options.word_vec_file, options.dim, options.ngram_vec_file, options.minn, options.maxn)
    print '\n----------------DIFF-------------------\n'
    print ET.cosine_sim('kids', 'children', full_word = 1)
    print ET.cosine_sim('loving', 'like', full_word = 1)
    print ET.cosine_sim('hello', 'hey', full_word = 1)
    print ET.cosine_sim('book', 'write', full_word = 1)
    print 'nice, good', ET.cosine_sim('nice', 'good', full_word = 1)
    print 'nice, bad', ET.cosine_sim('nice', 'bad', full_word = 1)

    print ET.cosine_sim('kids', 'children', full_word = 0)
    print ET.cosine_sim('loving', 'like', full_word = 0)
    print ET.cosine_sim('hello', 'hey', full_word = 0)
    print ET.cosine_sim('book', 'write', full_word = 0)

    print '\n----------------TYPOS-------------------\n'
    print ET.cosine_sim('keds', 'children', full_word = 1)
    print ET.cosine_sim('loveing', 'like', full_word = 1)
    print ET.cosine_sim('helo', 'hey', full_word = 1)
    print ET.cosine_sim('bok', 'write', full_word = 1)

    print ET.cosine_sim('keds', 'children', full_word = 0)
    print ET.cosine_sim('loveing', 'like', full_word = 0)
    print ET.cosine_sim('helo', 'hey', full_word = 0)
    print ET.cosine_sim('bok', 'write', full_word = 0)

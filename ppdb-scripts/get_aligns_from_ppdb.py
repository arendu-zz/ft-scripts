#!/usr/bin/env python
__author__ = 'arenduchintala'
import itertools
import traceback
import random
import sys
import numpy as np
sys.path.append('../') #forgive me father, for I have sinned
from embed_utils import CombinedEmbeddings
from normalization import strip_accents, normalize
import codecs
import argparse
from collections import defaultdict
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

NULL = "Null"

def cd(w1, w2):
    _cs = (1. + ET.cosine_sim(w1, w2)) * .5 #squeeze into +1,0 range from +1,-1 
    if np.isnan(_cs):
        return 1.0
    return 1. - _cs

def get_nulls(w1, w2s):
    cds = [cd(w1, w2) for w2 in w2s]
    non_null_idx = cds.index(min(cds))
    w2_non_null = w2s.pop(non_null_idx)
    return w2_non_null, w2s

def norm_word(w):
    if w == NULL:
        return w
    n_w = strip_accents(normalize(w))
    return n_w

def training_examples(w1, w2_non_null, w2_nulls):
    n_w2_non_null = norm_word(w2_non_null) 
    n_w1 = norm_word(w1) 
    l = []
    if n_w1 != '' and n_w2_non_null != '':
        l += [orderit(n_w1, n_w2_non_null)] 
        vocab_set.add(n_w1)
        vocab_set.add(n_w2_non_null)
    n_w2_nulls = [norm_word(wn) for wn in w2_nulls]
    vocab_set.update(n_w2_nulls)
    l += [NULL + '\t' + wn for wn in n_w2_nulls if wn != '']  #null always is first...
    return l

def loop_aligns(_p1, _p2, _direction, _align_dict):
    l = set([])
    for k in range(len(_p1)):
        vs = _align_dict[_direction, k]
        if len(vs) == 0:
            pass
        elif len(vs) > 1:
            w1 = _p1[k]
            w2s = [_p2[_v] for _v in vs]
            w2_non_null, w2_nulls = get_nulls(w1, w2s)
            l.update(training_examples(w1, w2_non_null, w2_nulls))
        else:
            w1 = _p1[k]
            w2_non_null = _p2[vs[0]]
            w2_nulls = []
            l.update(training_examples(w1, w2_non_null, w2_nulls))
    return l

def orderit(_w1, _w2):
    if _w1 == NULL:
        return (_w1 +'\t' + _w2) 
    elif _w2 == NULL:
        return (_w2 +'\t' + _w1) 
    else:
        return (_w1 +'\t' + _w2) if _w1 > _w2 else (_w2 + '\t' + _w1)

def random_orderit(p):
    _w1, _w2 = p.split()
    if np.random.rand() > 0.5:
        return _w1 + '\t' + _w2
    else:
        return _w2 + '\t' + _w1

def keep_pair(_w1, _w2):
    if norm_word(_w1) == '':
        return False
    elif norm_word(_w2) == '':
        return False
    elif _w1 == NULL and _w2 == NULL:
        return False
    else:
        return True


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    opt.add_argument('--word-vec', action='store', dest='word_vec_file', default = None)
    opt.add_argument('--ngram-vec', action='store', dest='ngram_vec_file', required = True)
    opt.add_argument('--dim', action='store', dest='dim', required = True, type = int)
    opt.add_argument('--minn', action='store', dest='minn', type= int, required = True)
    opt.add_argument('--maxn', action='store', dest='maxn', type= int, required = True)
    opt.add_argument('--ppdb', action='store' , dest='ppdb',  required = True)
    opt.add_argument('--ppdb-ver', action='store' , dest='ppdb_ver', type = int,  required = True)
    opt.add_argument('--out', action='store' , dest='output_prefix', required = True)
    options = opt.parse_args()
    ET = CombinedEmbeddings(options.word_vec_file, options.dim, options.ngram_vec_file, options.minn, options.maxn)
    vocab = codecs.open(options.output_prefix + '.vocab', 'w', 'utf-8')
    aligned = codecs.open(options.output_prefix + '.aligned', 'w', 'utf-8')
    non_aligned = codecs.open(options.output_prefix + '.non_aligned', 'w', 'utf-8')
    vocab_set = set([])
    for line_idx, line in enumerate(codecs.open(options.ppdb, 'r', 'utf-8').readlines()):
        if line_idx % 10000 == 0:
            sys.stdout.write('.')
        try:
            align_dict = defaultdict(list) #i am going to hell for using defaultdicts.. 
            items = line.split('|||')
            tag = items[0]
            p1 = items[1].split()
            p2 = items[2].split()
            aligns = items[-options.ppdb_ver].split()
            #print '******PHRASE******'
            #print p1, p2, aligns
            p1_with_null = [NULL] + p1
            p2_with_null = [NULL] + p2
            p1xp2 = [orderit(norm_word(_w1), norm_word(_w2)) for _w1, _w2 in itertools.product(p1_with_null, p2_with_null) if keep_pair(_w1, _w2)]
            a12 = [(int(i.split('-')[0]), int(i.split('-')[1])) for i in items[-options.ppdb_ver].split()]
            a21 = [(i[1], i[0]) for i in a12]
            _ = [align_dict['1-2', _k].append(_v) for _k, _v in a12]
            _ = [align_dict['2-1', _k].append(_v) for _k, _v in a21]

            #print '1-----------2'
            l1 = loop_aligns(p1, p2, '1-2', align_dict)
            #print '\n'.join(l1)
            #print '2-----------1'
            l2 = loop_aligns(p2, p1, '2-1', align_dict)
            #print '\n'.join(l2)
            #print '1-----------------2 2--------------------1'
            final_l = l1.intersection(l2)
            final_l.update([al1 for al1 in l1 if al1.startswith(NULL)] + [al2 for al2 in l2 if al2.startswith(NULL)])
            aligned_final_l = ['0\t' + random_orderit(_i) for _i in final_l]
            aligned.write('\n'.join(aligned_final_l) + '\n')
            p1xp2 = list(set(p1xp2) - final_l)
            random.shuffle(p1xp2)
            non_aligned_final_l = ['1\t' + random_orderit(_i) for _i in p1xp2[:len(final_l)]]
            non_aligned.write('\n'.join(non_aligned_final_l) + '\n') #join(list(set(p1xp2) - final_l)[:len(final_l)]) + '\n')
        except BaseException as e:
            #exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exc()
            sys.stderr.write('\n' + str(line_idx) + ' has an error\n')
    aligned.flush()
    aligned.close()
    non_aligned.flush()
    non_aligned.close()
    vocab.write('\n'.join(vocab_set))
    vocab.flush()
    vocab.close()

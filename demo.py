#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'arenduchintala'
import itertools
import sys
import argparse
from embed_utils import CombinedEmbeddings
from span_ed import SpanEditSearch
from pprint import pprint
import numpy as np
import sys
import codecs
reload(sys)
sys.setdefaultencoding("utf-8")
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def char_levenshetien_dist(a, b, h = 0):
    assert isinstance(a, list)
    assert isinstance(b, list)
    import editdistance as ed
    a = ''.join(a)
    b = ''.join(b)
    return int(ed.eval(a,b))  + h

def cosine_dist(a, b):
    dist = 0.
    dist_c = 0
    if len(a) == 0 or len(b) == 0:
        return 1.
    for a_idx, b_idx in itertools.product(a, b):
        if a_idx.strip() == '<EPS>' or b_idx.strip() == '<EPS>':
            pass
        else:
            cs = (1. + ET.cosine_sim(a_idx, b_idx, full_word = 0 if options.word_vec_file is None else 1)) * .5 #squeeze into +1,0 range from +1,-1 
            cd = 1 - cs 
            dist += cd
            dist_c += 1
    dist/= float(dist_c)
    dist = np.around(dist, 4)
    assert dist >= 0.
    return dist

def show_alignments(path, verbose = 0):
    a = []
    for p in path:
        if verbose == 0:
            a.append(u' '.join(p.src_str) + '->' + u' '.join(p.tar_str))
        else:
            a.append(u' '.join(p.src_str) + '->' + u' '.join(p.tar_str) + '(' + '%.4f' % p.node_cost + ')')
    return ', '.join(a)


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write ngrams from a corpus to stdout")
    opt.add_argument('--word-vec', action='store', dest='word_vec_file', default = None)
    opt.add_argument('--dim', action='store', dest='dim', required = True, type = int)
    opt.add_argument('--ngram-vec', action='store', dest='ngram_vec_file', required = True)
    opt.add_argument('--minn', action='store', dest='minn', type= int, required = True)
    opt.add_argument('--maxn', action='store', dest='maxn', type= int, required = True)
    opt.add_argument('--span_size', action='store', dest='span_size', type = int, required = False, default = 1, choices=range(1,10))
    opt.add_argument('-v', action='store', dest='verbose', required=False, default= 1, choices=[0,1,2], type = int)
    options = opt.parse_args()
    ET = CombinedEmbeddings(options.word_vec_file, options.dim, options.ngram_vec_file, options.minn, options.maxn)
    a = "i love to read books".split()
    b = "i like reading novels".split()
    #table, path = ss.span_edit_dist(a,b)
    #pprint(path)
    #print '\n-------------------------------------------\n'
    cs = SpanEditSearch(0., options.span_size, cosine_dist) 
    ls = SpanEditSearch(0., options.span_size, char_levenshetien_dist) 
    story_arcs = '' 
    while story_arcs is not None:
        story_arcs = raw_input("Enter chat options (comma separated):")
        if story_arcs.strip() == '':
            story_arcs = None
            continue
        story_arcs = story_arcs.strip().lower().split(',')
        user_intput = raw_input("Enter learner input:")
        user_intput = user_intput.strip().lower().split()
        print 'Cosine Distance based:'
        for sa_idx, sa in enumerate(story_arcs):
            sa = sa.split() 
            table, path = cs.span_edit_dist(user_intput,sa)
            if options.verbose == 2:
                pprint(path)
            print sa_idx, show_alignments(path, options.verbose), 'cost:', path[-1].cost
        print '------------------------------------------------\n'
        print 'Levenshtein Distance based:'
        for sa_idx, sa in enumerate(story_arcs):
            sa = sa.split() 
            table, path = ls.span_edit_dist(user_intput,sa)
            if options.verbose == 2:
                pprint(path)
            print sa_idx, show_alignments(path, options.verbose), 'cost:', path[-1].cost
        print '------------------------------------------------\n'

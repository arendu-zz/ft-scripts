#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'arenduchintala'
import itertools
import sys
import traceback
import argparse
from embed_utils import CombinedEmbeddings
from span_ed import SpanEditSearch
import numpy as np
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

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
            if np.isnan(cs):
                cs = 0.
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

def get_alignment_costs(path):
    a = []
    for p in path:
        a.append('%.4f' % p.node_cost)
    return ','.join(a)


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write ngrams from a corpus to stdout")
    opt.add_argument('--word-vec', action='store', dest='word_vec_file', default = None, required = False)
    opt.add_argument('--dim', action='store', dest='dim', required = True, type = int)
    opt.add_argument('--ngram-vec', action='store', dest='ngram_vec_file', required = True)
    opt.add_argument('--minn', action='store', dest='minn', type= int, required = True)
    opt.add_argument('--maxn', action='store', dest='maxn', type= int, required = True)
    opt.add_argument('--span_size', action='store', dest='span_size', type = int, required = False, default = 1, choices=range(1,10))
    opt.add_argument('-v', action='store', dest='verbose', required=False, default= 1, choices=[0,1,2], type = int)
    opt.add_argument('-l', action='store_true', dest='lowercase', required=False, default=False)
    options = opt.parse_args()
    ET = CombinedEmbeddings(options.word_vec_file, options.dim, options.ngram_vec_file, options.minn, options.maxn)
    cs = SpanEditSearch(0., options.span_size, cosine_dist)
    ls = SpanEditSearch(0., options.span_size, char_levenshetien_dist)
    headers = "REPLY CLOSEST_STRING COSINE_DISTANCE_ALIGNS COSINE_DISTANCE_ALIGN_COSTS COSINE_DISTANCE LEV_DISTANCE_ALIGN LEV_DISTANCE_ALIGN_COSTS LEV_DISTANCE BETTER_ANSWER".split()
    sys.stdout.write('\t'.join(headers) + '\n')
    line_num = -1
    #for o_line in codecs.open(options.data, 'r', 'utf-8').readlines():
    for o_line in sys.stdin: 
        line_num +=1
        try:
            sys.stderr.write('line_num:' + str(line_num) + '\n')
            if options.lowercase:
                line = o_line.lower()
            else:
                line = o_line
                pass
            items = line.strip().split('|')
            reply = items[4].strip().split()
            better_answer = items[8]
            closest_string = items[10].strip().split()
            table_cs, path_cs = cs.span_edit_dist(closest_string, reply)
            cs_dist = path_cs[-1].cost
            cs_align = show_alignments(path_cs, 0)

            table_ls, path_ls = ls.span_edit_dist(closest_string, reply)
            ls_dist = path_ls[-1].cost
            ls_align = show_alignments(path_ls, 0)
            w = [' '.join(reply), ' '.join(closest_string), cs_align, get_alignment_costs(path_cs), '%.4f' %cs_dist, ls_align, get_alignment_costs(path_ls), '%.4f' %ls_dist, better_answer]
            sys.stdout.write('\t'.join(w) + '\n')
        except BaseException as e:
            sys.stderr.write("ERROR:" + line)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

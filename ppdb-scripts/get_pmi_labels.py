#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import numpy as np
import codecs
import argparse
import traceback
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

NULL = "Null"

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('--ap', action='store' , dest='aligned_pairs', required = True)
    opt.add_argument('--nap', action='store' ,dest='non_aligned_pairs', required = True)
    opt.add_argument('--fp', action='store', dest='feats_prefix', required = True)
    options = opt.parse_args()
    feats0 = codecs.open(options.feats_prefix + '.fpmi', 'w', 'utf-8')
    counts_w1 = {}
    counts_w1_w2 = {}
    N = 0
    for line_idx, line in enumerate(codecs.open(options.aligned_pairs, 'r', 'utf-8')):
        try:
            _label, _w1, _w2 = line.strip().split()
            counts_w1[_w1] = counts_w1.get(_w1, 0) + 1.
            counts_w1[_w2] = counts_w1.get(_w2, 0) + 1.
            counts_w1_w2[tuple(sorted([_w1, _w2]))] = counts_w1_w2.get(tuple(sorted([_w1, _w2])), 0) + 1
            N += 1.
        except:
            sys.stderr.write('error in aligned line:' + str(line_idx) + '\t' + line.strip())
            traceback.print_exc()
    pmi_w1_w2 = {}
    for _pair, count in counts_w1_w2.iteritems():
        pmi_w1_w2[_pair] = np.log( (count * N) / (counts_w1[_pair[0]] * counts_w1[_pair[1]]))

    for line_idx, line in enumerate(codecs.open(options.aligned_pairs, 'r', 'utf-8')):
        try:
            _label, _w1, _w2 = line.strip().split()
            pmi = pmi_w1_w2[tuple(sorted([_w1, _w2]))]
            feats0.write('%.4f'%pmi  + '\n')
        except:
            sys.stderr.write('error in aligned line:' + str(line_idx) + '\t' + line.strip())
            traceback.print_exc()
    for line_idx, line in enumerate(codecs.open(options.non_aligned_pairs, 'r', 'utf-8')):
        try:
            _label, _w1, _w2 = line.strip().split()
            feats0.write('0.0'  + '\n')
        except:
            sys.stderr.write('error in non-aligned line:' + str(line_idx) + '\t' + line.strip())
            traceback.print_exc()
    feats0.flush()
    feats0.close()

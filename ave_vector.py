#!/usr/bin/env python
__author__ = 'arenduchintala'
import numpy as np
import sys
import codecs
import argparse
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

def to_float(_lst):
    return [float(i) for i in _lst]

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('-v', action='store' , dest='word_vecs',  required = True)
    options = opt.parse_args()
    ws = []
    vs = []
    for line in codecs.open(options.word_vecs, 'r', 'utf-8').readlines()[1:]:
        i = line.strip().split()
        w = i[0]
        v = to_float(i[1:])
        vs.append(v)
        ws.append(w)
    vs = np.array(vs)
    vs = vs.mean(axis=0)
    print '__eps__', ' '.join([str(float(_i)) for _i in np.around(vs,5)])


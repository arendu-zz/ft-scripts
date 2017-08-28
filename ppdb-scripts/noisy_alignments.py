#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import codecs
import argparse
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

NULL='Null'

def get_sample(w):
    global noise
    n_w = np.random.choice(noise[w][0], p=noise[w][1]) 
    return n_w

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="takes a ppdb file and a translations file (which contains words and their noisy translations and a score) and makes a noisy ppdb file")

    #insert options here
    opt.add_argument('--ppdb_alignments', action='store' , dest='ppdb_alignments', required = True)
    opt.add_argument('--translations', action='store' , dest='translations', required = True)
    opt.add_argument('--temp', action='store', dest='temp', required = False, type=float, default =0.35) #mostly pick the high probably incorrect spelling
    options = opt.parse_args()
    noise = {}
    for line in codecs.open(options.translations, 'r' 'utf-8').readlines():
        try:
            w,n,s = line.strip().split()
            n_list, s_list = noise.get(w, ([],[]))
            n_list.append(n)
            s_list.append(np.exp(float(s)/ options.temp))
            noise[w] = n_list, s_list
        except:
            pass
    for w,v in noise.iteritems():
        n_list = v[0]
        s_list = v[1]
        s_list = np.array(s_list) 
        s_list /= np.sum(s_list)
        noise[w] = (n_list, s_list)

    for line_idx, line in enumerate(codecs.open(options.ppdb_alignments, 'r', 'utf-8').readlines()):
        if line_idx % 10000 == 0:
            sys.stderr.write('.')
        label, w1, w2 = line.split()
        if w1 == NULL:
            n_w1 = w1
        else:
            n_w1 = get_sample(w1)
        print label, n_w1, w2

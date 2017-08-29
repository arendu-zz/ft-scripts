#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import numpy as np
import editdistance as ed
import codecs
import argparse
import traceback
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

EPS = "Null"
ngram_dict = {'UNK': 0}

def load_noise(noise_file, noise_temp):
    noise = {}
    for line in codecs.open(noise_file, 'r' 'utf-8').readlines():
        try:
            w,n,s = line.strip().split()
            n_list, s_list = noise.get(w, ([],[]))
            n_list.append(n)
            s_list.append(np.exp(float(s)/ noise_temp))
            noise[w] = n_list, s_list
        except:
            pass
    for w,v in noise.iteritems():
        n_list = v[0]
        s_list = v[1]
        s_list = np.array(s_list) 
        s_list /= np.sum(s_list)
        noise[w] = (n_list, s_list)
    return noise

def get_sample(w, noise):
    if w in noise:
        return np.random.choice(noise[w][0], p=noise[w][1]) 
    else:
        sys.stderr.write('cant find ' + w + ' in noise dist\n')
        return w

def arr2str(v):
    return ' '.join([str(int(n)) for n in v])

def get_ngrams(w, gram_size=2):
    global ngram_dict
    if w != EPS:
        w = '<' + w + '>'
        grams = set([w[i:i+n] for n in xrange(gram_size, gram_size + 1) for i in xrange(len(w)) if w[i:i+n] != '>'])
    else:
        grams = set([])
    for g in grams:
        ngram_dict[g] = ngram_dict.get(g, len(ngram_dict))
    return grams

def get_feat_list(w1, w2):
    global ngram_dict
    w1_grams = get_ngrams(w1)
    w2_grams = get_ngrams(w2)
    diff_gram_idxs = [ngram_dict[g] for g in w1_grams.symmetric_difference(w2_grams)]
    intersection_grams_idxs = [ngram_dict[g] for g in w1_grams.intersection(w2_grams)]
    union_grams_idx = diff_gram_idxs + intersection_grams_idxs
    diff_np = np.zeros(len(ngram_dict))
    intersection_np = np.zeros(len(ngram_dict))
    intersection_np[intersection_grams_idxs] = 1
    diff_np = np.zeros(len(ngram_dict))
    diff_np[diff_gram_idxs] = 1
    union_np = np.zeros(len(ngram_dict))
    union_np[union_grams_idx] = 1
    return diff_np, intersection_np, union_np

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('--data', action='store' ,dest='supervised_data', required = True)
    opt.add_argument('--fp', action='store', dest='feats_prefix', required = True)
    options = opt.parse_args()
    sd =  codecs.open(options.supervised_data, 'r', 'utf-8').readlines()
    for line in sd:
        label, w1, w2 = line.strip().split()
        get_ngrams(w1)
        get_ngrams(w2)
    feats0 = codecs.open(options.feats_prefix + '.f0', 'w', 'utf-8')
    feats1 = codecs.open(options.feats_prefix + '.f1', 'w', 'utf-8')
    feats2 = codecs.open(options.feats_prefix + '.f2', 'w', 'utf-8')
    feats3 = codecs.open(options.feats_prefix + '.f3', 'w', 'utf-8')
    for line in sd:
        _label, w1, w2 = line.strip().split()
        f1, f2 ,f3 = get_feat_list(w1, w2)
        feats0.write(_label  + '\n')
        feats1.write(arr2str(f1)  + '\n')
        feats2.write(arr2str(f2)  + '\n')
        feats3.write(arr2str(f3)  + '\n')
    feats0.flush()
    feats1.flush()
    feats2.flush()
    feats3.flush()
    feats0.close()
    feats1.close()
    feats2.close()
    feats3.close()
    feats_dict = codecs.open(options.feats_prefix + '.fdict', 'w', 'utf-8')
    for g,g_idx in ngram_dict.iteritems():
        feats_dict.write(g + '\t' + str(g_idx) + '\n')
    feats_dict.flush()
    feats_dict.close()





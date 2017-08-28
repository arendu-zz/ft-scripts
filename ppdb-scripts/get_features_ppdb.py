#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import numpy as np
import editdistance as ed
sys.path.append('../') #forgive me father, for I have sinned
from embed_utils import CombinedEmbeddings 
import codecs
import argparse
import traceback
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)

NULL = "Null"

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

def edit_sim(w1, w2):
    if w1 == 'Null' or w2 == 'Null':
        return 0.0
    else:
        return 1.0 - ((float(ed.eval(w1, w2))) / max(len(w1), len(w2)))

def arr2str(v):
    return ' '.join(['%.5f' % n for n in v])

def vecs(w, _et):
    if w == NULL:
        w_vec = np.ones(options.dim)
        w_norm = 1.0
    else:
        w_vec, w_norm = ET.get_vec(w, 1)
    return w_vec, w_norm

def get_feat_list(w1, w2, _et):
    #w1, w2 = (w1, w2) if np.random.rand() < 0.5 else (w2, w1)
    es = edit_sim(w1, w2) 
    w1_isNull = 1.0 if w1 == NULL else 0.0
    w2_isNull = 1.0 if  w2 == NULL else 0.0
    l1 = len(w1) if w1 != NULL else 0
    l2 = len(w2) if w2 != NULL else 0
    w1_vec, w1_norm = vecs(w1, ET)
    w2_vec, w2_norm = vecs(w2, ET)
    cs = w1_vec.dot(w2_vec) / (w1_norm * w2_norm)
    prod_w1_w2 = w1_vec * w2_vec
    prod_w1_w2_norm = np.linalg.norm(prod_w1_w2)
    prod_w1_w2 = prod_w1_w2 / prod_w1_w2_norm
    f1 = np.array([w1_isNull, w2_isNull])
    f2 = np.array([cs])
    f3 = np.array([es])
    f4 = np.array([np.abs(float(l1 - l2)/10.0), float(l1)/10.0, float(l2)/10.0]) 
    f5 = prod_w1_w2 
    f6 = np.concatenate((w1_vec, w2_vec), axis = 0)
    return f1, f2, f3, f4, f5, f6

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('--ap', action='store' , dest='aligned_pairs', required = True)
    opt.add_argument('--nap', action='store' ,dest='non_aligned_pairs', required = True)
    opt.add_argument('--word-vec', action='store', dest='word_vec_file', default = None)
    opt.add_argument('--ngram-vec', action='store', dest='ngram_vec_file', required = True)
    opt.add_argument('--dim', action='store', dest='dim', required = True, type = int)
    opt.add_argument('--minn', action='store', dest='minn', type= int, required = True)
    opt.add_argument('--maxn', action='store', dest='maxn', type= int, required = True)
    opt.add_argument('--fp', action='store', dest='feats_prefix', required = True)
    opt.add_argument('--noise', action='store' , dest='noise_file', required = False)
    opt.add_argument('--noise-temp', action='store', dest='noise_temp', required = False, type=float, default =0.35) #mostly pick the high probably incorrect spelling
    opt.add_argument('--noise-percentage', action='store', dest='noise_percentage', required = False, type=float, default =0.5) #50% of lines will be noisy
    options = opt.parse_args()
    add_noise = False 
    noise = {}
    if options.noise_file is not None:
        noise = load_noise(options.noise_file, options.noise_temp)
        add_noise = True
    else:
        pass
    ET = CombinedEmbeddings(options.word_vec_file, options.dim, options.ngram_vec_file, options.minn, options.maxn)
    feats0 = codecs.open(options.feats_prefix + '.f0', 'w', 'utf-8')
    feats1 = codecs.open(options.feats_prefix + '.f1', 'w', 'utf-8')
    feats2 = codecs.open(options.feats_prefix + '.f2', 'w', 'utf-8')
    feats3 = codecs.open(options.feats_prefix + '.f3', 'w', 'utf-8')
    feats4 = codecs.open(options.feats_prefix + '.f4', 'w', 'utf-8')
    feats5 = codecs.open(options.feats_prefix + '.f5', 'w', 'utf-8')
    feats6 = codecs.open(options.feats_prefix + '.f6', 'w', 'utf-8')
    for line_idx, line in enumerate(codecs.open(options.aligned_pairs, 'r', 'utf-8')):
        try:
            _label, _w1, _w2 = line.strip().split()
            _w1 = get_sample(_w1, noise) if (add_noise and np.random.rand() < options.noise_percentage) else _w1
            f1, f2, f3, f4, f5, f6 = get_feat_list(_w1, _w2, ET)
            feats0.write(_label  + '\n')
            feats1.write(arr2str(f1)  + '\n')
            feats2.write(arr2str(f2)  + '\n')
            feats3.write(arr2str(f3)  + '\n')
            feats4.write(arr2str(f4)  + '\n')
            feats5.write(arr2str(f5)  + '\n')
            feats6.write(arr2str(f6)  + '\n')
        except:
            sys.stderr.write('error in aligned line:' + str(line_idx) + '\t' + line.strip())
            traceback.print_exc()
    feats0.flush()
    feats1.flush()
    feats2.flush()
    feats3.flush()
    feats4.flush()
    feats5.flush()
    feats6.flush()

    for line_idx, line in enumerate(codecs.open(options.non_aligned_pairs, 'r', 'utf-8')):
        try:
            _label, _w1, _w2 = line.strip().split()
            _w1 = get_sample(_w1, noise) if (add_noise and np.random.rand() < options.noise_percentage) else _w1
            f1, f2, f3, f4, f5, f6 = get_feat_list(_w1, _w2, ET)
            feats0.write(_label  + '\n')
            feats1.write(arr2str(f1)  + '\n')
            feats2.write(arr2str(f2)  + '\n')
            feats3.write(arr2str(f3)  + '\n')
            feats4.write(arr2str(f4)  + '\n')
            feats5.write(arr2str(f5)  + '\n')
            feats6.write(arr2str(f6)  + '\n')
        except:
            sys.stderr.write('error in non-aligned line:' + str(line_idx) + '\t' + line.strip())
            traceback.print_exc()
    feats0.flush()
    feats1.flush()
    feats2.flush()
    feats3.flush()
    feats4.flush()
    feats5.flush()
    feats6.flush()
    feats0.close()
    feats1.close()
    feats2.close()
    feats3.close()
    feats4.close()
    feats5.close()
    feats6.close()

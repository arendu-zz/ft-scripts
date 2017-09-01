import editdistance
from feedforward_np_str import FeedForward
EPS = '<eps>'
ff_ins_del = FeedForward('./distance_models/en-ppdb-ver2.0-m-phrasal.filtered.out.insdel.feats.fdict',
        '1,2,3',
        './distance_models/en-ppdb-ver2.0-m-phrasal.filtered.out.insdel.feats.f.1,2,3.0.0.001.params')

ff_sub = FeedForward('./distance_models/en-ppdb-ver2.0-m-phrasal.filtered.out.sub.sim.uniq.feats.fdict',
        '1,2,3',
        './distance_models/en-ppdb-ver2.0-m-phrasal.filtered.out.sub.sim.uniq.feats.f.1,2,3.0.0.001.params')


def composite_score(a, b):
    global EPS, ff_ins_del, ff_sub
    if a == EPS or b == EPS:
        return ff_ins_del.score(a, b)
    else:
        if a.lower() == b.lower():
            return 0.0
        else:
            return ff_sub.score(a, b)

def levenshtein(w1, w2):
    if w1 == EPS:
        return float(len(w2))
    elif w2 == EPS:
        return float(len(w1))
    elif w1 != EPS and w2 != EPS:
        c = float(editdistance.eval(w1, w2))
        return c 
    else:
        raise BaseException("both words are empty")

def normalized_levenshtein(w1, w2):
    #l = float(levenshtein(w1,w1)) 
    lw1 = len(w1) if w1 != EPS else 0
    lw2 = len(w2) if w2 != EPS else 0
    m = lw1 if lw1 > lw2 else lw2
    #n = float(max(lw1,lw2))
    return levenshtein(w1, w2) / m

def match_dist(w1, w2):
    if w1 == w2:
        return 0.0
    else:
        return 1.0

def ntsf(x,k):
    #normalized tunable sigmoid function
    if k < -0.9999:
        k = -0.9999
    elif k > 0.9999:
        k = 0.9999
    else:
        pass
    y = (x - k*x) / (k - 2*x*k + 1.0)
    assert 0.0 <= y <= 1.0
    return y

def bigram_overlap(w1, w2):
    if w1 == EPS or w2 == EPS:
        return 1.0
    else:
        _w1 = '<' + w1 + '>'
        _w2 = '<' + w2 + '>'
        b1 = set([_w1[i:i+n] for n in xrange(2,3) for i in xrange(len(_w1)) if _w1[i: i+n] != '>'])
        b2 = set([_w2[i:i+n] for n in xrange(2,3) for i in xrange(len(_w2)) if _w2[i: i+n] != '>'])
        return 1.0 - (float(len(b1.intersection(b2))) / float(len(b1.union(b2))))


def meteor(w1, w2):
    if w1 == EPS or w2 == EPS:
        return 1.0
    else:
        m_1 = set([i for i in w1])
        m_2 = set([i for i in w2])
        m = float(len(m_1.intersection(m_2)))
        lm_1 = float(len(m_1))
        lm_2 = float(len(m_2))
        if m == 0:
            return 1.0
        else:
            inv_p = lm_1 / m
            inv_r = lm_2 / m
            hm = 2.0 / (inv_p + inv_r)
            if not 0.0 <= hm <= 1.0:
                print m_1, 'm_1'
                print m_2, 'm_2'
                print m, 'm'
                print lm_1, 'lm_1'
                print lm_2, 'lm_2'
                print hm, 'hm'
                raise BaseException("HM is outside [0,1]")
            return 1.0 - hm

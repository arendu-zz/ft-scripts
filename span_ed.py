#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
from pprint import pprint
__author__ = 'arenduchintala'
NO_OP = "<no_op>"
NO_TAG= "<no_tag>"
EPS = '<EPS>'

class SearchState(object):
    def __init__(self, cost, action, src_str, tar_str, next_pos):
        if isinstance(tar_str, list) and isinstance(src_str, list):
            pass
        else:
            print tar_str
            print src_str
            raise BaseException("these should be lists!")
        self.cost = cost
        self.action = action
        self.src_str = src_str
        self.src_l_context = [] 
        self.src_r_context = [] 
        self.tar_str = tar_str
        self.next_pos = next_pos
        self.node_cost = 0.

    def __repr__(self,):
        if isinstance(self.src_str, list) and isinstance(self.tar_str, list) and isinstance(self.src_l_context, list) and isinstance(self.src_r_context, list):
            src_l_context = ' '.join(self.src_l_context)
            src_r_context = ' '.join(self.src_r_context)
            src_str = ' '.join(self.src_str)
            tar_str = ' '.join(self.tar_str)
            return str(self.cost) + ', ' + str(self.node_cost)  + ', ' + self.action + ', (' + src_l_context + ',' + src_r_context + '), '  + (src_str if src_str.strip() != '' else EPS)  + '->' + (tar_str if tar_str.strip() != '' else EPS) + ", BP:" + str(self.next_pos)
        else:
            return str(self.cost) + ', ' + str(self.node_cost) + ', ' + self.action + ', (' + self.src_l_context + ',' + self.src_r_context + '), '  + (self.src_str if self.src_str.strip() != '' else EPS)  + '->' + (self.tar_str if self.tar_str.strip() != '' else EPS) + ", BP:" + str(self.next_pos)

class SpanEditSearch(object):
    def __init__(self, cost_per_action, span_size, dist_func):
        self.cost_per_action = cost_per_action
        self.span_size = span_size
        self.dist_func = dist_func

    def default_dist(self, s_a, s_b):
        if len(s_a) >= 1 and len(s_b) == 0:
            return len(s_a)
        elif len(s_a) == 0 and len(s_b) >= 1:
            return len(s_b)
        elif len(s_a) >= 1 and len(s_b) >=1:
            if s_a == s_b:
                return 0
            else:
                return len(s_a) + len(s_b)
        elif len(s_a) == 0 and len(s_b) == 0:
            return 0
        else:
            raise BaseException('no edit...')

    def apply_span_edits(self, a, s_a, s_b, bp_i, bp_j, old_cost):
        old_cost += self.cost_per_action 
        if self.dist_func is None:
            new_cost = self.default_dist(s_a, s_b)
        else:
            new_cost = self.dist_func(s_a, s_b)
        ss = SearchState(old_cost + new_cost, '-', s_a, s_b, (bp_i, bp_j))
        ss.node_cost = new_cost
        ss.src_l_context = a[bp_i - 1: bp_i] if a[bp_i - 1: bp_i] != [] else [EPS]
        ss.src_r_context = a[bp_i + len(s_a): bp_i + len(s_a) + 1] if a[bp_i + len(s_a): bp_i + len(s_a) + 1] != [] else [EPS]
        return ss

    def backwordPass(self, bp, goal, decorate = False):
        path = []
        #cost, action, src_str, tar_str, next_pos = bp[goal]
        s_state = bp[goal]
        next_pos = s_state.next_pos
        path.append(s_state) #(action, src_str, tar_str, cost))
        while next_pos != (0,0):
            #cost, action, src_str, tar_str, next_pos = bp[next_pos]
            s_state = bp[next_pos]
            next_pos = s_state.next_pos
            path.append(s_state) #(action, src_str, tar_str, cost))
        path.reverse()
        return path

    def span_edit_dist(self, a, b, decorate = False):
        edits_table = {}
        edits_table[0,0] = SearchState(0., NO_OP, [], [], None)
        for i, k in itertools.product(xrange(len(a) + 1), xrange(1, self.span_size + 1)):
            if i - k >= 0:
                old_cost = edits_table[i - k, 0].cost #[0]
                edits_table[i, 0] = self.apply_span_edits(a, a[i - k:i], b[0:0], i - k, 0, old_cost) # (i - 1, 0), (a[i - 1], EPS)

        for j, l in itertools.product(xrange(len(b) + 1), xrange(1, self.span_size + 1)):
            if j - l >= 0:
                old_cost = edits_table[0, j -l].cost #[0]
                edits_table[0, j] = self.apply_span_edits(a, a[0:0], b[j - l: j], 0, j - l, old_cost) # (0, j - 1), (EPS, b[j - 1])
        for i in xrange(1, len(a) + 1):
            for j in xrange(1, len(b) + 1):
                candidates = []
                for k, l in [(k,l) for k,l in itertools.product(xrange(self.span_size + 1), xrange(self.span_size + 1)) if k + l != 0]:
                    if i - k >= 0 and j - l >= 0:
                        old_cost = edits_table[i - k, j - l].cost #[0]
                        candidates.append(self.apply_span_edits(a, a[i - k:i], b[j - l:j], i - k, j - l, old_cost))
                    else:
                        pass
                candidates = [c for c in candidates if c.action != NO_OP]
                candidates.sort(key= lambda x: x.cost)
                edits_table[i,j] = candidates[0] #min(candidates)
        path = self.backwordPass(edits_table, (len(a), len(b)), decorate)
        return edits_table, path


def product_dist(a, b):
    assert isinstance(a, list)
    assert isinstance(b, list)
    dist = 0.
    for a_idx, b_idx in itertools.product(a, b):
        dist = dist + (0. if a_idx == b_idx else 1.)
        print 'prd dist', a_idx, b_idx, dist
    return dist

def char_levenshetien_dist(a, b):
    assert isinstance(a, list)
    assert isinstance(b, list)
    import editdistance as ed
    a = ''.join(a)
    b = ''.join(b)
    return int(ed.eval(a,b))


if __name__ == '__main__':
    ss = SpanEditSearch(0., 1, None) 
    a = "this is a phrase".split()
    b = "this is a sentence".split()
    print product_dist(a, b)
    print char_levenshetien_dist(a, b)
    table, path = ss.span_edit_dist(a,b)
    pprint(table)
    pprint(path)

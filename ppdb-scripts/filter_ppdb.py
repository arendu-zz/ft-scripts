#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import codecs
import argparse
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr)
sys.stdin = codecs.getreader('utf-8')(sys.stdin)


def check_words(p):
    global wf
    for w in p:
        if w in wf:
            pass
        else:
            return False
    return True

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('--wf', action='store' , dest='word_frequency', required = True)
    opt.add_argument('--ppdb', action='store' , dest='ppdb',  required = True)
    opt.add_argument('--limit', action='store', dest='limit', type=int, required=False, default=10000)
    options = opt.parse_args()
    wf = dict((l.split()[0] , int(l.split()[1])) for l in codecs.open(options.word_frequency, 'r', 'utf-8').readlines()[:options.limit])
    for line_idx, line in enumerate(codecs.open(options.ppdb, 'r', 'utf-8').readlines()):
        if line_idx % 10000 == 0:
            sys.stderr.write('.')
        try:
            items = line.split('|||')
            tag = items[0]
            p1 = items[1].split()
            if not check_words(p1):
                continue
            p2 = items[2].split()
            if not check_words(p2):
                continue
            print line.strip()
        except:
            pass

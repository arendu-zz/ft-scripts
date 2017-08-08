#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import codecs
import argparse
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

BOW = '<'
EOW = '>'

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write ngrams from a corpus to stdout")
    #insert options here
    opt.add_argument('--vec', action='store', dest='word_vec_file', required = True)
    options = opt.parse_args()
    attrs = options.word_vec_file.split('.')
    ming = int(attrs[attrs.index('ming') + 1])
    maxg = int(attrs[attrs.index('maxg') + 1])
    grams = set([])
    for line in codecs.open(options.word_vec_file, 'r', 'utf8').readlines()[1:]:
        word = line.strip().split()[0]
        word = BOW + word + EOW
        word_grams = [word[i:i + n] for n in xrange(ming, maxg + 1) for i in xrange(len(word))]
        grams.update(word_grams)

    if ming == 1:
        grams.remove(BOW)
        grams.remove(EOW)
    else:
        pass
    w = codecs.open(options.word_vec_file.replace('.vec', '.subwords'), 'w', 'utf-8')
    w.write('\n'.join(grams))
    w.flush()
    w.close()
    print 'subwords written out to', options.word_vec_file.replace('.vec', '.subwords')

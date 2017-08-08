# Demo:
0. Train subword models with `fastText` with `-wordInSubwords` option set to `0`.  Pretrained vectors are avaible, contact me.

```
$export WORD_EMBEDDING_FILE='../models/fastext.full-simple-wiki.e.50.m.skipgram.ming.1.maxg.4.winsubs.0.vec'
$export NGRAM_EMBEDDING_FILE='../models/fastext.full-simple-wiki.e.50.m.skipgram.ming.1.maxg.4.winsubs.0.ngrams'
```
1. To run the demo:
```
$./demo.py  --word-vec $WORD_EMBEDDING_FILE --ngram-vec $NGRAM_EMBEDDING_FILE --minn 1 --maxn 4 --dim 50 --span_size 1
../models/fastext.full-simple-wiki.e.50.m.skipgram.ming.1.maxg.4.winsubs.0.vec
../models/fastext.full-simple-wiki.e.50.m.skipgram.ming.1.maxg.4.winsubs.0.ngrams
Enter chat options (comma seperated):I like to watch movies, I like to read books
Enter learner input:I love novels
Cosine Distance based:
0 i->i(-0.0000), love->like(0.1015), ->to(1.0000), ->watch(1.0000), novels->movies(0.0549) cost: 2.1564
1 i->i(-0.0000), love->like(0.1015), ->to(1.0000), ->read(1.0000), novels->books(0.0324) cost: 2.1339
------------------------------------------------

Levenshtein Distance based:
0 i->i(0.0000), love->like(2.0000), ->to(2.0000), ->watch(5.0000), novels->movies(3.0000) cost: 12.0
1 i->i(0.0000), love->like(2.0000), ->to(2.0000), ->read(4.0000), novels->books(4.0000) cost: 12.0
------------------------------------------------
```
2. To use only subword vectors, simply drop the `--word-vec` argument.
```
$./demo.py --ngram-vec $NGRAM_EMBEDDING_FILE --minn 1 --maxn 4 --dim 50 --span_size 1
../models/fastext.full-simple-wiki.e.50.m.skipgram.ming.1.maxg.4.winsubs.0.ngrams
Enter chat options (comma seperated):I like to watch movies, I like to read books
Enter learner input:I love novels
Cosine Distance based:
0 i->i(-0.0000), love->like(0.0988), ->to(1.0000), ->watch(1.0000), novels->movies(0.0540) cost: 2.1528
1 i->i(-0.0000), love->like(0.0988), ->to(1.0000), ->read(1.0000), novels->books(0.0316) cost: 2.1304
------------------------------------------------

Levenshtein Distance based:
0 i->i(0.0000), love->like(2.0000), ->to(2.0000), ->watch(5.0000), novels->movies(3.0000) cost: 12.0
1 i->i(0.0000), love->like(2.0000), ->to(2.0000), ->read(4.0000), novels->books(4.0000) cost: 12.0
------------------------------------------------
```
3. Contiguous spans size can be changed using `--span_size' option
```
$./demo.py --ngram-vec $NGRAM_EMBEDDING_FILE --minn 1 --maxn 4 --dim 50 --span_size 2
../models/fastext.full-simple-wiki.e.50.m.skipgram.ming.1.maxg.4.winsubs.0.ngrams
Enter chat options (comma seperated):I like to watch movies, I like to read books
Enter learner input:I love novels
Cosine Distance based:
0 i->i(-0.0000), love->like to(0.2144), novels->watch movies(0.1334) cost: 0.3478
1 i->i(-0.0000), love->like to(0.2144), novels->read books(0.1099) cost: 0.3243
------------------------------------------------

Levenshtein Distance based:
0 i->i(0.0000), love->like(2.0000), ->to(2.0000), ->watch(5.0000), novels->movies(3.0000) cost: 12.0
1 i->i(0.0000), love->like(2.0000), novels->to read(4.0000), ->books(5.0000) cost: 11.0
------------------------------------------------
```


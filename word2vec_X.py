from pyspark import SparkContext
import sys
import os
import re
import itertools as it
import random as rd
import numpy as np
from pyspark.mllib.feature import Word2Vec

if __name__ == '__main__':
    sc = SparkContext('local[40]', 'word2vec')
    if __name__ == '__main__':
        input = 'controller/*/c0-0c0s[48]/messages-*'
        #input = 'messages-2015021[12].cleansed'
        inp = sc.wholeTextFiles(input).map(lambda cor__name: cor__name[1])\
            .map(lambda strs: re.sub('0x[0-9a-f]+', 'mem_addr', strs)).map(lambda strs: re.sub('[0-9]+', '*', strs))\
            .map(lambda corpus: corpus.split('\n'))


    #local test
    # local = [['a', 's safa', 'ssa fa', 'sss ss', 'ssa', 'sss ss', 'sss ss', 'sss ss',
    #           'sss ss', 'sss ss', 'ssa', 'ssa', 'ssa', 'ssa', 'ssa'], ['ssasdf', 'safasdf', 'ssafasf',
    #         'ssdffff', 'aaaaabbb,', 'ssa', 'ssa', 'ssa', 'ssa', 'sssss', 'sssss', 'sssss', 'sss ss', 'sss ss',
    #         'sss ss', 'ssa', 'sss ss', 'sss ss', 'sss ss', 'sss ss', 'sss ss', 'ssa', 'ssa', 'ssa',
    #                                                                    'ssa', 'ssa', 'ssa', 'ssa']]
    # k = sc.parallelize(local)
    model = Word2Vec().setVectorSize(50).setSeed(42).setMinCount(1).fit(inp)
    out = model.getVectors()
    with open ('word2vec_dic_c0s48', 'w') as f:
        f.write(repr(out))

    sn = model.findSynonyms('<*>* *-*-*T*:*:*-*:* c*-*c*s* kernel - - -  micro_i*c_xfer_multi: *:mem_addr bad i*c rc value, i*c rc=mem_addr', 5)

    with open ('test4', 'w') as f1:
         f1.write(repr(sn) + '\n')


    #inpK = inp.map(lambda x: x[0].split('\n'))

    #print inpK.collect()


    # word2vec = Word2Vec()
    # model = word2vec.fit(inpK)
    # synonyms = model.findSynonyms('1', 5)
    #
    # for word, cosine_distance in synonyms:
    #     print("{}: {}".format(word, cosine_distance))
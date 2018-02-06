from pyspark import SparkContext
import sys
import os
import re
import itertools as it
import random as rd
import numpy as np

if __name__ == '__main__':
    sc = SparkContext('local[1]', 'formatTester')
    output = 'test'
    k = sc.textFile('testData/._messages-20150211.cleansed')
    k.collect()
    print k
    def tpSearch(tpLines, tp):
        for tps in tpLines:
            if tp == tps:
                return 1
        return 0

    # term pair value
    # this is test
    tRdd = sc.parallelize([[('term', 'pair'), ('term', 'value'), ('pair', 'value')], [('term', 'pair'), ('this', 'test'), ('is', 'test')]]).cache()

    C = tRdd.collect()
    pt = 0

    f = open(output, 'w')
    for x in C:
        for tp in x:
            n = tRdd.map(lambda tpLines: tpSearch(tpLines, tp)).reduce(lambda x, y: x + y)
            f.write('n = ' + repr(n) + '\n')

            pt = pt + n * np.square(1.*n/len(C))
            f.write('pt = ' + repr(pt) + '\n')

    f.close()

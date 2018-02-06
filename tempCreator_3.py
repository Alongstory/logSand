from pyspark import SparkContext
import sys
import os
import re
import itertools as it
import random as rd
import numpy as np

if __name__ == '__main__':
    sc = SparkContext('local[40]', 'coalesce')
    okok_lis = os.listdir('OKOK')
    okok_lis = map(lambda dirs: re.sub('OK', '', dirs), okok_lis)
    okok_lis = map(lambda dirs: re.sub(' ', '', dirs), okok_lis)

    def reCover(lines):
        return "".join(lines)

    for op in okok_lis:
        #(1)memo address
        input = 'OKOK/' + op + ' OK/part*'
        output = 'OKOK/' + op + ' OK/' + op + '_pre_pro/'
        sc.textFile(input).map(lambda strs: re.sub('0x[0-9a-f]+', 'mem_addr', strs))\
        .map(lambda strs: re.sub('[0-9]+', '*', strs))\
        .map(lambda strs: (strs, 1))\
        .reduceByKey(lambda x, y: x + y)\
        .coalesce(1).saveAsTextFile(output)
        #.map(lambda words: re.sub('[0-9]+:|[0-9]+\.|[0-9]+]', '*', words))\
        #.map(lambda lines: reMemo(lines))
from pyspark import SparkContext
import sys
import os
import re
import itertools as it
import random as rd
import numpy as np

if __name__ == '__main__':
    sc = SparkContext('local[40]', 'temGather')
    okok_lis = os.listdir('OKOK')
    okok_lis = map(lambda dirs: re.sub('OK', '', dirs), okok_lis)
    okok_lis = map(lambda dirs: re.sub(' ', '', dirs), okok_lis)

    for op in okok_lis:
        input = 'OKOK/*/*pro/part*'
        output = 'templates_ALL'
        sc.textFile(input).coalesce(1).saveAsTextFile(output)
    
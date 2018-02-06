"DIVEDE all the logs according to operators"
from pyspark import SparkContext
import sys
import os
import re
import itertools as it
import random as rd
import numpy as np

if __name__ == '__main__':
    sc = SparkContext('local[40]', 'divider')
    #input = 'testData/c0-0c0s1/messages-201502*'
    #input = 'controller/*/*/messages-*'
    input = 'controller/*/*/messages-*'
    inputOP = 'op'
    pre_log = sc.textFile(input)\
    .map(lambda lines: lines.encode("utf-8"))\
    .map(lambda lines: lines.split()).cache()

    print pre_log

    operator = []
    with open(inputOP, 'r') as f:
        for op in f:
            operator.append(op.rstrip())

    for op in operator:
        f_name = op + ' OK'
        pre_log.filter(lambda lines: lines[3] == op).coalesce(1).saveAsTextFile('OKOK/' + f_name)




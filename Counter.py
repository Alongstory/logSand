from pyspark import SparkContext
import sys
import os
import re
import itertools as it
import random as rd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
from operator import add


if __name__ == '__main__':
    sc = SparkContext('local[40]', 'count')
    input = 'controller/*/c0-0c0s4/messages-*'
    if __name__ == '__main__':
        zip_log = sc.textFile(input).map(lambda strs: re.sub('0x[0-9a-f]+', 'mem_addr', strs))\
            .map(lambda strs: re.sub('[0-9]+', '*', strs))\
            .map(lambda strs: 1)\
            .reduce(lambda a, b: a + b)
        with open ('counter_results/c0-0c2s10/counter', 'w') as f:
            f.write(repr(zip_log))


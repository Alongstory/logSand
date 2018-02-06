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
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

if __name__ == '__main__':
    sc = SparkContext('local[40]', 'count')
    input = 'c0-0*/messages-*'
    zip_log = sc.textFile(input).map(lambda strs: re.sub('0x[0-9a-f]+', 'mem_addr', strs))\
        .map(lambda strs: re.sub('[0-9]+', '*', strs)) \
        .map(lambda strs: (strs, 1))\
        .reduceByKey(lambda a, b: a + b)\
        .map(lambda (x, y): (y, x))\
        .sortByKey()\
        .map(lambda (x, y): y)\
        .zipWithIndex()\
        .coalesce(1).saveAsTextFile('featureDic')


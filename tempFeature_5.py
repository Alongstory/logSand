from pyspark import SparkContext
import sys
import os
import re
import itertools as it
import random as rd
import numpy as np

if __name__ == '__main__':
    sc = SparkContext('local[40]', 'temGather')
    temp = sc.textFile('templates_ALL/part*')
    temp.map(lambda x: eval(x)).map(lambda x: ' '.join(eval(x[0]))).coalesce(1).saveAsTextFile('tempJoined')

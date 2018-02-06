from pyspark import SparkContext
import sys
import os
import re
import itertools as it
import random as rd
import numpy as np

if __name__ == '__main__':
    sc = SparkContext('local[40]', 'coalesce')
    ok_lis = os.listdir('OK')
    for dir in ok_lis:
        input = 'OK/' + dir + '/part-*'
        pre_temp = sc.textFile(input).coalesce(1).saveAsTextFile('OKOK/' + dir)
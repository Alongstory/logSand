from pyspark import SparkContext
import sys
import os
import re
import itertools as it
import random as rd
import numpy as np
from datetime import datetime as dt
from operator import add

def tStamp(line):
    '<13>1 2015-02-11T17:09:07-06:00 c0-0 cvcd 7278 - -  cpu_load_watermark_error: changed from 0.00 to 30.00 (10.00-20.00-30.00)  (script)'
    temp = line.split(' ', 2)
    temp = re.split(':|-|T', temp[1])
    temp = '-'.join(temp[0:3]) + ' ' + ':'.join(temp[3:6])
    temp = dt.strptime(temp, "%Y-%m-%d %H:%M:%S")
    return (temp, line)

if __name__ == '__main__':
    sc = SparkContext('local[40]', 'test')
    k = sc.textFile('messages-20150211.cleansed')
    a = k.map(lambda line: tStamp(line)).sortByKey().coalesce(1).saveAsTextFile('timeSort')
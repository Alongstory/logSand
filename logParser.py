from pyspark import SparkContext
import sys
import os
import re
import itertools as it
import random as rd
import numpy as np
import time
import gc

if __name__ == '__main__':
    #spark initial
    sc = SparkContext('local[40]', 'log spider')


    #para define
    #system para
    #input = 'messages-20150211.cleansed'
    input = 'testData/c0-0c0s1/messages-2015021[23456]*' #test/*'
    output1 = 'cluster_dic'
    output2 = 'cluster_before_sig'
    output3 = 'cluster_siged'
    ex_time = []

    #preprocessing para
    weekdays = ['Mon', 'Tue', 'Wed', 'Thr', 'Fri', 'Sat', 'Sun']

    #cluster para
    k = 10                      #cluster num
    cluster_cur = []            #cluster
    pt = 0                      #potential
    tp_occur_map = []           #the occurrence number map for each term pair
    pt_diff = 0                 #difference of moving single logs
    pt_diff_max = [0, 0]        #for comparison, first one is cluster number, second value
    halfLen = 0                 #half length of each cluster

    ###create dictionary for term pair numbers & cluster length & cluster potential, and signature candidate list
    cluster_dic = {}
    sigCandi = []
    cluster_len_pt = {}

    for i in range(0, k):
        cluster_dic[i] = {}
        cluster_len_pt[i] = {}
        sigCandi.append([])

    #function define
    def special(log):
        "for special logs started with weekdays after removed the first n columns"
        if (log[0] in weekdays):
            return log[9:]
        return log

    def tpGen(lines):
        "generating the term pairs"
        return list(it.combinations(lines, 2))


    def tpSearch(tpLines, tp):
        "search the occurance of a term pair tp in a line"
        for tps in tpLines:
            #the numbers pairs should not be take into consideration
            if tp != ('*', '*') and tp == tps:
                return 1
        return 0

    #step 0: pre-processing
    pre_log = sc.textFile(input)
    log = pre_log.map(lambda log: log.encode("utf-8").strip())\
    .map(lambda lines: re.sub('[:\|;,]', ' ', lines))\
    .map(lambda lines: re.sub('=', ' = ', lines)) \
    .map(lambda lines: re.sub(' \d* ', ' * ', lines))
    .map(lambda lines: re.split('\s+', lines))\
    .map(lambda lines: lines[10:])\
    .map(lambda lines: special(lines))\
    #removed all the ':', '|', ';'
    #seperated all the '=' from strings
    #TODO: (1)'.' after some numbers (2)'-' after all the operations (3) numbers attached with variables should all be removed
    # (4)timestamps and other system info could be replaced as 'timestamp', 'sysinfo'(like temprature system or electrical system, tagged them out)

    #logSig Method
    #step 1: term pair generation
    tPairs = log.map(lambda lines: tpGen(lines))

    #step 2: clustering initial & local search of clusters
    ###pre-cluster, divide all log lines into K groups randomly
    #tPairs.map(lambda tpLines: len(tpLines))###check later if LENGTH could be the metrics for the initial clustering

    pre_cluster_tpLines = tPairs.map(lambda tpLines: (rd.randint(0, k - 1), tpLines))\
    .groupByKey()\
    .mapValues(list)\
    .cache()
    #TODO:unpersist after every iteration

    ex_time.append(time.time())

    ###calculate the potential of initial clusters(take the longest time, need to be improved, timer here to improve )
    for i in range(0, k):
        #collect cluster "i" rdd as list
        # timer 11
        ex_time.append(time.time() - ex_time[0])

        cluster_cur.append(pre_cluster_tpLines.filter(lambda x: x[0] == i).values().collect())

        # timer 22
        ex_time.append(time.time() - ex_time[0])

        cluster_cur[i] = cluster_cur[i][0]          #one more '[ ]' out of the data after collect, remove it by this op
        cluster_len = len(cluster_cur[i])
        cluster_len_pt[i]['length'] = cluster_len
        #for each term pair in cluster, do a local search
        for lines in cluster_cur[i]:
            for tp in lines:
                #the occurrence "n" of a term pair
                tp_occur_map = map(lambda tpLines: tpSearch(tpLines, tp), cluster_cur[i])
                n = reduce(lambda x, y: x + y, tp_occur_map)
                #add single term pair occurrence score to the total potential
                pt = pt + n * np.square(1. * n / cluster_len)
                tp_occur_map = []               #clear tp occur map
                cluster_dic[i][tp] = n          #store the occurrence of term pair in cluster i into a table
        cluster_len_pt[i]['potential'] = pt
        pt = 0                                  #clear potential value

    #timer end
    ex_time.append(time.time() - ex_time[0])

    #step 3: optimizing clusters by moving & calculating cluster potential difference
    for i in range(0, k):
        for index, lines in enumerate(cluster_cur[i]):
            for j in (range(0, i) + range(i + 1, k)):
                #as for one sentence, for each group(except for self), calculate the potential diffrence after moving
                for tp in lines:
                    #check if tp exits in J, if not add a 0 to its dic
                    if not tp in cluster_dic[j]:
                        cluster_dic[j][tp] = 0
                    #main calculate
                    pt_diff = pt_diff + 3 * ((1.*cluster_dic[j][tp]/(1.*cluster_len_pt[j]['length'])) ** 2 - (1.*cluster_dic[i][tp]/(1.*cluster_len_pt[i]['length'])) ** 2)
                #store the largest difference
                if pt_diff > pt_diff_max[1]:
                    pt_diff_max[0] = j
                    pt_diff_max[1] = pt_diff
                pt_diff = 0                     #clear potential difference
            #if log move is better(max is positive), move it
            if pt_diff_max[1] > 0:
                cluster_cur[pt_diff_max[0]].append(lines)
                del cluster_cur[i][index]       #it could be during the map coz the former-used lines wouldn't be used again(compare with other clusters not self cluster)
                #update the dic after moving
                cluster_len_pt[i]['length'] -= 1
                cluster_len_pt[j]['length'] += 1
                for tp in lines:
                    cluster_dic[i][tp] -= 1
                    cluster_dic[j][tp] += 1

    #data visualization
    f = open(output1, 'w')
    f.write(repr(cluster_dic))
    f.close()
    f = open(output2, 'w')
    f.write(repr(cluster_cur[9]))
    f.close()

    #step 4: generate the message signature
     # for i in range(0, k):
     #     for index, lines in enumerate(cluster_cur[i]):
     #        for tp in lines:
     #            if not cluster_dic[i][tp] >= 1.*cluster_dic[i]['length']/2:
     #                del cluster_cur[i][index]
     #                break

    for i in range(0, k):
        halfLen = 1. * cluster_len_pt[i]['length'] / 2
        for index, lines in enumerate(cluster_cur[i]):
            for tp in lines:
                if not cluster_dic[i][tp] >= halfLen:
                    del cluster_cur[i][index]
                    break
    halfLen = 0     #clear halfLen

    sigLine = ''
    #TODO: From term pairs back to message
    for i in range(0, k):
        for lines in cluster_cur[i]:
            for index, tp in enumerate(lines):
                if index == len(lines) - 1:
                    sigLine += tp[0]
                elif tp[0] != lines[index+1][0]:
                        sigLine += (tp[0] + ' ')
            sigCandi[i].append(sigLine)
            sigLine = ''


    # data visualization
    f = open('sigCandi', 'w')
    f.write(repr(sigCandi))
    f.close()

    f = open('time', 'w')
    f.write(repr(ex_time))
    f.close()

    gc.collect()


from pyspark import SparkContext
import sys
import os
import re
import itertools as it
import random as rd
import numpy as np
from operator import add
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel

def window_div_fixed(x):
    if x % w_size == 0:
        return x / w_size
    else:

        return x / w_size + 1

def window_div_sliding(x):
    if x < w_size:
        return range(x, x + 1)
    else:
        return range(x - w_size + 1, x)

def window_calc(x):
    temp = map(lambda line: w2v_dic[line], x)
    return np.mean(temp, axis = 0)

def label(x, y):
    if x in target_log_window_code:
        return LabeledPoint(1.0, y)
    else:
        return LabeledPoint(0.0, y)

def label2(x, y):
    if x in target_log_window_code2:
        return LabeledPoint(1.0, y)
    else:
        return LabeledPoint(0.0, y)

def tnCount(lp):
    global TP              #true positive
    global TN              #true negative
    global FP              #false positive
    global FN              #false negative
    if lp[0] == lp[1]:
        if lp[0] == 1.0:
            TP = TP + 1
        else:
            TN = TN + 1
    elif lp[0] == 0.0:
        FP = FP + 1
    else:
        FN = FN + 1
    TP = TP + 1
    return lp

def tStamp():
    pass


if __name__ == '__main__':
    sc = SparkContext('local[40]', 'window')

    #inputLog = 'messages-2015021[12].cleansed'
    inputLog = 'controller/*/*/messages-*'
    #test set
    inputLog2 = 'controller/*/c0-0c0s2/messages-*'
    inputDic = 'word2vec_dic_all'

    window_type = 0     # fixed 0, sliding 1
    w_size = 5          # window size
    mlAlgo = 0          #SVM 0, Logistic Regression 1, neural network 2, K_means clustering 3
    TP = 0              #true positive
    TN = 0              #true negative
    FP = 0              #false positive
    FN = 0              #false negative

    #preprocessing
    log = sc.textFile(inputLog).map(lambda strs: re.sub('0x[0-9a-f]+', 'mem_addr', strs)).map(lambda strs: re.sub('[0-9]+', '*', strs))
    
    log2 = sc.textFile(inputLog2).map(lambda strs: re.sub('0x[0-9a-f]+', 'mem_addr', strs)).map(lambda strs: re.sub('[0-9]+', '*', strs))
    #code_log = log.zipWithIndex().coalesce(1).saveAsTextFile('zipped_File_X')


    #code all the logs, let the code start from 1
    zip_log = log.zipWithIndex().mapValues(lambda code: code + 1)
    zip_log2 = log2.zipWithIndex().mapValues(lambda code: code + 1)
    num_log = zip_log.count()
    #target_log = '<13>1 2015-02-11T17:09:07-06:00 c0-0 cvcd 7278 - -  cpu_load_watermark_error: changed from 0.00 to 30.00 (10.00-20.00-30.00)  (script)'
    target_error = 'cpu_load_watermark_error: changed from *.* to *.*'


    if window_type == 0:
        #read the dictonary of word2vec feature vectors
        w2v_dic = eval(open(inputDic, 'r').read())
        #create start&end special window para

        windows = zip_log.map(lambda (x, y): (y, x))\
            .map(lambda (x, y): (window_div_fixed(x), y))\
            .groupByKey()\
            .mapValues(lambda x: window_calc(x))\
            .sortByKey().cache()
        #.coalesce(1).saveAsTextFile('windows_Fixed')
        window2 = zip_log2.map(lambda (x, y): (y, x))\
            .map(lambda (x, y): (window_div_fixed(x), y))\
            .groupByKey()\
            .mapValues(lambda x: window_calc(x))\
            .sortByKey().cache()

        #labeling pre-process for supervised algorithms
        target_log_window_code = zip_log.filter(lambda (x, y): target_error in x).values().map(lambda x: x - w_size).collect()
        target_log_window_code2 = zip_log2.filter(lambda (x, y): target_error in x).values().map(lambda x: x - w_size).collect()

        if mlAlgo == 0:
            #label the data
            labeled_window = windows.map(lambda (x, y): label(x, y)).cache()
            labeled_window2 = window2.map(lambda (x, y): label2(x, y)).cache()
            #SVM model
            SVM_model = SVMWithSGD.train(labeled_window, iterations = 100)
            SVM_labelsAndPreds = labeled_window.map(lambda p: (p.label, SVM_model.predict(p.features))).cache()
            SVM_trainErr = SVM_labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(labeled_window.count())
            TP = SVM_labelsAndPreds.filter(lambda lp: (lp[0] == lp[1]) & (lp[0] == 1)).count()
            TN = SVM_labelsAndPreds.filter(lambda lp: (lp[0] == lp[1]) & (lp[0] == 0)).count()
            FP = SVM_labelsAndPreds.filter(lambda lp: (lp[0] != lp[1]) & (lp[0] == 0)).count()
            FN = SVM_labelsAndPreds.filter(lambda lp: (lp[0] != lp[1]) & (lp[0] == 1)).count()

            print("Training Error = " + str(SVM_trainErr))
            with open('SVM_err_fixed', 'w') as f:
                f.write(repr(SVM_trainErr))
            f.close()

            with open('SVM_Measure_fixed', 'w') as f1:
                f1.write('TP' + repr(TP))
                f1.write('TN' + repr(TN))
                f1.write('FP' + repr(FP))
                f1.write('FN' + repr(FN))
            f1.close()

        elif mlAlgo == 1:
            # label the data
            labeled_window = windows.map(lambda (x, y): label(x, y)).cache()
            #Logistic Regression
            LG_model = LogisticRegressionWithLBFGS.train(labeled_window)
            LG_labelsAndPreds = labeled_window.map(lambda p: (p.label, LG_model.predict(p.features)))
            LG_trainErr = LG_labelsAndPreds.map(lambda lp: tnCount(lp)).filter(lambda lp: lp[0] != lp[1]).count() / float(labeled_window.count())
            with open('LG_err_fixed', 'w') as f:
                f.write(repr(LG_trainErr))

        elif mlAlgo == 2:
            pass

        elif mlAlgo == 3:
            pass





    elif window_type == 1:
        #read the dictonary of word2vec feature vectors
        w2v_dic = eval(open(inputDic, 'r').read())

        windows = zip_log.map(lambda (x, y): (x, window_div_sliding(y)))\
            .flatMapValues(lambda x: x)\
            .map(lambda (x, y): (y, x))\
            .groupByKey()\
            .mapValues(lambda x: window_calc(x))\
            .sortByKey().cache()
        #.coalesce(1).saveAsTextFile('windows_Sliding')

        #fetching target window array before target_log
        target_log_window_code = zip_log.filter(lambda (x, y): target_error in x).values().map(lambda x: x - w_size).collect()

        if mlAlgo == 0:
            #label the data
            labeled_window = windows.map(lambda (x, y): label(x, y)).cache()
            #SVM model
            SVM_model = SVMWithSGD.train(labeled_window, iterations = 100)
            SVM_labelsAndPreds = labeled_window.map(lambda p: (p.label, SVM_model.predict(p.features)))
            SVM_trainErr = SVM_labelsAndPreds.map(lambda lp: tnCount(lp)).filter(lambda lp: lp[0] != lp[1]).count() / float(labeled_window.count())
            print("Training Error = " + str(SVM_trainErr))
            with open('SVM_err_sliding', 'w') as f:
                f.write(repr(SVM_trainErr))

        elif mlAlgo == 1:
            # label the data
            labeled_window = windows.map(lambda (x, y): label(x, y)).cache()
            #Logistic Regression
            LG_model = LogisticRegressionWithLBFGS.train(labeled_window)
            LG_labelsAndPreds = labeled_window.map(lambda p: (p.label, LG_model.predict(p.features)))
            LG_trainErr = LG_labelsAndPreds.map(lambda lp: tnCount(lp)).filter(lambda lp: lp[0] != lp[1]).count() / float(labeled_window.count())
            with open('LG_err_Sliding', 'w') as f:
                f.write(repr(LG_trainErr))






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

def window_div_sliding(date_time, w_interval):
    '2015-02-11 17:09:07'
    return tuple(filter(lambda x: (date_time >= x[0]) & (date_time <= x[1]) , w_interval))

def window_calc(x, cal_type, dic):
    if cal_type == 0:
        temp = map(lambda line: w2v_dic[line], x)
        return np.mean(temp, axis=0)
    elif cal_type == 1:
        temp = map(lambda line: dic[line], x)
        return np.mean(temp, axis=0)


def target_window_finder(t, t_w_interval):
    '2015-02-11 17:09:07 construct the target window code stamp'
    return tuple(filter(lambda x: (t >= x[0]) & (t <= x[1]), t_w_interval))

def label(t, y, target_window_interval):
    if t in target_window_interval:
        return LabeledPoint(1.0, y)
    else:
        return LabeledPoint(0.0, y)

def tStamp(line):
    '<13>1 2015-02-11T17:09:07-06:00 c0-0 cvcd 7278 - -  cpu_load_watermark_error: changed from 0.00 to 30.00 (10.00-20.00-30.00)  (script)'
    temp = line.split(' ', 2)
    temp = re.split(':|-|T', temp[1])
    temp = '-'.join(temp[0:3]) + ' ' + ':'.join(temp[3:6])
    temp = dt.strptime(temp, "%Y-%m-%d %H:%M:%S")
    seconds = (temp - dt(1970, 1, 1)).total_seconds()
    return (seconds, line)


if __name__ == '__main__':
    #Spark configure
    sc = SparkContext('local[40]', 'window')

    #training set
    inputLog = 'controller/*/c0-0c0s4/messages-*'
    #inputLog = 'messages-2015021[12].cleansed'
    target_error = 'micro_i*c_xfer_multi: *:mem_addr bad i*c rc value, i*c rc=mem_addr'
    #test set
    inputLog_test = 'controller/*/c0-0c0s8/messages-*'

    #window parameters
    w_size = 5     # window size, time interval(min)
    w_step = 2      # step size should be less or equal than window size(unless resampling)
    w_number = 0     # window numbers

    w_interval = []
    w_interval_test = []
    #algorithm choice
    mlAlgo = 0      # SVM 0, Logistic Regression 1, neural network 2, K_means clustering 3, decision tree 4
    cal_type = 1    # word2vec 0, event count 1
    if cal_type == 0:
        #feature vector dictionary
        inputDic = 'word2vec_dic_c0s48'
        inputDic_all = 'word2vec_dic_all'
        #read the dictonary of word2vec feature vectors
        w2v_dic = eval(open(inputDic, 'r').read())
        #w2v_dic_all = eval(open(inputDic_all, 'r').read())
    elif cal_type == 1:
        inputDic = 'feature_dict'
        feature_dict = eval(open(inputDic, 'r').read())

    #evaluation parameters
    TP = 0  # true positive
    TN = 0  # true negative
    FP = 0  # false positive
    FN = 0  # false negative

    ####################################################################################################################
    #pre-processing
    #code timestamp for all the logs, replace memory address and numbers
    zip_log = sc.textFile(inputLog).map(lambda line: tStamp(line))\
        .map(lambda (t, strs): (t, re.sub('0x[0-9a-f]+', 'mem_addr', strs)))\
        .map(lambda (t, strs): (t, re.sub('[0-9]+', '*', strs))).sortByKey().cache()
    zip_log_test = sc.textFile(inputLog_test).map(lambda line: tStamp(line))\
        .map(lambda (t, strs): (t, re.sub('0x[0-9a-f]+', 'mem_addr', strs)))\
        .map(lambda (t, strs): (t, re.sub('[0-9]+', '*', strs))).sortByKey().cache()

    # [(timestamp, u'log informations here'), (...)]
    # code_log = log.zipWithIndex().coalesce(1).saveAsTextFile('zipped_File_X')

    #total window numbers
    stamps = zip_log.keys().collect()
    stStart = stamps[0]
    stEnd = stamps[-1]
    w_number = int((stEnd - stStart - w_size) / w_step + 1)

    stamps_test = zip_log_test.keys().collect()
    stStart_test = stamps_test[0]
    stEnd_test = stamps_test[-1]

    w_number_test = int((stEnd_test - stStart_test - w_size) / w_step + 1)

    # window distribution
    for i in range(0, w_number):
        w_interval.append((stStart + i * w_step, stStart + i * w_step + w_size))
    w_interval.append((stStart + w_number * w_step, stEnd))
    # [(window start time, window end time), (window 1 start time, window 2 end time), (...)]
    for i in range(0, w_number_test):
        w_interval_test.append((stStart_test + i * w_step, stStart_test + i * w_step + w_size))
    w_interval_test.append((stStart_test + w_number_test * w_step, stEnd_test))
    # [(window start time, window end time), (window 1 start time, window 2 end time), (...)]



    # target extra positive windows construct & labelling
    target_window_interval = zip_log.filter(lambda (t, y): target_error in y).keys().map(lambda x: (x - w_size, x)).collect()
    target_window_interval_test = zip_log_test.filter(lambda (t, y): target_error in y).keys().map(lambda x: (x - w_size, x)).collect()

    # [(target window start time, target window end time), (target window 1 start time, target window 2 end time), (...)]
    target_windows = zip_log.map(lambda (t, y): (target_window_finder(t, target_window_interval), y))\
                        .filter(lambda (t, y): t != ())\
                        .map(lambda (t, y): (y, t))\
                        .flatMapValues(lambda group: group)\
                        .map(lambda (y, t): (t, y))\
                        .groupByKey()\
                        .map(lambda (x, y): list(y))\
                        .mapValues(lambda y: window_calc(y, cal_type, dic = feature_dict))\
                        .map(lambda (t, y): LabeledPoint(1.0, y))
    #[LabeledPoint(1.0, window array 1), LabeledPoint(1.0, window array 2), (...)]
    
    #construct main windows & labelling
    windows = zip_log.map(lambda (t, y): (y, window_div_sliding(t, w_interval)))\
        .flatMapValues(lambda group: group)\
        .map(lambda (y, t): (t, y))\
        .groupByKey()\
        .mapValues(lambda y: window_calc(y, cal_type, dic = feature_dict))\
        .map(lambda (t, y): label(t, y, target_window_interval))
    # [LabelPoint(1.0, window array 1), LabelPoint(0.0, window array 2), LabelPoint(...)]

    # construct test windows & labelling
    windows_test = zip_log.map(lambda (t, y): (y, window_div_sliding(t, w_interval_test)))\
        .flatMapValues(lambda group: group)\
        .map(lambda (y, t): (t, y))\
        .groupByKey()\
        .mapValues(lambda y: window_calc(y, cal_type, dic = feature_dict))\
        .map(lambda (t, y): label(t, y, target_window_interval_test))
    # [LabelPoint(1.0, window array 1), LabelPoint(0.0, window array 2), LabelPoint(...)]

    # merge windows
    labeled_window = windows.union(target_windows)

    ####################################################################################################################
    #start training
    
    if mlAlgo == 0:
        SVM_model = SVMWithSGD.train(labeled_window, iterations=100)
        SVM_labelsAndPreds = labeled_window.map(lambda p: (p.label, SVM_model.predict(p.features))).cache()
        SVM_trainErr = SVM_labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(labeled_window.count())
        TP = SVM_labelsAndPreds.filter(lambda lp: (lp[0] == 1) & (lp[1] == 1)).count()
        TN = SVM_labelsAndPreds.filter(lambda lp: (lp[0] == 0) & (lp[1] == 0)).count()
        FP = SVM_labelsAndPreds.filter(lambda lp: (lp[0] == 0) & (lp[1] == 1)).count()
        FN = SVM_labelsAndPreds.filter(lambda lp: (lp[0] == 1) & (lp[1] == 0)).count()

        print("Training Error = " + str(SVM_trainErr))
        with open('SVM_err', 'w') as f:
            f.write(repr(SVM_trainErr))
        f.close()

        with open('SVM_Measure', 'w') as f1:
            f1.write('TP:' + repr(TP) + '\n')
            f1.write('TN:' + repr(TN) + '\n')
            f1.write('FP:' + repr(FP) + '\n')
            f1.write('FN:' + repr(FN) + '\n')
        f1.close()

    elif mlAlgo == 4:
        trainingData = labeled_window
        testData = windows_test
        model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=5, maxBins=32)
        predictions = model.predict(testData.map(lambda x: x.features))
        labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
        testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
        TP = labelsAndPredictions.filter(lambda lp: (lp[0] == 1) & (lp[1] == 1)).count()
        TN = labelsAndPredictions.filter(lambda lp: (lp[0] == 0) & (lp[1] == 0)).count()
        FP = labelsAndPredictions.filter(lambda lp: (lp[0] == 0) & (lp[1] == 1)).count()
        FN = labelsAndPredictions.filter(lambda lp: (lp[0] == 1) & (lp[1] == 0)).count()
        print('Test Error = ' + str(testErr))
        print(model.toDebugString())

        with open('DT_err', 'w') as f:
            f.write(repr(testErr))
        f.close()

        with open('DT_Measure', 'w') as f1:
            f1.write('TP:' + repr(TP) + '\n')
            f1.write('TN:' + repr(TN) + '\n')
            f1.write('FP:' + repr(FP) + '\n')
            f1.write('FN:' + repr(FN) + '\n')
        f1.close()



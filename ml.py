from __future__ import division
from collections import defaultdict
import random
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

param = defaultdict(list)

f_name = [sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]
with open('train1.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[7]]
with open('train2.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[3], sys.argv[4], sys.argv[6], sys.argv[7]]
with open('train3.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[3], sys.argv[5], sys.argv[6], sys.argv[7]]
with open('train4.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]]
with open('train5.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

t_file1 = tuple(open("train1.txt","r"))
t_file2 = tuple(open("train2.txt","r"))
t_file3 = tuple(open("train3.txt","r"))
t_file4 = tuple(open("train4.txt","r"))
t_file5 = tuple(open("train5.txt","r"))

split_file1 = tuple(open(sys.argv[3], "r"))
split_file2 = tuple(open(sys.argv[4], "r"))
split_file3 = tuple(open(sys.argv[5], "r"))
split_file4 = tuple(open(sys.argv[6], "r"))
split_file5 = tuple(open(sys.argv[7], "r"))

def simple(learning_rate, weight, bias, t_file, updates):
    line_array = random.sample(t_file, len(t_file))
    for t in range(len(t_file)):
        line = line_array[t]
        feature = []
        index = []
        value = []
        line = line.strip("\n").split()
        output = line[0]
        output = int(output)
        for i in range(len(line)):
            if i != 0:
                wording = line[i].split(":")
                index.append(int(wording[0]))
                value.append(float(wording[1]))
        indx_margin = 0
        for apnd in range(69):
            if apnd == index[indx_margin]:
                feature.append(value[indx_margin])
                if indx_margin < (len(index)-2):
                    indx_margin = indx_margin + 1
            else:
                feature.append(0)
        Sum = 0
        for i in range(69):
            Sum = Sum + (weight[i] * feature[i])
        res_sum = Sum + bias
        if (output * res_sum) < 0:
            updates = updates + 1
            bias = bias + (learning_rate * output)
            for w in range(69):
                x = float(learning_rate * output * feature[w])
                weight[w] = weight[w] + x
    return (weight, bias, updates)

def dynamic(dynamic_rate, weight, bias, t_file, t, updates):
    line_array = random.sample(t_file, len(t_file))
    for z in range(len(t_file)):
        line = line_array[z]
        feature = []
        index = []
        value = []
        line = line.strip("\n").split()
        output = line[0]
        output = int(output)
        for i in range(len(line)):
            if i != 0:
                wording = line[i].split(":")
                index.append(int(wording[0]))
                value.append(float(wording[1]))
        indx_margin = 0
        for apnd in range(69):
            if apnd == index[indx_margin]:
                feature.append(value[indx_margin])
                if indx_margin < (len(index)-2):
                    indx_margin = indx_margin + 1
            else:
                feature.append(0)
        t = t + 1
        dynamic_rate = dynamic_rate/(1 + t)
        Sum = 0
        for i in range(69):
            Sum = Sum + (weight[i] * feature[i])
        res_sum = Sum + bias
        if (output * res_sum) < 0:
            updates = updates + 1
            bias = bias + (dynamic_rate * output)
            for w in range(69):
                x = float(dynamic_rate * output * feature[w])
                weight[w] = weight[w] + x
    return (weight, bias, t, updates)

def marginp(learning_rate, weight, bias, t_file, margin, updates):
    line_array = random.sample(t_file, len(t_file))
    for t in range(len(t_file)):
        line = line_array[t]
        feature = []
        index = []
        value = []
        line = line.strip("\n").split()
        output = line[0]
        output = int(output)
        learning_rate = float(learning_rate)
        for i in range(len(line)):
            if i != 0:
                wording = line[i].split(":")
                index.append(int(wording[0]))
                value.append(float(wording[1]))
        indx_margin = 0
        for apnd in range(69):
            if apnd == index[indx_margin]:
                feature.append(value[indx_margin])
                if indx_margin < (len(index)-2):
                    indx_margin = indx_margin + 1
            else:
                feature.append(0)
        Sum = 0
        for i in range(69):
            Sum = Sum + (weight[i] * feature[i])
        res_sum = Sum + bias
        if (output * res_sum) < margin:
            updates = updates + 1
            bias = bias + (learning_rate * output)
            for w in range(69):
                x = float(learning_rate * output * feature[w])
                weight[w] = weight[w] + x
    return (weight, bias, updates)

def averaged(learning_rate, weight, bias, t_file, updates, avg_w, avg_b):
    line_array = random.sample(t_file, len(t_file))
    for t in range(len(t_file)):
        line = line_array[t]
        feature = []
        index = []
        value = []
        line = line.strip("\n").split()
        output = line[0]
        output = int(output)
        for i in range(len(line)):
            if i != 0:
                wording = line[i].split(":")
                index.append(int(wording[0]))
                value.append(float(wording[1]))
        indx_margin = 0
        for apnd in range(69):
            if apnd == index[indx_margin]:
                feature.append(value[indx_margin])
                if indx_margin < (len(index)-2):
                    indx_margin = indx_margin + 1
            else:
                feature.append(0)
        Sum = 0
        for i in range(69):
            Sum = Sum + (weight[i] * feature[i])
        res_sum = Sum + bias
        if (output * res_sum) < 0:
            updates = updates + 1
            bias = bias + (learning_rate * output)
            for w in range(69):
                x = float(learning_rate * output * feature[w])
                weight[w] = weight[w] + x
        for i in range(69):
            avg_w[i] = avg_w[i] + weight[i]
        avg_b = avg_b + bias
    return (weight, bias, updates, avg_w, avg_b)

def aggresive(learning_rate, weight, t_file, margin, updates):
    line_array = random.sample(t_file, len(t_file))
    for t in range(len(t_file)):
        line = line_array[t]
        feature = []
        index = []
        value = []
        line = line.strip("\n").split()
        output = line[0]
        output = int(output)
        for i in range(len(line)):
            if i != 0:
                wording = line[i].split(":")
                index.append(int(wording[0]))
                value.append(float(wording[1]))
        indx_margin = 0
        for apnd in range(69):
            if apnd == index[indx_margin]:
                feature.append(value[indx_margin])
                if indx_margin < (len(index)-2):
                    indx_margin = indx_margin + 1
            else:
                feature.append(0)
        Sum = 0
        for i in range(69):
            Sum = Sum + (weight[i] * feature[i])
        #res_sum = Sum + bias
        if (output * Sum) <= margin:
            updates = updates + 1
            for w in range(69):
                feature[w] = feature[w] * feature[w]
                num = margin - (output * Sum)
                learning_rate = num/(feature[w] + 1)
                x = float(learning_rate * output * feature[w])
                weight[w] = weight[w] + x
    return (weight, updates)

def dev(weight, bias, d_file, agg):
    indexD = []
    valueD = []
    accuracy = 0.0
    instances = 0.0
    line_arrayD = random.sample(d_file, len(d_file))
    for s in range(len(d_file)):
        lineD = line_arrayD[s]
        featureD = []
        indexD = []
        valueD = []
        instances = instances + 1
        lineD = lineD.strip("\n").split()
        outputD = lineD[0]
        outputD = int(outputD)
        for iD in range(len(lineD)):
            if iD != 0:
                wordingD = lineD[iD].split(":")
                indexD.append(int(wordingD[0]))
                valueD.append(float(wordingD[1]))
        indx_marginD = 0
        for apndD in range(69):
            if apndD == indexD[indx_marginD]:
                featureD.append(valueD[indx_marginD])
                if indx_marginD < (len(indexD)-2):
                    indx_marginD = indx_marginD + 1
            else:
                featureD.append(0)
        SumDev = 0
        for i in range(69):
            SumDev = SumDev + (weight[i] * featureD[i])
        if agg != 1:
            res_sum = SumDev + bias
        else:
            res_sum = SumDev
        if res_sum <= 0:
            r = -1
        else:
            r = 1
        if r == outputD:
            accuracy = accuracy + 1
    return (accuracy/instances) * 100

def frequent_label(files):
    neg = pos = 0.0
    for line in files:
        line = line.strip("\n").split()
        line[0] = int(line[0])
        if line[0] == -1:
            neg = neg + 1
        else:
            pos = pos + 1
    if neg > pos:
        label = -1
    else:
        label = 1
    return label

def freq(label, files):
    instances = correct = 0.0
    for line in files:
        instances = instances + 1
        line = line.strip("\n").split()
        line[0] = int(line[0])
        if line[0] == label:
            correct = correct + 1
    return (correct/instances) * 100

def main():
    rate = [1, 0.1, 0.01]
    margin = [1, 0.1, 0.01]
    print "\nMajority Baseline"
    t_file = tuple(open(sys.argv[8],'r'))
    d_file = tuple(open(sys.argv[2],'r'))
    l = frequent_label(t_file)
    a1 = freq(l, t_file)
    print "Accuracy on Test Set: ", a1
    l = frequent_label(d_file)
    a2 = freq(l, d_file)
    print "Accuracy on Development Set: ", a2
    weight = [0]*69
    bias = random.uniform(-0.01, 0.01)
    for i in range(69):
        z = random.uniform(-0.01, 0.01)
        weight.append(z)
    print "\nSimple Perceptron with Cross Validation"
    avg_acc = updates = 0.0
    for r in rate:
        for i in range(10):
            acc = 0.0
            weight, bias, updates = simple(r, weight, bias, t_file1, updates)
            a = dev(weight, bias, split_file5, 0)
            acc = acc + a
            weight, bias, updates = simple(r, weight, bias, t_file2, updates)
            a = dev(weight, bias, split_file4, 0)
            acc = acc + a
            weight, bias, updates = simple(r, weight, bias, t_file3, updates)
            a = dev(weight, bias, split_file3, 0)
            acc = acc + a
            weight, bias, updates = simple(r, weight, bias, t_file4, updates)
            a = dev(weight, bias, split_file2, 0)
            acc = acc + a
            weight, bias, updates = simple(r, weight, bias, t_file5, updates)
            a = dev(weight, bias, split_file1, 0)
            acc = acc + a
            if i == 9:
                avg_acc = acc/5.0
                param[r] = avg_acc
    best = max(param, key=param.get)
    print "Best hyperparamerter: ", best
    print "Best hyperparamerter's Accuracy: ", param[best]
    param.clear()
    acc = avg_acc = dev_acc = 0.0
    print "\nSimple Perceptron with Best Hyperparameter"
    updates = max_acc = test_bias = end_updates = 0.0
    test_weight = [0]*69
    x = []
    y = []
    for i in range(20):
        t_file = tuple(open(sys.argv[1],'r'))
        d_file = tuple(open(sys.argv[2],'r'))
        weight, bias, updates = simple(best, weight, bias, t_file, updates)
        a = dev(weight, bias, d_file, 0)
        if i != 20:
            x.append(i)
            y.append(a)
        if a > max_acc:
            test_weight = weight
            test_bias = bias
            end_updates = updates
        dev_acc = dev_acc + a
    print "Devlopment Set Acccuracy: ", dev_acc/20.0
    print "Total number of updates on the traning set: ", end_updates
    test_file = tuple(open(sys.argv[8],'r'))
    a = dev(test_weight, test_bias, test_file, 0)
    print "Test Set Acccuracy: ", a
    #plt.plot(x,y)
    #plt.show()
#####################################
    del weight[:]
    weight = [0]*69
    bias = random.uniform(-0.01, 0.01)
    for i in range(69):
        z = random.uniform(-0.01, 0.01)
        weight.append(z)
    print "\nPerceptron with Dynamic Learning Rate with Cross Validation"
    avg_acc = updates = 0
    for r in rate:
        t = 0.0
        for i in range(10):
            acc = 0.0
            weight, bias, t, updates = dynamic(r, weight, bias, t_file1, t, updates)
            a = dev(weight, bias, split_file5, 0)
            acc = acc + a
            weight, bias, t, updates = dynamic(r, weight, bias, t_file2, t, updates)
            a = dev(weight, bias, split_file4, 0)
            acc = acc + a
            weight, bias, t, updates = dynamic(r, weight, bias, t_file3, t, updates)
            a = dev(weight, bias, split_file3, 0)
            acc = acc + a
            weight, bias, t, updates = dynamic(r, weight, bias, t_file4, t, updates)
            a = dev(weight, bias, split_file2, 0)
            acc = acc + a
            weight, bias, t, updates = dynamic(r, weight, bias, t_file5, t, updates)
            a = dev(weight, bias, split_file1, 0)
            acc = acc + a
            if i == 9:
                avg_acc = acc/5.0
                param[r] = avg_acc
        avg_acc = acc/5.0
        param[r] = avg_acc
    best = max(param, key=param.get)
    print "Best hyperparamerter: ", best
    print "Best hyperparamerter's Accuracy: ", param[best]
    param.clear()
    acc = avg_acc = dev_acc = 0.0
    print "\nPerceptron with Dynamic Learning Rate with Best Hyperparameter"
    updates = max_acc = test_bias = t = end_updates = 0.0
    test_weight = [0]*69
    x = []
    y = []
    for i in range(20):
        t_file = tuple(open(sys.argv[1],'r'))
        d_file = tuple(open(sys.argv[2],'r'))
        weight, bias, t, updates = dynamic(best, weight, bias, t_file, t, updates)
        a = dev(weight, bias, d_file, 0)
        if i != 20:
            x.append(i)
            y.append(a)
        if a > max_acc:
            test_weight = weight
            test_bias = bias
            end_updates = updates
        dev_acc = dev_acc + a
    print "Devlopment Set Acccuracy: ", dev_acc/20
    print "Total number of updates on the traning set: ", end_updates
    test_file = tuple(open(sys.argv[8],'r'))
    a = dev(test_weight, test_bias, test_file, 0)
    print "Test Set Acccuracy: ", a
    #plt.plot(x,y)
    #plt.show()
#####################################
    del weight[:]
    weight = [0]*69
    bias = random.uniform(-0.01, 0.01)
    for i in range(69):
        z = random.uniform(-0.01, 0.01)
        weight.append(z)
    print "\nMargin Perceptron with Cross Validation"
    avg_acc = updates = 0
    for r in rate:
        for m in margin:
            for i in range(10):
                acc = 0.0
                weight, bias, updates = marginp(r, weight, bias, t_file1, m, updates)
                a = dev(weight, bias, split_file5, 0)
                acc = acc + a
                weight, bias, updates = marginp(r, weight, bias, t_file2, m, updates)
                a = dev(weight, bias, split_file4, 0)
                acc = acc + a
                weight, bias, updates = marginp(r, weight, bias, t_file3, m, updates)
                a = dev(weight, bias, split_file3, 0)
                acc = acc + a
                weight, bias, updates = marginp(r, weight, bias, t_file4, m, updates)
                a = dev(weight, bias, split_file2, 0)
                acc = acc + a
                weight, bias, updates = marginp(r, weight, bias, t_file5, m, updates)
                a = dev(weight, bias, split_file1, 0)
                acc = acc + a
                if i == 9:
                    avg_acc = acc/5.0
                    r = str(r)
                    m = str(m)
                    key = r + ":" + m
                    param[key] = avg_acc
    best = max(param, key=param.get)
    b = best.split(":")
    print "Best hyperparamerter: Learning Rate is ", b[0], "Margin is ", b[1]
    print "Best hyperparamerter's Accuracy: ", param[best]
    b[0] = float(b[0])
    b[1] = float(b[1])
    param.clear()
    acc = avg_acc = dev_acc = 0.0
    print "\nMargin Perceptron with Best Hyperparameter"
    updates = max_acc = test_bias = end_updates = 0.0
    test_weight = [0]*69
    x = []
    y = []
    for i in range(20):
        t_file = tuple(open(sys.argv[1],'r'))
        d_file = tuple(open(sys.argv[2],'r'))
        weight, bias, updates = marginp(b[0], weight, bias, t_file, b[1], updates)
        a = dev(weight, bias, d_file, 0)
        if x != 19:
            x.append(i)
            y.append(a)
        if a > max_acc:
            test_weight = weight
            test_bias = bias
            end_updates = updates
        dev_acc = dev_acc + a
    print "Devlopment Set Acccuracy: ", dev_acc/20
    print "Total number of updates on the traning set: ", end_updates
    test_file = tuple(open(sys.argv[8],'r'))
    a = dev(test_weight, test_bias, test_file, 0)
    print "Test Set Acccuracy: ", a
    #plt.plot(x,y)
    #plt.show()
#####################################
    del weight[:]
    weight = [0]*69
    bias = random.uniform(-0.01, 0.01)
    for i in range(69):
        z = random.uniform(-0.01, 0.01)
        weight.append(z)
    print "\nAveraged Perceptron with Cross Validation"
    avg_acc = updates = avg_b = 0.0
    avg_w = [0] * 69
    for r in rate:
        for i in range(10):
            acc = 0.0
            weight, bias, updates, avg_w, avg_b = averaged(r, weight, bias, t_file1, updates, avg_w, avg_b)
            a = dev(avg_w, avg_b, split_file5, 0)
            acc = acc + a
            weight, bias, updates, avg_w, avg_b = averaged(r, weight, bias, t_file2, updates, avg_w, avg_b)
            a = dev(avg_w, avg_b, split_file4, 0)
            acc = acc + a
            weight, bias, updates, avg_w, avg_b = averaged(r, weight, bias, t_file3, updates, avg_w, avg_b)
            a = dev(avg_w, avg_b, split_file3, 0)
            acc = acc + a
            weight, bias, updates, avg_w, avg_b = averaged(r, weight, bias, t_file4, updates, avg_w, avg_b)
            a = dev(avg_w, avg_b, split_file2, 0)
            acc = acc + a
            weight, bias, updates, avg_w, avg_b = averaged(r, weight, bias, t_file5, updates, avg_w, avg_b)
            a = dev(avg_w, avg_b, split_file1, 0)
            acc = acc + a
            if i == 9:
                avg_acc = acc/5.0
                param[r] = avg_acc
    best = max(param, key=param.get)
    print "Best hyperparamerter: ", best
    print "Best hyperparamerter's Accuracy: ", param[best]
    param.clear()
    acc = avg_acc = dev_acc = 0.0
    print "\nAveraged Perceptron with Best Hyperparameter"
    updates = max_acc = test_bias = avg_b = end_updates = 0.0
    test_weight = avg_w = [0]*69
    x = []
    y = []
    for i in range(20):
        t_file = tuple(open(sys.argv[1],'r'))
        d_file = tuple(open(sys.argv[2],'r'))
        weight, bias, updates, avg_w, avg_b = averaged(best, weight, bias, t_file, updates, avg_w, avg_b)
        a = dev(avg_w, avg_b, d_file, 0)
        if i != 19:
            x.append(i)
            y.append(a)
        if a > max_acc:
            test_weight = weight
            test_bias = bias
            end_updates = updates
        dev_acc = dev_acc + a
    print "Devlopment Set Acccuracy: ", dev_acc/20
    print "Total number of updates on the traning set: ", end_updates
    test_file = tuple(open(sys.argv[8],'r'))
    a = dev(test_weight, test_bias, test_file, 0)
    print "Test Set Acccuracy: ", a
    #plt.plot(x,y)
    #plt.show()
#####################################
    del weight[:]
    weight = [0]*69
    bias = random.uniform(-0.01, 0.01)
    for i in range(69):
        z = random.uniform(-0.01, 0.01)
        weight.append(z)
    print "\nAggressive Perceptron with Cross Validation"
    avg_acc = updates = 0
    for r in rate:
        for m in margin:
            for i in range(10):
                acc = 0.0
                weight, updates = aggresive(r, weight, t_file1, m, updates)
                a = dev(weight, bias, split_file5, 1)
                acc = acc + a
                weight, updates = aggresive(r, weight, t_file2, m, updates)
                a = dev(weight, bias, split_file4, 1)
                acc = acc + a
                weight, updates = aggresive(r, weight, t_file3, m, updates)
                a = dev(weight, bias, split_file3, 1)
                acc = acc + a
                weight, updates = aggresive(r, weight, t_file4, m, updates)
                a = dev(weight, bias, split_file2, 1)
                acc = acc + a
                weight, updates = aggresive(r, weight, t_file5, m, updates)
                a = dev(weight, bias, split_file1, 1)
                acc = acc + a
                if i == 9:
                    avg_acc = acc/5.0
                    r = str(r)
                    m = str(m)
                    key = r + ":" + m
                    param[key] = avg_acc
    best = max(param, key=param.get)
    b = best.split(":")
    print "Best hyperparamerter: Learning Rate is ", b[0], "Margin is ", b[1]
    print "Best hyperparamerter's Accuracy: ", param[best]
    b[0] = float(b[0])
    b[1] = float(b[1])
    param.clear()
    acc = avg_acc = dev_acc = 0.0
    print "\nAggressive Perceptron with Best Hyperparameter"
    updates = max_acc = test_bias = end_updates = 0.0
    test_weight = [0]*69
    x = []
    y = []
    for i in range(20):
        t_file = tuple(open(sys.argv[1],'r'))
        d_file = tuple(open(sys.argv[2],'r'))
        weight, updates = aggresive(b[0], weight, t_file, b[1], updates)
        a = dev(weight, bias, d_file, 0)
        x.append(i)
        y.append(a)
        if a > max_acc:
            test_weight = weight
            test_bias = bias
            end_updates = updates
        dev_acc = dev_acc + a
    print "Devlopment Set Acccuracy: ", dev_acc/20
    print "Total number of updates on the traning set: ", end_updates
    test_file = tuple(open(sys.argv[8],'r'))
    a = dev(test_weight, test_bias, test_file, 1)
    print "Test Set Acccuracy: ", a
    #plt.plot(x,y)
    #plt.show()

if __name__ == "__main__":
    main()

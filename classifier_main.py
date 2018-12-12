__author__ = 'BorisMirage'
# --- coding:utf-8 ---

'''
Create by BorisMirage
File Name: classifier_main
Create Time: 2018-12-02 16:23
'''

import classifier_bayes
import classifier_svm
import numpy as np
import read_data_file


def bayes(training, training_label, test_set, test_label_set, vocabulary_list):
    c = classifier_bayes.NaiveBayesClassifier(training, training_label, test_set, test_label_set, vocabulary_list)
    bayes_out = c.performance()
    return bayes_out


def svm(training, training_label, test_set, test_label_set, vocabulary_list):
    c = classifier_svm.SVMClassifier(training, training_label, test_set, test_label_set, vocabulary_list)
    svm_out = c.performance()
    return svm_out


def stat_all(bayes_out, svm_out, r, trs, ts):
    """

    :param bayes_out: bayes model output
    :param svm_out: svm model output
    :param r: running rounds
    :param trs: test size
    :param ts: training set size
    :return: None
    """
    print("")
    print("Naive Bayes Classifier Statistics for %d rounds:" % r)
    print("Training %d messages" % trs)
    print("Test %d messages" % ts)
    print("Average model training time: %.2fs" % np.mean(bayes_out[0]))
    print("Average model test time: %.2fs" % np.mean(bayes_out[1]))
    print("Average recall rate: %.2f%%" % np.mean(bayes_out[2]))
    print("Average precision rate: %.2f%%" % np.mean(bayes_out[3]))
    print("Average error rate: %.2f%%" % np.mean(bayes_out[4]))

    print("")
    print("SVM Classifier Statistics for %d rounds:" % r)
    print("Training %d messages" % trs)
    print("Test %d messages" % ts)
    print("Average model training time: %.2fs" % np.mean(svm_out[0]))
    print("Average model test time: %.2fs" % np.mean(svm_out[1]))
    print("Average recall rate: %.2f%%" % np.mean(svm_out[2]))
    print("Average precision rate: %.2f%%" % np.mean(svm_out[3]))
    print("Average error rate: %.2f%%" % np.mean(svm_out[4]))


def single_test(msg, train_size, test_size):
    c = classifier_bayes.NaiveBayesClassifier(train_size, test_size)
    c.single_input_classification(msg)
    c1 = classifier_svm.SVMClassifier(train_size, test_size)
    c1.single_input_classification(msg)


def get_data_set(training_size, test_size):
    return read_data_file.read(training_size, test_size)


def result(tss, ts_size, tr):
    """

    :param tss: Training set size
    :param ts_size: test set size
    :param tr: running round
    :return:
    """
    # Type check
    if type(tr) is not int:
        raise TypeError("Wrong type! 'round' parameter should be integer!")
    if tr < 1:
        raise ValueError("Round # should larger than 1!")

    b_train_time, b_test_time, b_re, b_pre, b_err = np.zeros(tr), np.zeros(tr), np.zeros(tr), np.zeros(tr), np.zeros(tr)

    s_train_time, s_test_time, s_re, s_pre, s_err = np.zeros(tr), np.zeros(tr), np.zeros(tr), np.zeros(tr), np.zeros(tr)

    b = []
    s = []

    for i in range(0, tr):
        training, training_label, test_set, test_label_set, vocabulary_list = get_data_set(tss, ts_size)

        print("Bayes Round %d:" % (i + 1))
        br = bayes(training, training_label, test_set, test_label_set, vocabulary_list)
        b_train_time[i], b_test_time[i], b_re[i], b_pre[i], b_err[i] = br[0], br[1], br[2], br[3], br[4]

        print("SVM Round %d:" % (i + 1))
        sr = svm(training, training_label, test_set, test_label_set, vocabulary_list)
        s_train_time[i], s_test_time[i], s_re[i], s_pre[i], s_err[i] = sr[0], sr[1], sr[2], sr[3], sr[4]

    print("----------")
    print(b_train_time)
    print(b_test_time)
    print(b_re)
    print(b_pre)
    print(b_err)

    print("----------")
    print(s_train_time)
    print(s_test_time)
    print(s_re)
    print(s_pre)
    print(s_err)
    print("----------")

    b.append(b_train_time)
    b.append(b_test_time)
    b.append(b_re)
    b.append(b_pre)
    b.append(b_err)

    s.append(s_train_time)
    s.append(s_test_time)
    s.append(s_re)
    s.append(s_pre)
    s.append(s_err)
    stat_all(b, s, tr, tss, ts_size)

    return


if __name__ == '__main__':
    """
    Unit test
    """

    # t = read_data_file.read(1000, 500)
    # # print(t[1])
    # # print(t[3])
    # spam = 0
    # sp = 0
    #
    # for i in t[1]:
    #     if i == 1:
    #         sp += 1
    #
    # for i in t[3]:
    #     if i == 1:
    #         spam += 1
    #
    # print(sp)
    # print(spam)

    training_set_size = 1000
    test_set_size = 500
    test_round = 20
    result(training_set_size, test_set_size, test_round)

    # msg = "Our biggest sale of the year is coming to a close, don't miss your chance to save up to 50% on your favorites including Epionce, SkinMedica, Jurlique, and many more!"
    # single_test(msg, 400, 1)

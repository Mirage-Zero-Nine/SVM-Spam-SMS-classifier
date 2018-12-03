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


def bayes(train_size, test_size, run_round):
    c = classifier_bayes.NaiveBayesClassifier(train_size, test_size)
    bayes_out = c.performance(run_round)
    return bayes_out


def svm(train_size, test_size, run_round):
    c = classifier_svm.SVMClassifier(train_size, test_size)
    svm_out = c.performance(run_round)
    return svm_out


def statistics(bayes_out, svm_out):
    print("")
    print("Naive Bayes Classifier Statistics for %d rounds:" % len(bayes_out[0]))
    print("Average time: %.2fs" % np.mean(bayes_out[0]))
    print("Average recall rate: %.2f%%" % np.mean(bayes_out[1]))
    print("Average precision rate: %.2f%%" % np.mean(bayes_out[2]))
    print("Average error rate: %.2f%%" % np.mean(bayes_out[3]))

    print("")
    print("SVM Classifier Statistics for %d rounds:" % len(svm_out[0]))
    print("Average time: %.2fs" % np.mean(svm_out[0]))
    print("Average recall rate: %.2f%%" % np.mean(svm_out[1]))
    print("Average precision rate: %.2f%%" % np.mean(svm_out[2]))
    print("Average error rate: %.2f%%" % np.mean(svm_out[3]))


def single_test(msg, train_size, test_size):
    c = classifier_bayes.NaiveBayesClassifier(train_size, test_size)
    c.single_input_classification(msg)
    c1 = classifier_svm.SVMClassifier(train_size, test_size)
    c1.single_input_classification(msg)


if __name__ == '__main__':
    """
    Unit test
    """
    statistics(bayes(50, 100, 5), svm(50, 100, 5))
    # msg = "Our biggest sale of the year is coming to a close, don't miss your chance to save up to 50% on your favorites including Epionce, SkinMedica, Jurlique, and many more!"
    # single_test(msg, 400, 1)

__author__ = 'BorisMirage'
# --- coding:utf-8 ---

'''
Create by BorisMirage
File Name: classifier_svm
Create Time: 2018-12-02 12:29
'''

import numpy as np
import time
import re
from sklearn import svm
import plot


class SVMClassifier(object):

    def __init__(self, training_data_set, training_data_label, test_set, test_label_set, vocabulary_list, plot=False):

        """
        Class initialization.
        :param training_data_set: training data set (message set)
        :param training_data_label: each training message's label in training data set
        :param test_set: data set that contains test messages
        :param test_label_set: each test message's label in test data set
        :param vocabulary_list: vocabulary list for training data set
        """

        self.training_set = training_data_set
        self.training_label = training_data_label
        self.test_set = test_set
        self.test_label = test_label_set
        self.vocabulary_list = vocabulary_list
        self.plot = plot

    @staticmethod
    def __create_vocabulary_list(data):
        """
        Create a vocabulary set based on given data.
        :param data: data
        :return: list contains all words
        """
        out_set = set([])
        for document in data:
            out_set = out_set | set(document)
        return list(out_set)

    @staticmethod
    def __convert_to_text_list(string):
        """
        Create a word list based on given string.
        :param string: given string
        :return: word list
        """
        words_list = re.split(r'\W+', string)
        remove = ['the', 'are', 'this', 'that', 'and', 'with', 'for']
        res = []
        for w in words_list:
            if len(w) > 2 and w.lower() not in remove:
                res.append(w)
        return res

    @staticmethod
    def __data_to_vector(vocab_list, input_set):
        out = [0] * len(vocab_list)

        for word in input_set:
            if word in vocab_list:
                # Calculate word frequency
                out[vocab_list.index(word)] += 1
        return out

    def __read_data(self):
        """
        Generate both training set and test set for Naive Bayes classifier model.
        This funcrion was wrapped in read_data_file.py for identical data set input to both classifier.
        :return: training set, corresponding training messages label, test set, corresponding label, vocabulary
        """

        return self.training_set, self.training_label, self.test_set, self.test_label, self.vocabulary_list

    def __svm_training(self, training_messages, training_label, vocabulary_list):
        """
        SVM model training.
        Note that if the penalty parameter C in SVM function is too large, the result model may be over fit.
        :param training_messages:
        :param training_label:
        :param vocabulary_list:
        :return:
        """

        print('Start training SVM model. ')
        start = time.clock()

        data = []
        label = []
        for i in range(0, len(training_messages)):
            data.append(self.__data_to_vector(vocabulary_list, training_messages[i]))
            label.append(training_label[i])
        data = np.asarray(data)
        label = np.asarray(label)

        """
        C is penalty. Output model model may be over fit if C is too large
        kernel function: gaussian (rbf)
        """
        # kernel function: gaussian
        out = svm.SVC(C=10, kernel='linear')

        model = out.fit(data, label)

        end = time.clock()
        total_time = end - start

        print("Total training time: %.2fs" % total_time)
        print('SVM model completed. ')

        if self.plot:
            plot.svm_plot(data, label)
        return model

    def __check_accuracy(self, svm_model, test_message, test_label, vocabulary_list):
        """

        :param svm_model:
        :param test_message:
        :param test_label:
        :param vocabulary_list:
        :return:
        """

        correct, error, spam_but_ham, ham_but_spam = 0, 0, 0, 0

        # Test accuracy
        for i in range(0, len(test_message)):
            v = self.__data_to_vector(vocabulary_list, test_message[i])
            res = int(svm_model.predict(np.array(v).reshape(1, -1))[0])
            if res == test_label[i]:
                correct += 1
            elif res == 1 and test_label[i] == 0:

                print("Classify ham message to spam message")

                error += 1
                ham_but_spam += 1
            elif res == 0 and test_label[i] == 1:

                print("Classify spam message to ham message")

                error += 1
                spam_but_ham += 1

        recall = (correct / (correct + spam_but_ham)) * 100
        precision = (correct / (correct + ham_but_spam)) * 100
        err = (error / len(test_message)) * 100

        print('')
        print('Recall Rate: %.2f%%' % recall)
        print('Precision: %.2f%%' % precision)
        print('Error Rate: %.2f%%' % err)
        print('')
        return recall, precision, err

    def performance(self):
        """
        Running whole classifier.
        :return: time, recall rate, precision, error rate
        """

        # Read data
        training, training_label, test_set, test_label_set, vocabulary_list = self.__read_data()

        # Timer start
        start = time.clock()
        svm_model = self.__svm_training(training, training_label, vocabulary_list)
        res = self.__check_accuracy(svm_model, test_set, test_label_set, vocabulary_list)

        # Timer end
        end = time.clock()
        t = end - start
        print("SVM Classifier Round Time: %.2f" % t)

        # Return time, recall rate, precision, error rate
        return t, res[0], res[1], res[2]

    def single_input_classification(self, msg):
        """
        Give a input message and test what class will this message be classified to.
        :param msg: given message
        :return: None
        """

        msg = self.__convert_to_text_list(msg)

        training = self.__read_data()

        model = self.__svm_training(training[0], training[1], training[4])

        v = self.__data_to_vector(training[4], msg)

        res = int(model.predict(np.array(v).reshape(1, -1))[0])
        if res == 1:
            print("This message is classified to spam message!")
        elif res == 0:
            print("This message is classified to ham message!")
        else:
            print("Wrong classification!")
        return


if __name__ == '__main__':
    """
    Unit test
    """
    # demo(1)
    # c = SVMClassifier(1000, 10)
    # c.single_input_classification(
    #     "Hi I'm sue. I am 20 years old and work as a lapdancer. I love sex. Text me live - I'm i my bedroom now. text SUE to 89555. By TextOperator G2 1DA 150ppmsg 18+")
    pass

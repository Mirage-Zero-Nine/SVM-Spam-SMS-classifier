__author__ = 'BorisMirage'
# --- coding:utf-8 ---

'''
Create by BorisMirage
File Name: classifier_bayes
Create Time: 2018-12-01 15:46
'''

import numpy as np
import time
import re


class NaiveBayesClassifier(object):
    def __init__(self, training_data_set, training_data_label, test_set, test_label_set, vocabulary_list):

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

    @staticmethod
    def __create_vocabulary_list(data):
        """
        Create a vocabulary set based on given data.
        :param data: data
        :return: list contains all words
        """
        vocal_list = set([])
        for document in data:
            vocal_list = vocal_list | set(document)
        return list(vocal_list)

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
    def __data_to_vector(vocab_list, input_list):
        """

        :param vocab_list:
        :param input_list:
        :return:
        """
        out = [0] * len(vocab_list)
        for word in input_list:
            if word in vocab_list:
                # Calculate word frequency
                out[vocab_list.index(word)] += 1
        return out

    def __training_bayes_model(self, train_msg, train_label, vocabulary_list):
        """
        Generate word frequency vector for each message and train Naive Bayes classifier.
        :param train_msg:
        :param train_label:
        :param vocabulary_list:
        :return: p0_vector, p1_vector, p_spam
        """
        print('Start training Naive Bayes model. ')
        print('Size of vocabulary list: %d' % len(vocabulary_list))
        start = time.clock()

        # Create message vector for each training message and append each message's label
        training_arr = []

        for i in range(0, len(train_msg)):
            training_arr.append(self.__data_to_vector(vocabulary_list, train_msg[i]))
            # print("%d th messages loaded..." % (i + 1))

        data = np.array(training_arr)
        label = np.array(train_label)

        n_words = len(data[0])
        spam_in_total_msg = np.sum(label) / len(data)

        # Laplace smoothing
        p0_num = np.ones(n_words)
        p1_num = np.ones(n_words)

        p0_denominator = 2.0
        p1_denominator = 2.0

        # Calculate probability and total appearance, label 0 for ham message, 1 for spam message
        for i in range(len(data)):
            if label[i] == 1:
                p1_num += data[i]
                p1_denominator += np.sum(data[i])
            else:
                p0_num += data[i]
                p0_denominator += np.sum(data[i])

        # use log() to obtain better accuracy
        p1_vector = np.log(p1_num / p1_denominator)
        p0_vector = np.log(p0_num / p0_denominator)

        end = time.clock()
        total_time = end - start

        print('Naive Bayes model created. ')
        print("Total training time: %.2fs " % total_time)

        return p0_vector, p1_vector, spam_in_total_msg

    @staticmethod
    def __naive_bayes_classification(test_vec, p0_vec, p1_vec, p_class1):
        """
        Classify input vector to spam message or ham message.
        Multiply message vector with spam message probability and ham message probability.
        Return larger probability.
        :param test_vec: message vector
        :param p0_vec:
        :param p1_vec:
        :param p_class1:
        :return: 0 for ham message, 1 for spam message
        """
        p1 = np.sum(test_vec * p1_vec) + np.log(p_class1)
        p0 = np.sum(test_vec * p0_vec) + np.log(1.0 - p_class1)
        if p1 > p0:
            return 1  # spam message
        else:
            return 0  # ham message

    def __read_data(self):
        """
        Generate both training set and test set for Naive Bayes classifier model.
        This funcrion was wrapped in read_data_file.py for identical data set input to both classifier.
        :return: training set, corresponding training messages label, test set, corresponding label, vocabulary
        """

        return self.training_set, self.training_label, self.test_set, self.test_label, self.vocabulary_list

    def __check_accuracy(self, test_message, test_label, vocabulary_list, p0_vector, p1_vector, p_spam):
        """
        Test model accuracy based on test set that is created before.
        :param test_message:
        :param test_label:
        :param vocabulary_list:
        :param p0_vector:
        :param p1_vector:
        :param p_spam:
        :return:
        """
        correct = 0
        error = 0
        spam_but_ham = 0
        ham_but_spam = 0

        # Test accuracy
        for i in range(0, len(test_message)):
            word_vector = self.__data_to_vector(vocabulary_list, test_message[i])

            # Return type is int
            res = self.__naive_bayes_classification(np.array(word_vector), p0_vector, p1_vector, p_spam)

            # Correct result
            if res == test_label[i]:
                correct += 1

            # Classify spam message to ham message
            elif test_label[i] == 1 and res == 0:
                error += 1
                spam_but_ham += 1

                print("Classify spam message to ham message!")

            # Classify ham message to spam message
            elif test_label[i] == 0 and res == 1:
                error += 1
                ham_but_spam += 1
                print("Classify ham message to spam message!")

        recall = (correct / (correct + spam_but_ham)) * 100
        precision = (correct / (correct + ham_but_spam)) * 100
        err = (error / len(test_message)) * 100

        print('')
        print('Recall Rate: %.2f%%' % recall)
        print('Precision: %.2f%%' % precision)
        print('Error Rate: %.2f%%' % err)
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

        # Training Naive Bayes model
        p0_vector, p1_vector, p_spam = self.__training_bayes_model(training, training_label, vocabulary_list)

        # Test accuracy
        res = self.__check_accuracy(test_set, test_label_set, vocabulary_list, p0_vector, p1_vector, p_spam)

        # Timer end
        end = time.clock()
        t = end - start
        print("Naive Bayes Classifier Round Time: %.2fs" % t)
        print('')

        # Return time, recall rate, precision, error rate
        return t, res[0], res[1], res[2]

    def single_input_classification(self, msg):
        """
        Give a input message and test what class will this message be classified to.
        :param msg: given message
        :return: None
        """

        msg = self.__convert_to_text_list(msg)

        training, training_label, test_set, test_label_set, vocabulary_list = self.__read_data()

        p0_vector, p1_vector, p_spam = self.__training_bayes_model(training, training_label, vocabulary_list)

        v = self.__data_to_vector(vocabulary_list, msg)
        res = self.__naive_bayes_classification(np.array(v), p0_vector, p1_vector, p_spam)
        if res == 1:
            print("This message is classified to spam message!")
        elif res == 0:
            print("This message is classified to ham message!")
        else:
            print("Wrong classification!")
        return


if __name__ == '__main__':
    # c = NaiveBayesClassifier(20, 10)
    # c.performance(1)
    # msg = "Hi I'm sue. I am 20 years old and work as a lapdancer. I love sex. Text me live - I'm i my bedroom now. text SUE to 89555. By TextOperator G2 1DA 150ppmsg 18+"
    # NaiveBayesClassifier(1000, 1).single_input_classification(msg)
    # single_test(msg, 200, 10)
    pass

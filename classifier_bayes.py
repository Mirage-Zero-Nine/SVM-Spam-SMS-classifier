__author__ = 'BorisMirage'
# --- coding:utf-8 ---

'''
Create by BorisMirage
File Name: classifier_bayes
Create Time: 2018-12-01 15:46
'''

import random
import numpy as np
import time
import re


class NaiveBayesClassifier(object):
    def __init__(self, training_data_size=200, test_data_size=50):
        """
        Class initialization.
        :param training_data_size: training message number, default value is 200 (2 < size < 1540)
        :param test_data_size: test message number, default value is 50
        """

        if (isinstance(training_data_size, int) is False) or (isinstance(test_data_size, int) is False):
            raise TypeError("Size should be integer!")

        if training_data_size < 3:
            raise ValueError("Training data size should larger than 2!")
        elif training_data_size > 1540:
            raise ValueError("Training data size too large, insufficient messages amount.")
        elif test_data_size < 1:
            raise ValueError("Test message number should larger than 1!")
        if training_data_size + test_data_size > 5996:
            raise ValueError("Larger than all messages!")

        self.train_size = training_data_size
        self.test_size = test_data_size

    def __read_data(self):
        """
        Generate both training set and test set for Naive Bayes classifier model.
        :return: training set, corresponding training messages label, test set, corresponding label, vocabulary
        """
        # Check if value is legal

        msg_list = []
        msg_label_list = []

        print("Start loading messages.")

        # Read all spam messages (771 in total)
        for i in range(1, 455):
            spam = self.__convert_to_text_list(open('spam1/%d.txt' % i).read())
            msg_list.append(spam)
            msg_label_list.append(1)

        print("All spam messages loaded. ")

        # Read all ham messages (4825 in total)
        for i in range(1, 2412):
            ham = self.__convert_to_text_list(open('ham1/%d.txt' % i).read())
            msg_list.append(ham)
            msg_label_list.append(0)

        print("All ham messages loaded. ")

        vocabulary_list = self.__create_vocabulary_list(msg_list)

        print("Vocabulary list created. ")

        training_data_set = []
        training_data_label = []
        test_set = []
        test_label_set = []

        print("Create training spam set. ")

        # Create training set (spam)
        while len(training_data_set) < int(self.train_size / 2):
            index = random.randint(0, len(msg_list) - 1)
            if msg_label_list[index] == 1:
                training_data_set.append(msg_list[index])
                training_data_label.append(msg_label_list[index])

                # Avoid duplication
                del msg_list[index]
                del msg_label_list[index]

        print("Training spam set created. ")
        print("Create training ham set. ")

        # Create training set (ham)
        while len(training_data_set) < self.train_size:
            index = random.randint(0, len(msg_list) - 1)
            if msg_label_list[index] == 0:
                training_data_set.append(msg_list[index])
                training_data_label.append(msg_label_list[index])

                # Avoid duplication
                del msg_list[index]
                del msg_label_list[index]

        print("Training ham set created. ")
        print("Create test set. ")

        # Create test set
        for i in range(0, self.test_size):
            index = random.randint(0, len(msg_list) - 1)
            test_set.append(msg_list[index])
            test_label_set.append(msg_label_list[index])

            # Avoid duplication
            del msg_list[index]
            del msg_label_list[index]

        print("Data set created. ")

        return training_data_set, training_data_label, test_set, test_label_set, vocabulary_list

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

    def __training_bayes_model(self, train_msg, train_label, vocabulary_list):
        """
        Generate word frequency vector for each message and train Naive Bayes classifier.
        :param train_msg:
        :param train_label:
        :param vocabulary_list:
        :return: p0_vector, p1_vector, p_spam
        """
        print('Start training Bayes model. ')
        start = time.clock()

        # Create message vector for each training message and append each message's label
        training_arr = []

        print('Start converting message to vector. ')

        for i in range(0, len(train_msg)):
            training_arr.append(self.__data_to_vector(vocabulary_list, train_msg[i]))

        print('Converted to vector. ')

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

        print('Naive Bayes model completed. ')
        print("Total training time: %.2fs" % total_time)

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

    @staticmethod
    def __convert_to_text_list(string):
        """
        Create a word list based on given string.
        :param string: given string
        :return: word list
        """
        list_of_tokens = re.split(r'\W+', string)
        return [tok.lower() for tok in list_of_tokens if len(tok) > 2]

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

    def __check_accuracy(self, test_message, test_label, vocabulary_list, p0_vector, p1_vector, p_spam):
        """

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

                print("Classify spam message to ham message")

            # Classify ham message to spam message
            elif test_label[i] == 0 and res == 1:
                error += 1
                ham_but_spam += 1
                print("Classify ham message to spam message")

        recall = (correct / (correct + spam_but_ham)) * 100
        precision = (correct / (correct + ham_but_spam)) * 100
        err = (error / len(test_message)) * 100

        print('')
        print('Recall Rate: %.2f%%' % recall)
        print('Precision: %.2f%%' % precision)
        print('Error Rate: %.2f%%' % err)
        print('')
        return recall, precision, err

    def performance(self, test_round):
        """
        Running whole classifier.
        :param test_round: running round for testing.
        :return: time, recall rate, precision rate, error rate for each round respectively, wrapped in a list
        """

        # Type check
        if type(test_round) is not int:
            raise TypeError("Wrong type! 'round' parameter should be integer!")
        if test_round < 1:
            raise ValueError("Round # should larger than 1!")

        time_list, recall_list, precision_list, error_list = np.zeros(test_round, dtype=float), \
                                                             np.zeros(test_round, dtype=float), \
                                                             np.zeros(test_round, dtype=float), \
                                                             np.zeros(test_round, dtype=float)

        # Test
        for i in range(0, test_round):
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

            # Add time
            time_list[i] = t

            # Add recall rate
            recall_list[i] = res[0]

            # Add precision
            precision_list[i] = res[1]

            # Add error rate
            error_list[i] = res[2]

        return time_list, recall_list, precision_list, error_list

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
    msg = "Hi I'm sue. I am 20 years old and work as a lapdancer. I love sex. Text me live - I'm i my bedroom now. text SUE to 89555. By TextOperator G2 1DA 150ppmsg 18+"
    NaiveBayesClassifier(1000, 1).single_input_classification(msg)
    # single_test(msg, 200, 10)

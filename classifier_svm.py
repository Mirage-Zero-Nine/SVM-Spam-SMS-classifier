__author__ = 'BorisMirage'
# --- coding:utf-8 ---

'''
Create by BorisMirage
File Name: classifier_svm
Create Time: 2018-12-02 12:29
'''

import random
import numpy as np
import time
import re
from sklearn import svm
import plot


class SVMClassifier(object):
    def __init__(self, training_data_size, test_data_size, do_plot=False):
        """
        Class initialization.
        :param training_data_size: training message number, default value is 200 (2 < size < 1540)
        :param test_data_size: test message number, default value is 50
        :param do_plot: whether plot data via t-SNE, default is False
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
        self.plot = do_plot

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
        list_of_tokens = re.split(r'\W+', string)
        return [tok.lower() for tok in list_of_tokens if len(tok) > 2]

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
        :return: training_data_set, training_data_label, test_set, test_label_set, vocabulary_list
        """

        # Check if value is legal
        if self.train_size + self.test_size > 5996:
            raise ValueError("Larger than all messages!")

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

    def performance(self, test_round):
        """
        Running whole classifier.
        :param test_round: running round for testing.
        :return: time, recall rate, precision rate, error rate for each round respectively, wrapped in a list
        """

        # Type check
        if type(test_round) is not int:
            raise TypeError("Wrong type! Given 'round' parameter should be integer!")
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
            svm_model = self.__svm_training(training, training_label, vocabulary_list)
            res = self.__check_accuracy(svm_model, test_set, test_label_set, vocabulary_list)

            # Timer end
            end = time.clock()
            t = end - start
            print("SVM Classifier Round Time: %.2f" % t)

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
    c = SVMClassifier(1000, 10)
    c.single_input_classification(
        "Hi I'm sue. I am 20 years old and work as a lapdancer. I love sex. Text me live - I'm i my bedroom now. text SUE to 89555. By TextOperator G2 1DA 150ppmsg 18+")

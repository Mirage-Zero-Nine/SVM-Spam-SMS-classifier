__author__ = 'BorisMirage'
# --- coding:utf-8 ---

'''
Create by BorisMirage
File Name: read_data
Create Time: 2018-12-03 18:47
'''

import random
import re


def __convert_to_text_list(string):
    """
    Create a word list based on given string.
    :param string: given string
    :return: word list
    """
    words_list = re.split(r'\W+', string)
    # remove = ['the', 'are', 'this', 'that', 'and', 'with', 'for']
    remove = []
    res = []
    for w in words_list:
        if len(w) > 2 and w.lower() not in remove:
            res.append(w)

    return res


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


def read(train_size, test_size):
    """
    Generate both training set and test set for Naive Bayes classifier model.
    :return: training set, corresponding training messages label, test set, corresponding label, vocabulary
    """
    # Check if value is legal
    if (isinstance(train_size, int) is False) or (isinstance(test_size, int) is False):
        raise TypeError("Size should be integer!")

    if train_size < 3:
        raise ValueError("Training data size should larger than 2!")
    elif train_size > 1540:
        raise ValueError("Training data size too large, insufficient messages amount.")
    elif test_size < 1:
        raise ValueError("Test message number should larger than 1!")
    if train_size + test_size > 5996:
        raise ValueError("Larger than all messages!")
    msg_list = []
    msg_label_list = []

    print("Start loading messages.")

    # Read all spam messages (771 in total)
    for i in range(1, 455):
        # for i in range(1, 771):

        spam = __convert_to_text_list(open('spam1/%d.txt' % i).read())
        msg_list.append(spam)
        msg_label_list.append(1)

    # Read all ham messages (4825 in total)
    for i in range(1, 2413):
        # for i in range(1, 4825):
        ham = __convert_to_text_list(open('ham1/%d.txt' % i).read())
        msg_list.append(ham)
        msg_label_list.append(0)

    print("All messages loaded. ")

    training_data_set = []
    training_data_label = []
    test_set = []
    test_label_set = []

    print("Create training set and test set. ")

    # Create training set (spam)
    while len(training_data_set) < int(train_size / 4):
        index = random.randint(0, len(msg_list) - 1)
        if msg_label_list[index] == 1:
            training_data_set.append(msg_list[index])
            training_data_label.append(msg_label_list[index])

            # Avoid duplication
            del msg_list[index]
            del msg_label_list[index]

    # Create training set (ham)
    while len(training_data_set) < train_size:
        index = random.randint(0, len(msg_list) - 1)
        if msg_label_list[index] == 0:
            training_data_set.append(msg_list[index])
            training_data_label.append(msg_label_list[index])

            # Avoid duplication
            del msg_list[index]
            del msg_label_list[index]

    # Count spam message and ham message in test set
    s = 0
    h = 0

    # Create test set
    for i in range(0, test_size):
        index = random.randint(0, len(msg_list) - 1)
        if msg_label_list[index] == 1:
            s += 1
        else:
            h += 1
        test_set.append(msg_list[index])
        test_label_set.append(msg_label_list[index])

        # Avoid duplication
        del msg_list[index]
        del msg_label_list[index]

    print("Training set and test set created. ")
    print("Spam message in training set: %d " % s)
    print("Ham message in training set: %d " % h)

    print("Creating vocabulary list of training data set. ")
    vocabulary_list = __create_vocabulary_list(training_data_set)
    print("Vocabulary list created. ")

    return training_data_set, training_data_label, test_set, test_label_set, vocabulary_list


if __name__ == '__main__':
    pass

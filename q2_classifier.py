#!/usr/bin/env python
import sys
from collections import Counter

__author__ = 'Shreyas Bhatia'

from optparse import OptionParser
import pandas as pd
import numpy as np


class NaiveBayes:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.ham_word_dict = Counter()
        self.spam_word_dict = Counter()
        self.pSpam = 1.0
        self.pNotSpam = 1.0
        self.train()

    def train(self):
        # read training file
        train_file = open(self.train_data, "r")
        total = 0
        spam = 0
        # process each line
        for line in train_file:
            data = line.split(' ')
            # spam or ham
            res = data[1]
            # get email id
            email = data[1]
            for items in range(2, len(data), 2):
                total += 1
                if res == "spam":
                    spam += 1
                    self.spam_word_dict[data[items]] += int(data[items + 1])
                else:
                    self.ham_word_dict[data[items]] += int(data[items + 1])
        self.pSpam *= float(spam) / total
        self.pNotSpam = 1 - self.pSpam

    def conditional_word_prob(self, word, count, spam):
        if spam:
            if word in self.spam_word_dict:
                return float(self.spam_word_dict[word] / float(sum(self.spam_word_dict.itervalues()))) ** count
            else:
                return float(self.ham_word_dict[word] / float(sum((self.ham_word_dict.itervalues())))) ** count

    def conditional_prob(self, email, is_spam):
        result = 1.0

        for i in range(0,len(email), 2):
            item = email[i]
            count = int(email[i + 1])

            result = result * self.conditional_word_prob(item, count, is_spam)
        return result

    def classify(self, email):
        isSpam = self.pSpam * self.conditional_prob(email, True)
        notSpam = self.pNotSpam * self.conditional_prob(email, False)
        return isSpam > notSpam


if __name__ == "__main__":

    new_argv = []
    for arg in sys.argv:
        if arg.startswith('-') and len(arg) > 2:
            arg = '-' + arg
        new_argv.append(arg)
    sys.argv = new_argv

    optparser = OptionParser()
    optparser.add_option('-t', '--f1', dest="traindata", help="train data csv")
    optparser.add_option('-q', '--f2', dest="testdata", help="test data csv")
    optparser.add_option('-o', '--output', dest="output", help="output file")

    (options, args) = optparser.parse_args()

    train_data = options.traindata
    test_data = options.testdata
    output = options.output

    bayes_classifier = NaiveBayes(train_data, test_data)

    testfile = open(test_data, 'r')
    correct = 0
    wrong = 0
    for items in testfile:
        items = items.split(' ')
        email_id = items[0]
        res = items[1]
        body = items[2:]

        prediction = bayes_classifier.classify(body)
        test = ""
        if prediction == False:
            test = "ham"
        else:
            test = "spam"

        if test == res:
            correct+=1
        else:
            wrong +=1

    print correct, wrong
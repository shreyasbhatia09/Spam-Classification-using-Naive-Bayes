#!/usr/bin/env python
import sys
import csv
from collections import Counter

__author__ = 'Shreyas Bhatia, Prakhar Avasthi'

from optparse import OptionParser
import numpy as np

# Class for Naive Bayes
class NaiveBayes:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.ham_word_dict = Counter()
        self.spam_word_dict = Counter()
        self.pSpam = 1.0
        self.pNotSpam = 1.0
        self.total_spam_words = 0.0
        self.total_ham_words = 0.0
        self.alpha = 0.1
        self.train()

    # Training function
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

            # Add the word to spam or ham dictionary
            for items in range(2, len(data), 2):
                total += 1
                if res == "spam":
                    spam += 1
                    self.spam_word_dict[data[items]] += int(data[items + 1])
                else:
                    self.ham_word_dict[data[items]] += int(data[items + 1])

        self.total_spam_words += float(sum(self.spam_word_dict.itervalues()))
        self.total_ham_words += float(sum(self.ham_word_dict.itervalues()))
        self.pSpam = float(float(spam) / float(total))
        self.pNotSpam = 1 - self.pSpam

    # Calculating P(word|Spam)
    def conditional_word_prob(self, word, count, spam):
        # taking logarithms values so that too small probabilities can't be neglected due to overflow
        # P(W="abc"/S="Spam")^count is written as count*log(P(W="abc"/S="Spam"))
        if spam:
            if word in self.spam_word_dict:
                return np.log10(float(self.spam_word_dict[word]) / float(self.total_spam_words)) * count
            else:
                Ns = len(self.spam_word_dict)
                return np.log10(self.alpha / float(
                    self.total_spam_words + (Ns * self.alpha)))  # laplace blending, when complete new word is found
        else:
            if word in self.ham_word_dict:
                return np.log10(float(self.ham_word_dict[word]) / float(self.total_ham_words)) * count
            else:
                Nh = len(self.ham_word_dict)
                return np.log10(self.alpha / float(
                    self.total_ham_words + (Nh * self.alpha)))  # laplace blending, when complete new word is found

    # Calculating P(Spam, word1, word2,....wordn)
    def conditional_prob(self, email, is_spam):
        result = 0.0

        for i in range(0, len(email), 2):
            item = email[i]
            count = int(email[i + 1])

            # Since conditional prob are in log, so we should add rather than multiply
            cond_prob = self.conditional_word_prob(item, count, is_spam)
            if cond_prob is not None:
                result += cond_prob
        return result

    # Classify ham or spam
    def classify(self, email):
        # Due to probabilities in logarithms, prob. of spam or ham is also taken in logarithms form
        isSpam = np.log10(self.pSpam) + self.conditional_prob(email, True)
        notSpam = np.log10(self.pNotSpam) + self.conditional_prob(email, False)
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

    # Create bayes net classifier
    bayes_classifier = NaiveBayes(train_data, test_data)

    testfile = open(test_data, 'r')
    correct = 0
    wrong = 0
    # let us test!

    # True positive, True Negative, False Positive, False Negative
    test_result = list()
    actual_spam_count = 0
    actual_ham_count = 0
    correct_pred_spam_count = 0
    correct_pred_ham_count = 0
    total_pred_spam_count = 0
    total_pred_ham_count = 0

    # Read the testfile for prediction
    for items in testfile:
        items = items.split(' ')
        email_id = items[0]
        actual_res = items[1]
        body = items[2:]

        prediction = bayes_classifier.classify(body)
        pred_result = ""

        # count true positive, true negative, false positive, false negative for Accuracy, Recall and F1 score
        if actual_res == "ham":
            actual_ham_count += 1
        else:
            actual_spam_count += 1

        if prediction == False:
            pred_result = "ham"
            total_pred_ham_count += 1
        else:
            pred_result = "spam"
            total_pred_spam_count += 1

        test_result.append((email_id, pred_result))
        if pred_result == actual_res:
            if pred_result == "spam":
                correct_pred_spam_count += 1
            else:
                correct_pred_ham_count += 1
            correct += 1
        else:
            wrong += 1

    # Write the output to a csv
    with open(output + ".csv", 'wb') as output_file:
        wr = csv.writer(output_file, quoting=csv.QUOTE_ALL)
        for result in test_result:
            wr.writerow(result)

    # Print the accuracy, f1 and recall
    print "Accuracy: ", float(correct) / (correct + wrong) * 100, "%"
    print "Precision: ", float(correct_pred_spam_count) / total_pred_spam_count * 100, "%"
    print "Recall: ", float(correct_pred_spam_count) / actual_spam_count * 100, "%"

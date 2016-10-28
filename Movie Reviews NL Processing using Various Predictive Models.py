import nltk

# 2,000 movie reviews in total: 1,000 positive, 1,000 negative
#
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI # inherit NLTK's classifier class
from nltk.tokenize import word_tokenize

from statistics import mode # for choosing which classifier got the most votes
import random
import pickle

import numpy
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

pickled = False
FEAT_CNT = 5000 # try 4000 or 3000
TRAIN_SIZE = 10000

# datapath for saving via pickle out training data
#
data_path = "C:\\ML_Data\\...our main data path"
pickle_path2 = data_path + "movie_reviews_pickle2"

print("data_path: ", data_path, "\npickle_path2: ", pickle_path2)


# define our own classifier that votes on multiple
# classifiers
#
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = [] 
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return(mode(votes)) # which classifier got the most votes
    
    def confidence(self, features):
        votes = [] 
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            
# count how many occurances of the most popular vote
# were in that list
#
        choice_votes = votes.count(mode(votes)) 
        conf = choice_votes / len(votes)
        return conf
    
# read in our pos and neg moview review records (around 10k recs each)
#
short_pos_revs = open(data_path + "positive.txt", "r").read()
short_neg_revs = open(data_path + "negative.txt", "r").read() 
    
docs = []

for rev in short_pos_revs.split('\n'):
    docs.append( (rev, "pos") )

for rev in short_neg_revs.split('\n'):
    docs.append( (rev, "neg") )
    
all_words = []

# tokenize individual words from both review types
#
short_pos_words = word_tokenize(short_pos_revs)
short_neg_words = word_tokenize(short_neg_revs)
    
for w in short_pos_words:
    all_words.append(w.lower())
    
for w in short_neg_words:
    all_words.append(w.lower())  

# we're done with populating all_words


# next, convert our word list to an  NLTK frequency distribution
#
all_words = nltk.FreqDist(all_words)

#print("Most common: ", all_words.most_common(15))

# we want to have some sort of limit on the # of words
# FreqDist is for the most common to least common
# words, to filter common words such as "s" "the" "in"
# (then we'll train against these top words)
#
word_features = list(all_words.keys())[:FEAT_CNT]

def find_features(document):
    words = word_tokenize(document)   # document is one long str, so tokenize it
    features = {}
    
    for w in word_features:
        features[w] = (w in words)
        
    return features


# print top words and their category: True or False
#
# each doc witll be a review of the words, then convert 
# words to features
#
feature_set = [(find_features(rev), category) for(rev, category) in docs ]

random.shuffle(feature_set)

# for our analysis, we're going to have a training set
# and a testing set
#
# positive training set:
#
training_set = feature_set[:TRAIN_SIZE]
testing_set  = feature_set[TRAIN_SIZE:]

# negative training set:
#
#training_set = feature_set[100:]
#testing_set  = feature_set[:100]

# with testing_set we don't tell the machine what the category
# is to see how accurate the predictor is: instead, we'll use a 
# Naive Bayes Classifier, not necessarily the best, but scalable
#
# Naive Bayes Classifier are computationally simple, so it can
# be scaled to very large data sets
# The NB algo is run twice: once for pos and once for neg
#
# posterior (likelihood) = (prior occurances * likelihood) / current evidence
#

# first time through before pickling, pickle our classifier data
#
# open our previously saved pickle file if it exists
# otherwise save the pickle file for the first time
#
if (pickled == True):
    classifier_saved_file = open(pickle_path2, "rb")
    classifier = pickle.load(classifier_saved_file) 
    classifier_saved_file.close()
else:
    classifier = nltk.NaiveBayesClassifier.train(training_set)

# one-time pickle save of the classifier data
#
if (pickled == False):
    save_classifier = open(pickle_path2, "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


# show the most popular words on both sides
# and whether they tend to be positive or negative
#
classifier.show_most_informative_features(15)

    
# Naive Bayes classifier 
# now we can print out the accuracy now in % based on testing set
#
print("Original NaiveBayesClassifier accuracy %: ", (nltk.classify.accuracy(classifier, testing_set)) * 100)

# now run Training/Testing on various other
# Predictive Models:

# Multinomial Naive Bayes Classifier:
#
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB Classifier accuracy %: ", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

# GaussianNB Naive Bayes Classifier:
#
#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print("GaussianNB Classifier accuracy %: ", (nltk.classify.accuracy(GaussianNB_classifier, testing_set)) * 100)

# BernoulliNB Naive Bayes Classifier:
#
BernoulliNB_classifier = SklearnClassifier(MultinomialNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB Classifier accuracy %: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC

# LogisticRegression Naive Bayes Classifier:
#
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Classifier accuracy %: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)


# SGDC Classifier:
#
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Classifier accuracy %: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

# SVC Classifier:
# withdrawn as it produces poor results
#
#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(training_set)
#print("SVC Classifier accuracy %: ", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)

# LinearSVC Classifier:
#
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Classifier accuracy %: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

# NuSVC Classifier:
#
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Classifier accuracy %: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

# now we'll build our own classifier which is
# a compilation of all the classifier
# don't use SVC as it produced poor results
# we have seven classifiers
#

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)


print("voted_classifier accuracy %: ", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

print("Done")

print("Classification: ", voted_classifier.classify(testing_set[0][0]),
     "Confidence 0 %: ", voted_classifier.confidence(testing_set[0][0])* 100)

print("Classification: ", voted_classifier.classify(testing_set[1][0]),
     "Confidence 1 %: ", voted_classifier.confidence(testing_set[1][0])* 100)

print("Classification: ", voted_classifier.classify(testing_set[2][0]),
     "Confidence 2 %: ", voted_classifier.confidence(testing_set[2][0])* 100)

print("Classification: ", voted_classifier.classify(testing_set[3][0]),
     "Confidence 3 %: ", voted_classifier.confidence(testing_set[3][0])* 100)

print("Classification: ", voted_classifier.classify(testing_set[4][0]),
     "Confidence 4 %: ", voted_classifier.confidence(testing_set[4][0])* 100)

print("Classification: ", voted_classifier.classify(testing_set[5][0]),
     "Confidence 5 %: ", voted_classifier.confidence(testing_set[5][0])* 100)

'''
Output Results Example:
data_path:  C:\ML_Data\_sentdex_data\03 - Python - Natural Lang Processing\ 
pickle_path:  C:\ML_Data\_sentdex_data\03 - Python - Natural Lang Processing\movie_reviews_pickle2
Data path:  C:\ML_Data\_sentdex_data\03 - Python - Natural Lang Processing\positive.txt
Most Informative Features
                 routine = True              neg : pos    =     15.0 : 1.0
            refreshingly = True              pos : neg    =     13.0 : 1.0
                   waste = True              neg : pos    =      9.8 : 1.0
                chilling = True              pos : neg    =      9.7 : 1.0
              meandering = True              neg : pos    =      9.7 : 1.0
                  tender = True              pos : neg    =      9.0 : 1.0
                   jokes = True              neg : pos    =      8.8 : 1.0
                supposed = True              neg : pos    =      8.6 : 1.0
                resonant = True              pos : neg    =      8.3 : 1.0
               absorbing = True              pos : neg    =      7.8 : 1.0
                     wry = True              pos : neg    =      7.7 : 1.0
                  animal = True              neg : pos    =      7.7 : 1.0
               affecting = True              pos : neg    =      7.4 : 1.0
                      tv = True              neg : pos    =      7.4 : 1.0
                portrait = True              pos : neg    =      7.3 : 1.0
Original NaiveBayesClassifier accuracy %:  64.30722891566265
MNB Classifier accuracy %:  65.06024096385542
BernoulliNB Classifier accuracy %:  65.06024096385542
LogisticRegression Classifier accuracy %:  65.06024096385542
SGDClassifier Classifier accuracy %:  63.55421686746988
LinearSVC Classifier accuracy %:  63.403614457831324
NuSVC Classifier accuracy %:  60.54216867469879
voted_classifier accuracy %:  65.51204819277109
Done
'''
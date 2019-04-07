'''
Created on Mar 23, 2019

@author: nasir uddin
'''
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
##nltk.download()

def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict


def main():
    neg_reviews = []
    pos_reviews = []
    
    #load the positive review
    for fileid in movie_reviews.fileids('pos'):
        words = movie_reviews.words(fileid)
        pos_reviews.append((create_word_features(words), "positive"))
 
    #load the positive review
    #load the negative review
    for fileid in movie_reviews.fileids('neg'):
        words1 = movie_reviews.words(fileid)
        neg_reviews.append((create_word_features(words1), "negative"))
    #load the negative review

    #ready train_set and test_set
    train_set = neg_reviews[:750] + pos_reviews[:750]
    test_set = neg_reviews[750:] + pos_reviews[750:]
    
    #ready train_set and test_set
     
    #Training Text Classification Model and Evaluating The Model
    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = nltk.classify.util.accuracy(classifier, test_set)
    print(accuracy * 100) 
    #Training Text Classification Model and Evaluating The Model

if __name__ == '__main__':
     main()


'''
Created on Mar 28, 2019

@author: nasir_uddin
'''
import numpy as np  
import re  
import nltk  
from sklearn.datasets import load_files  
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle  
from nltk.corpus import stopwords  

documents = []
#nltk.download('stopwords')  


movie_data = load_files(r"txt_sentoken")  
#print("the len value is ", len(movie_data.data))
X, y = movie_data.data, movie_data.target  



stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):  
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)
    
    # convert text documents into TFIDF feature values
     
tfidfconverter = TfidfVectorizer(max_features=2000,stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()  

 
# convert text documents into TFIDF feature values
#split Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
#split Training and Testing Sets
 
#Training Text Classification Model and Predicting Sentiment
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
#Training Text Classification Model and Predicting Sentiment

#Evaluating the Model
print(accuracy_score(y_test, y_pred))  
#Evaluating the Model
 
#Saving and Loading the Model
with open('text_classifier', 'wb') as picklefile:  
    pickle.dump(gnb,picklefile)
#Saving and Loading the Model

#To load the model and predict
with open('text_classifier', 'rb') as training_model:  
    model = pickle.load(training_model)
y_pred2 = model.predict(X_test)
print(accuracy_score(y_test, y_pred2))
#To load the model

  

if __name__ == '__main__':
    pass


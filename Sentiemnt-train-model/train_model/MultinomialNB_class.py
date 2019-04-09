'''
Created on Apr 10, 2019

@author: red5-nasir
'''
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np  
import re  
import nltk  
from sklearn.datasets import load_files  
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer 
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
    
vectorizer = CountVectorizer(max_features=1500,stop_words=stopwords.words('english'))  
X = vectorizer.fit_transform(documents).toarray()  
tfidfconverter = TfidfTransformer()  
X = tfidfconverter.fit_transform(X).toarray()  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
nay_ob=MultinomialNB();
nay_ob.fit(X_train, y_train)
y_pred = nay_ob.predict(X_test)
print(accuracy_score(y_test, y_pred))  


#Saving and Loading the Model
with open('text_classifier', 'wb') as picklefile:  
    pickle.dump(nay_ob,picklefile)
#Saving and Loading the Model

#To load the model and predict
with open('text_classifier', 'rb') as training_model:  
    model = pickle.load(training_model)
    
print(model.predict(vectorizer.transform(["This is a bad movie"])))
if __name__ == '__main__':
    pass
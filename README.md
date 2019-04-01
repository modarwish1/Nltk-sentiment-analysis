<h1>Sentiment Analysis with Scikit-Learn</h1>
<p>We will use Python&#39;s Scikit-Learn library for machine learning to train a text classification model.</p>

<p>Following are the steps required to create a text classification model in Python:</p>

<ol>
	<li>Importing Libraries</li>
	<li>Importing The dataset</li>
	<li>Text Preprocessing</li>
	<li>Converting Text to Numbers</li>
	<li>Training and Test Sets</li>
	<li>Training Text Classification Model and Predicting Sentiment</li>
	<li>Evaluating The Model</li>
	<li>Saving and Loading the Model</li>
</ol>

<h2>Importing Libraries</h2>

<p>Execute the following script to import the required libraries:</p>

<p>import numpy as np &nbsp;<br />
import re &nbsp;<br />
import nltk &nbsp;<br />
from sklearn.datasets import load_files &nbsp;<br />
from nltk.stem import WordNetLemmatizer<br />
from sklearn.feature_extraction.text import TfidfVectorizer&nbsp;<br />
from sklearn.model_selection import train_test_split<br />
from sklearn.ensemble import RandomForestClassifier<br />
from sklearn.naive_bayes import GaussianNB<br />
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score<br />
import pickle &nbsp;<br />
from nltk.corpus import stopwords &nbsp;</p>

<h2>Importing the Dataset</h2>

<p>Execute the following script to see&nbsp;<code>load_files</code>&nbsp;function in action:</p>

<p>movie_data = load_files(r&quot;D:\txt_sentoken&quot;)</p>

<p>X, y = movie_data.data, movie_data.target</p>
<p>In the script above, the&nbsp;<code>load_files</code>&nbsp;function loads the data from both &quot;neg&quot; and &quot;pos&quot; folders into the&nbsp;<code>X</code>&nbsp;variable, while the target categories are stored in&nbsp;<code>y</code>. Here&nbsp;<code>X</code>&nbsp;is a list of 2000 string type elements where each element corresponds to single user review. Similarly,&nbsp;<code>y</code>&nbsp;is a numpy array of size 2000. If you print&nbsp;<code>y</code>&nbsp;on the screen, you will see an array of 1s and 0s. This is because, for each category, the&nbsp;<code>load_files</code>&nbsp;function adds a number to the target numpy array. We have two categories: &quot;neg&quot; and &quot;pos&quot;, therefore 1s and 0s have been added to the target array.</p>

<p>&nbsp;</p>

<h2>Text Preprocessing</h2>

<p>Execute the following script to preprocess the data:</p>

<p>documents = []<br />
nltk.download(&#39;stopwords&#39;)</p>

<p>stemmer = WordNetLemmatizer()</p>

<p>for sen in range(0, len(X)): &nbsp;<br />
&nbsp; &nbsp; # Remove all the special characters<br />
&nbsp; &nbsp; document = re.sub(r&#39;\W&#39;, &#39; &#39;, str(X[sen]))</p>

<p>&nbsp; &nbsp; # remove all single characters<br />
&nbsp; &nbsp; document = re.sub(r&#39;\s+[a-zA-Z]\s+&#39;, &#39; &#39;, document)</p>

<p>&nbsp; &nbsp; # Remove single characters from the start<br />
&nbsp; &nbsp; document = re.sub(r&#39;\^[a-zA-Z]\s+&#39;, &#39; &#39;, document)&nbsp;</p>

<p>&nbsp; &nbsp; # Substituting multiple spaces with single space<br />
&nbsp; &nbsp; document = re.sub(r&#39;\s+&#39;, &#39; &#39;, document, flags=re.I)</p>

<p>&nbsp; &nbsp; # Removing prefixed &#39;b&#39;<br />
&nbsp; &nbsp; document = re.sub(r&#39;^b\s+&#39;, &#39;&#39;, document)</p>

<p>&nbsp; &nbsp; # Converting to Lowercase<br />
&nbsp; &nbsp; document = document.lower()</p>

<p>&nbsp; &nbsp; # Lemmatization<br />
&nbsp; &nbsp; document = document.split()</p>

<p>&nbsp; &nbsp; document = [stemmer.lemmatize(word) for word in document]<br />
&nbsp; &nbsp; document = &#39; &#39;.join(document)</p>

<p>&nbsp; &nbsp; documents.append(document)</p>


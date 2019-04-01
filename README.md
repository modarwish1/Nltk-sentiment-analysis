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

<p>In the script above, the&nbsp;<code>load_files</code>&nbsp;function loads the data from both &quot;neg&quot; and &quot;pos&quot; folders into the&nbsp;<code>X</code>&nbsp;variable, while the target categories are stored in&nbsp;<code>y</code>. Here&nbsp;<code>X</code>&nbsp;is a list of 2000 string type elements where each element corresponds to single user review. Similarly,&nbsp;<code>y</code>&nbsp;is a numpy array of size 2000. If you print&nbsp;<code>y</code>&nbsp;on the screen, you will see an array of 1s and 0s. This is because, for each category, the&nbsp;<code>load_files</code>&nbsp;function adds a number to the target numpy array. We have two categories: &quot;neg&quot; and &quot;pos&quot;, therefore 1s and 0s have been added to the target array.</p>

<p>&nbsp;</p>

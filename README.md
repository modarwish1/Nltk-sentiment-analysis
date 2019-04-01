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

<p style="text-align:justify">In the script above we use&nbsp;<a href="https://stackabuse.com/using-regex-for-text-manipulation-in-python/" target="_blank">Regex Expressions from Python re library</a>&nbsp;to perform different preprocessing tasks. We start by removing all non-word characters such as special characters, numbers, etc.</p>

<p style="text-align:justify">Next, we remove all the single characters. For instance, when we remove the punctuation mark from &quot;David&#39;s&quot; and replace it with a space, we get &quot;David&quot; and a single character &quot;s&quot;, which has no meaning. To remove such single characters we use&nbsp;<code>\s+[a-zA-Z]\s+</code>&nbsp;regular expression which substitutes all the single characters having spaces on either side, with a single space.</p>

<p style="text-align:justify">Next, we use the&nbsp;<code>\^[a-zA-Z]\s+</code>&nbsp;regular expression to replace a single character from the beginning of the document, with a single space. Replacing single characters with a single space may result in multiple spaces, which is not ideal.</p>

<p style="text-align:justify">We again use the regular expression&nbsp;<code>\s+</code>&nbsp;to replace one or more spaces with a single space. When you have a dataset in bytes format, the alphabet letter &quot;b&quot; is appended before every string. The regex&nbsp;<code>^b\s+</code>&nbsp;removes &quot;b&quot; from the start of a string. The next step is to convert the data to lower case so that the words that are actually the same but have different cases can be treated equally.</p>

<p style="text-align:justify">The final preprocessing step is the&nbsp;<a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf" rel="nofollow" target="_blank">lemmatization</a>. In lemmatization, we reduce the word into dictionary root form. For instance &quot;cats&quot; is converted into &quot;cat&quot;. Lemmatization is done in order to avoid creating features that are semantically similar but syntactically different. For instance, we don&#39;t want two different features named &quot;cats&quot; and &quot;cat&quot;, which are semantically similar, therefore we perform lemmatization.</p>

<p style="text-align:justify">&nbsp;</p>



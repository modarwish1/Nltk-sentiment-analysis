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

<h2>Converting Text to Numbers</h2>

<p>Machines, unlike humans, cannot understand the raw text. Machines can only see numbers. Particularly, statistical techniques such as machine learning can only deal with numbers. Therefore, we need to convert our text into numbers.</p>

<p>You can directly convert text documents into TFIDF feature values (without first converting documents to bag of words features) using the following script:</p>

<p>from sklearn.feature_extraction.text import TfidfVectorizer</p>

<p>tfidfconverter = TfidfVectorizer(max_features=2000,stop_words=stopwords.words(&#39;english&#39;))<br />
X = tfidfconverter.fit_transform(documents).toarray() &nbsp;</p>

<p>&nbsp;</p>

<h2>Training and Testing Sets</h2>

<p>Like any other supervised machine learning problem, we need to divide our data into training and testing sets. To do so, we will use the&nbsp;<code>train_test_split</code>&nbsp;utility from the&nbsp;<code>sklearn.model_selection</code>library. Execute the following script:</p>

<p>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)</p>

<p>The above script divides data into 20% test set and 80% training set.</p>

<h2>Training Text Classification Model and Predicting Sentiment</h2>

<p>We have divided our data into training and testing set. Now is the time to see the real action. We will use the&nbsp;Naive Bayes to train our model.To train our machine learning model using the Naive Bayes algorithm we will use&nbsp;<code>GaussianNB</code>&nbsp;class from the&nbsp;<code>sklearn.naive_bayes</code>&nbsp;library. The&nbsp;<code>fit</code>&nbsp;method of this class is used to train the algorithm. We need to pass the training data and training target sets to this method. Take a look at the following script:</p>

<p>gnb = GaussianNB()<br />
gnb.fit(X_train, y_train)<br />
Finally, to predict the sentiment for the documents in our test set we can use the&nbsp;<code>predict</code>&nbsp;method of the&nbsp;GaussianNB&nbsp;class as shown below:</p>
<p>To load the model, we can use the following code:</p>

<p>with open(&#39;text_classifier&#39;, &#39;rb&#39;) as training_model: &nbsp;<br />
&nbsp; &nbsp; model = pickle.load(training_model)</p>

<p>We loaded our trained model and stored it in the&nbsp;<code>model</code>&nbsp;variable. Let&#39;s predict the sentiment for the test set using our loaded model and see if we can get the same results. Execute the following script:</p>

<p>with open(&#39;text_classifier&#39;, &#39;rb&#39;) as training_model: &nbsp;<br />
&nbsp; &nbsp; model = pickle.load(training_model)<br />
y_pred2 = model.predict(X_test)<br />
print(&quot;the accuracy level after load &quot;,accuracy_score(y_test, y_pred2))</p>

<p>y_pred = gnb.predict(X_test)</p>

<h2>Evaluating the Model:</h2>
<p>To evaluate the performance of a classification model such as the one that we just trained&nbsp;we can use accuracy score.for calculating accuracy score run the following script:&nbsp;</p>

<p>print(accuracy_score(y_test, y_pred)) &nbsp;</p>
<h2>Saving and Loading the Model</h2>

<p>We can save our model as a&nbsp;<code>pickle</code>&nbsp;object in Python. To do so, execute the following script:</p>

<p>with open(&#39;text_classifier&#39;, &#39;wb&#39;) as picklefile: &nbsp;<br />
&nbsp; &nbsp; pickle.dump(gnb,picklefile)</p>

<p>Once you execute the above script, you can see the&nbsp;<code>text_classifier</code>&nbsp;file in your working directory</p>


<p><strong>complere source code you can get in this link&nbsp;<a href="https://github.com/sapnilcsecu/Nltk-sentiment-analysis/blob/master/Sentiemnt-train-model/train_model/TFIDF_nativebayes.py" target="_self">TFIDF_nativebayes</a></strong></p>

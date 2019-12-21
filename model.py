import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import string
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
def stemmer(text):
    for i,t in enumerate(text):
        text[i] = PorterStemmer().stem(text[i])
def text_process(text):
    nopunc = ''.join([w for w in text if w not in string.punctuation])
    nodigit = ''.join([re.sub('[0-9]','',w) for w in nopunc])
    nostop = [w.lower() for w in nodigit.split() if w.lower() not in stopwords.words('english')]
    stemmer(nostop)
    return nostop


comments = pd.read_csv('train.csv')
df= comments[:20000]
X = df['comment_text']
y = df.drop(['id','comment_text'], axis=1)

x_train,x_test,y_train,y_test = train_test_split(X,y)

pipeline = Pipeline([
            ('bag_of_words', CountVectorizer(analyzer=text_process)),
            ('tfidf', TfidfTransformer()),
            ('classifier', RandomForestClassifier())
        ])
pipeline.fit(x_train,y_train)
pred = pipeline.predict(x_test)
print(accuracy_score(y_test,pred))

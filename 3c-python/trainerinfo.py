import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

dta = pd.read_csv('TrainerInfo1.csv',encoding='latin-1')
df = pd.DataFrame(dta, columns = ['StatusCode','Text'])
train, test = train_test_split(df, test_size = 0.8)
print('Length of train data')
print(len(train))
print('Length of test data')
print(len(test))
print(train.head(1))
print(test.head(1))
text_clf = Pipeline([('vect', CountVectorizer()),('clf', SGDClassifier()),])
					 
_ = text_clf.fit(train.Text, train.StatusCode)

predicted = text_clf.predict(test.Text)

res = np.mean(predicted == test.StatusCode) 
print(res)

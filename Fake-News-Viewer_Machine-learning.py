# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 21:32:32 2022

@author: franc
"""


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

############################ Importing user modules

import Modules_User.cleaning as cleaning 


############################ 
path_train = 'C:/Users/franc/Desktop/TechLabs/fake-news/train.csv'

df = pd.read_csv(path_train)

############################ dropping nan:
    
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

df['title'] = df['title'].fillna('None')
df['author'] = df['author'].fillna('None')
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

############################ removing stopwords, numbers and punctuations

df_clean = df

df_clean['title'] = df_clean['title'].apply(lambda x: cleaning.clean_numbers(x))
df_clean['title'] = df_clean['title'].apply(cleaning.clean_steapwords())
df_clean['title'] = df_clean['title'].apply(lambda x: cleaning.clean_punctuations(x))


df_clean['text'] = df_clean['text'].apply(lambda x: cleaning.clean_numbers(x))
df_clean['text'] = df_clean['text'].apply(cleaning.clean_steapwords())
df_clean['text'] = df_clean['text'].apply(lambda x: cleaning.clean_punctuations(x))

############################ Prepare for Machine Learning: Count Vectorizer

cv = TfidfVectorizer(min_df=1)

df_X = cv.fit(df_clean['text'])
df_X = cv.transform(df_clean['text'])

print('Full vector: ')
print(df_X.toarray)

df_y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)


Models = [LogisticRegression(),
          MultinomialNB(),
          RandomForestClassifier(),
          DecisionTreeClassifier(),
          AdaBoostClassifier()]

Model_accuracy = []
Model_test = []
                   
for model_name in Models:
    
    model_name.fit(X_train, y_train)
    print(model_name)

    y_pred = model_name.predict(X_test)
    y_expect = y_test
    accuracy_model = accuracy_score(y_expect, y_pred)
    Model_accuracy.append(accuracy_model)
    print (accuracy_model)

    y_pred_test = model_name.predict(X_train)
    y_expect_test = y_train
    accuracy_test = accuracy_score(y_expect_test, y_pred_test)
    Model_test.append(accuracy_test)
    print (accuracy_test)
    
    
df_models = pd.DataFrame({'Models': Models, 
                         'Accuracy': Model_accuracy , 
                         'Test': Model_test, 
                         })









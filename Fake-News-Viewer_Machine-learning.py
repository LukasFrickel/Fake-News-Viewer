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


############################ Importing user modules

import Modules_User.cleaning as cleaning 

path_train = 'C:/Users/franc/Desktop/TechLabs/fake-news/train.csv'

############################ removing stopwords, numbers and punctuations

ath_train = 'C:/Users/franc/Desktop/TechLabs/fake-news/train.csv'

df = pd.read_csv(path_train)
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

############################ dropping nan:
    
df['title'] = df['title'].fillna('None')
df['author'] = df['author'].fillna('None')
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

############################ removing stopwords, numbers and punctuations

df_stopwords = df

df_stopwords['title'] = df_stopwords['title'].apply(lambda x: cleaning.clean_numbers(x))
df_stopwords['title'] = df_stopwords['title'].apply(cleaning.clean_steapwords())
df_stopwords['title'] = df_stopwords['title'].apply(lambda x: cleaning.clean_punctuations(x))


df_stopwords['text'] = df_stopwords['text'].apply(lambda x: cleaning.clean_numbers(x))
df_stopwords['text'] = df_stopwords['text'].apply(cleaning.clean_steapwords())
df_stopwords['text'] = df_stopwords['text'].apply(lambda x: cleaning.clean_punctuations(x))

############################ Prepare for Machine Learning: Count Vectorizer

cv = TfidfVectorizer(min_df=1)

df_X = cv.fit(df_stopwords['text'])
df_X = cv.transform(df_stopwords['text'])

print('Full vector: ')
print(df_X.toarray)

df_y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)

mnb = MultinomialNB()

mnb.fit(X_train, y_train)
print(mnb)

y_pred = mnb.predict(X_test)
y_expect = y_test

print (accuracy_score(y_expect, y_pred))


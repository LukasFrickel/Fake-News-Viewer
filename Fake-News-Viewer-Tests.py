#Created by Francisco van Riel Neto

############################ Importing necessaire modelos

import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
#from nltk.tokenize import blankline_tokenize
from nltk.probability import FreqDist
#from nltk.util import bigrams, trigrams, ngrams
from nltk.stem import SnowballStemmer
import re
import string
from nltk import ne_chunk 

#nltk.download('all')

path_train = 'C:/Users/franc/Desktop/TechLabs/fake-news/train.csv'

df = pd.read_csv(path_train)
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

############################ cleaning data:
    
df['title'] = df['title'].fillna('None')
df['author'] = df['author'].fillna('None')
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

print(df.isna().sum())

############################ removing stopwords, numbers and punctuations

stopw = stopwords.words('english')

new_stepw = list()

for word in stopw:
    if "'" in word:
        word_new1 = word.replace("'","’")
        word_new2 = word.replace("'","‘")
        new_stepw.append(word_new1)
        new_stepw.append(word_new2)

stopw.extend(new_stepw)

df_stopwords = df.applymap(lambda x: x.lower() if type(x) == str else x)


punct = string.punctuation
punct = punct + "‘" + "’" + '“' + '”' + '—' + "–"

def clean_punctuations(text):
    
    text = re.sub('[%s]' % punct, '', text)

    return(text)

def clean_numbers(text):
    
    text = re.sub(r'[0-9]', '', text)

    return(text)


df_stopwords['title'] = df_stopwords['title'].apply(lambda x: clean_numbers(x))

df_stopwords['title'] = df_stopwords['title'].apply(lambda x: ' '.join([word for word in 
                                                      x.split() if word not in (stopw)]))

df_stopwords['title'] = df_stopwords['title'].apply(lambda x: clean_punctuations(x))


df_stopwords['text'] = df_stopwords['text'].apply(lambda x: clean_numbers(x))

df_stopwords['text'] = df_stopwords['text'].apply(lambda x: ' '.join([word for word in 
                                                      x.split() if word not in (stopw)]))

df_stopwords['text'] = df_stopwords['text'].apply(lambda x: clean_punctuations(x))


############################ tokenization and spliting datasets:
    
df_token = df_stopwords

df_token['title'] = df_token['title'].apply(word_tokenize)
df_token['author'] = df_token['author'].apply(word_tokenize)
df_token['text'] = df_token['text'].apply(word_tokenize)
  
############################ punctuation and stopword #oldway

#stopw = stopwords.words('english')

#punctuation = string.punctuation

#def Convert(string):
    #list1=[]
    #list1[:0]=string
    #return list1
#punctuation = (Convert(punctuation))
#punctuation.append("’")

#stop_punc = stopw + punctuation

#n = 0

#for n in range(len(df_token['title'])):
    #for word in df_token.iloc[n]['title']:
        #if word in stop_punc:
            #df_token.iloc[n]['title'].remove(word)
            #print(word)
    #n +=1 
     
    
#n = 0

#for n in range(len(df_token['text'])):
    #for word in df_token.iloc[n]['text']:
        #if word in stop_punc:
            #df_token.iloc[n]['text'].remove(word)  
            #print(word)
    #n +=1 

############################ spliting the data into reliable and unreliable

df_token_reliable = df_token.drop(df[df.label == 0].index)
df_token_unreliable = df_token.drop(df[df.label == 1].index)
    
############################ word frequencies in titles

fdist_title_reliable = FreqDist()
counter = 0

for counter in range(len(df_token_reliable['title'])):
    for word in df_token_reliable.iloc[counter]['title']:
        fdist_title_reliable[word.lower()]+=1
    counter +=1
    
fdist_title_unreliable = FreqDist()
counter = 0

for counter in range(len(df_token_unreliable['title'])):
    for word in df_token_unreliable.iloc[counter]['title']:
        fdist_title_unreliable[word.lower()]+=1
    counter +=1    

print (fdist_title_reliable.most_common(10))      
print (fdist_title_unreliable.most_common(10))

fdist_text_reliable = FreqDist()
counter = 0

for counter in range(len(df_token_reliable['text'])):
    for word in df_token_reliable.iloc[counter]['text']:
        fdist_text_reliable[word.lower()]+=1
    counter +=1
    
fdist_text_unreliable = FreqDist()
counter = 0

for counter in range(len(df_token_unreliable['text'])):
    for word in df_token_unreliable.iloc[counter]['text']:
        fdist_text_unreliable[word.lower()]+=1
    counter +=1     

print (fdist_text_reliable.most_common(10))      
print (fdist_text_unreliable.most_common(10))

############################ Stemming

stemmer = SnowballStemmer("english")

df_token_reliable_stemmed = df_token_reliable

df_token_reliable_stemmed['title'] = df_token_reliable['title'].apply(lambda x: [stemmer.stem(y) for y in x])  # Stem every word.
df_token_reliable_stemmed['text'] = df_token_reliable['text'].apply(lambda x: [stemmer.stem(y) for y in x])  # Stem every word.

df_token_unreliable_stemmed = df_token_unreliable

df_token_unreliable_stemmed['title'] = df_token_unreliable['title'].apply(lambda x: [stemmer.stem(y) for y in x])  # Stem every word.
df_token_unreliable_stemmed['text'] = df_token_unreliable['text'].apply(lambda x: [stemmer.stem(y) for y in x])  # Stem every word.

############################ POS - Parts of Speech

df_token_reliable_pos = df_token_reliable
df_token_reliable_pos['title'] = df_token_reliable_pos['title'].apply(nltk.pos_tag)
df_token_reliable_pos['text'] = df_token_reliable_pos['text'].apply(nltk.pos_tag)


df_token_unreliable_pos = df_token_unreliable
df_token_unreliable_pos['title'] = df_token_unreliable_pos['title'].apply(nltk.pos_tag)
df_token_unreliable_pos['text'] = df_token_unreliable_pos['text'].apply(nltk.pos_tag)

############################ Name ententity

df_token_reliable_ne = df_token_reliable_pos
df_token_reliable_ne['title'] = df_token_reliable_ne['title'].apply(ne_chunk)
df_token_reliable_ne['text'] = df_token_reliable_ne['text'].apply(ne_chunk)


df_token_unreliable_ne = df_token_unreliable_pos
df_token_unreliable_ne['title'] = df_token_unreliable_ne['title'].apply(ne_chunk)
df_token_unreliable_ne['text'] = df_token_unreliable_ne['text'].apply(ne_chunk)
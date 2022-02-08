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

#path_train = 'C:/Users/Francisco Riel Neto/Desktop/Github/Fake-News-Viewer/fake-news/train.csv'
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
  

############################ spliting the data into reliable and unreliable

df_token_reliable = df_token.drop(df[df.label == 0].index)
df_token_unreliable = df_token.drop(df[df.label == 1].index)
   
############################ word frequencies in titles

fdist_title_reliable = FreqDist()

for counter in range(len(df_token_reliable['title'])):
    for word in df_token_reliable.iloc[counter]['title']:
        fdist_title_reliable[word.lower()]+=1
    counter +=1
    
fdist_title_unreliable = FreqDist()

for counter in range(len(df_token_unreliable['title'])):
    for word in df_token_unreliable.iloc[counter]['title']:
        fdist_title_unreliable[word.lower()]+=1
    counter +=1    

print ('10 most commun words in the Reliable title: \n %s' % fdist_title_reliable.most_common(10))      
print ('10 most commun words in the Unreliable title: \n %s' % fdist_title_unreliable.most_common(10))

fdist_text_reliable = FreqDist()

for counter in range(len(df_token_reliable['text'])):
    for word in df_token_reliable.iloc[counter]['text']:
        fdist_text_reliable[word.lower()]+=1
    
fdist_text_unreliable = FreqDist()

for counter in range(len(df_token_unreliable['text'])):
    for word in df_token_unreliable.iloc[counter]['text']:
        fdist_text_unreliable[word.lower()]+=1    

print ('10 most commun words in the Reliable texts: \n %s' % fdist_text_reliable.most_common(10))      
print ('10 most commun words in the Unreliable texts: \n %s' % fdist_text_unreliable.most_common(10))

############################ dist. frequencies ploter

def dist_freq_plot(fdist, n, name, u_color):
    
    x = fdist.most_common(n)
    df_fdist = pd.DataFrame(x)
    df_fdist.columns = ['Words', 'Frequency']

    ax = df_fdist.plot(x = 'Words', y = 'Frequency', color = u_color, title = name, kind = 'bar', figsize=(15, 10), legend=True, fontsize=12)
    ax.set_xlabel("Words", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    return(ax)

fdist_title_reliable_plot = dist_freq_plot(fdist_title_reliable , 10, 
                                           '10 most commun words in the Reliable title', 'blue')
fdist_text_reliable_plot = dist_freq_plot(fdist_text_reliable , 10, 
                                          '10 most commun words in the Reliable texts', 'red')

fdist_title_unreliable_plot = dist_freq_plot(fdist_title_unreliable , 10, 
                                             '10 most commun words in the Unreliable title', 'green')
fdist_text_unreliable_plot = dist_freq_plot(fdist_text_unreliable , 10, 
                                            '10 most commun words in the Unreliable texts', 'cyan')

############################ Stemming

stemmer = SnowballStemmer("english")

df_token_reliable_stemmed = df_token_reliable
df_token_reliable_stemmed['title'] = df_token_reliable['title'].apply(lambda x: [stemmer.stem(y) for 
                                                                                 y in x])  # Stem every word.
df_token_reliable_stemmed['text'] = df_token_reliable['text'].apply(lambda x: [stemmer.stem(y) for 
                                                                               y in x])  # Stem every word.

df_token_unreliable_stemmed = df_token_unreliable
df_token_unreliable_stemmed['title'] = df_token_unreliable['title'].apply(lambda x: [stemmer.stem(y) for 
                                                                                     y in x])  # Stem every word.
df_token_unreliable_stemmed['text'] = df_token_unreliable['text'].apply(lambda x: [stemmer.stem(y) for 
                                                                                   y in x])  # Stem every word.

############################ POS - Parts of Speech

df_token_reliable_pos = df_token_reliable
df_token_reliable_pos['title'] = df_token_reliable_pos['title'].apply(nltk.pos_tag)
df_token_reliable_pos['text'] = df_token_reliable_pos['text'].apply(nltk.pos_tag)


df_token_unreliable_pos = df_token_unreliable
df_token_unreliable_pos['title'] = df_token_unreliable_pos['title'].apply(nltk.pos_tag)
df_token_unreliable_pos['text'] = df_token_unreliable_pos['text'].apply(nltk.pos_tag)

############################ POS - Parts of Speech - plots

#nltk.help.upenn_tagset()


tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 
        'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 
        'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

counter = 0

tags_title_reliable = FreqDist()

for counter in range(len(df_token_reliable_pos['title'])):
    for word in df_token_reliable_pos.iloc[counter]['title']:
        if word[1] in tags:
            tags_title_reliable[word[1]]+=1

tags_text_reliable = FreqDist()

for counter in range(len(df_token_reliable_pos['text'])):
    for word in df_token_reliable_pos.iloc[counter]['text']:
        if word[1] in tags:
            tags_text_reliable[word[1]]+=1

print ('10 most commun POS in the Reliable titles: \n %s' % tags_title_reliable.most_common(10))
print ('10 most commun POS in the Reliable texts: \n %s' % tags_text_reliable.most_common(10))

tags_title_unreliable = FreqDist()

for counter in range(len(df_token_unreliable_pos['title'])):
    for word in df_token_unreliable_pos.iloc[counter]['title']:
        if word[1] in tags:
            tags_title_unreliable[word[1]]+=1

tags_text_unreliable = FreqDist()

for counter in range(len(df_token_unreliable_pos['text'])):
    for word in df_token_unreliable_pos.iloc[counter]['text']:
        if word[1] in tags:
            tags_text_unreliable[word[1]]+=1

print ('10 most commun POS in the Unreliable titles: \n %s' % tags_title_unreliable.most_common(10))
print ('10 most commun POS in the Unreliable texts: \n %s' % tags_text_unreliable.most_common(10))

fdist_title_reliable_plot = dist_freq_plot(tags_title_reliable.most_common(10) , 10, 
                                           '10 most commun POS in the Reliable titles', 'blue')
fdist_text_reliable_plot = dist_freq_plot(tags_text_reliable.most_common , 10, 
                                          '10 most commun POS in the Reliable texts', 'red')

fdist_title_unreliable_plot = dist_freq_plot(tags_title_unreliable.most_common(10) , 10, 
                                             '10 most commun POS in the Unreliable titles', 'green')
fdist_text_unreliable_plot = dist_freq_plot(tags_text_unreliable.most_common(10) , 10, 
                                            '10 most commun POS in the Unreliable texts', 'cyan')
   
############################ Name ententity

df_token_reliable_ne = df_token_reliable_pos
df_token_reliable_ne['title'] = df_token_reliable_ne['title'].apply(ne_chunk)
df_token_reliable_ne['text'] = df_token_reliable_ne['text'].apply(ne_chunk)


df_token_unreliable_ne = df_token_unreliable_pos
df_token_unreliable_ne['title'] = df_token_unreliable_ne['title'].apply(ne_chunk)
df_token_unreliable_ne['text'] = df_token_unreliable_ne['text'].apply(ne_chunk)
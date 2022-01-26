#Created by Francisco van Riel Neto

import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.tokenize import blankline_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

#nltk.download('all')

#path_train = '//gilch.ad.hhu.de/Daten/Home/Riel Neto/Privat/TechLabs/fake-news/train.csv'
path_train = 'C:/Users/franc/Desktop/TechLabs/fake-news/train.csv'

df = pd.read_csv(path_train)
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

# cleaning data and spliting datasets:
    
df['title'] = df['title'].fillna('None')
df['author'] = df['author'].fillna('None')
df = df[df['text'].notna()]
df.reset_index(drop=True, inplace=True)

print(df.isna().sum())

df_reliable = df.drop(df[df.label == 0].index)
df_unreliable = df.drop(df[df.label == 1].index)

# tokenization and spliting datasets:
    
df_token = df

df_token['title'] = df_token['title'].apply(word_tokenize)
df_token['author'] = df_token['author'].apply(word_tokenize)
df_token['text'] = df_token['text'].apply(word_tokenize)

#spliting the data into reliable and unreliable

df_token_reliable = df_token.drop(df[df.label == 0].index)
df_token_unreliable = df_token.drop(df[df.label == 1].index)

#word frequencies

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
     
# number of paragraphs


print(blankline_tokenize(df_token_reliable['title']))
print(blankline_tokenize(df_token_unreliable['title']))

# word cloud plot:

df_reliable_text = ' '.join(df_reliable['text'].tolist())
df_unreliable_text = ' '.join(df_unreliable['text'].tolist())

word_cloud_true = WordCloud(max_font_size=50, max_words=100, 
                            collocations = False, 
                            stopwords=STOPWORDS, 
                            background_color = 'white', 
                            width=400, 
                            height=300).generate(df_reliable_text)

word_cloud_fake = WordCloud(max_font_size=50, 
                            max_words=100, 
                            collocations = False, 
                            stopwords=STOPWORDS, 
                            background_color = 'black', 
                            width=400,
                            height=300).generate(df_unreliable_text)

fig, ax = plt.subplots(1, 2, figsize  = (30,30))

ax[0].imshow(word_cloud_true, interpolation = 'bilinear')
ax[0].set_title('Word Cloud Reliable', fontsize=30)
ax[0].axis('off')   
     
ax[1].imshow(word_cloud_fake, interpolation = 'bilinear')
ax[1].set_title('Word Cloud Unreliable', fontsize=30)
ax[1].axis('off')
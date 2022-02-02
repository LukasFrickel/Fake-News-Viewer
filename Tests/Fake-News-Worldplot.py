#Created by Francisco van Riel Neto

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

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
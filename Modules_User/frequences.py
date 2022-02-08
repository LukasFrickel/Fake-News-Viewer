# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 23:04:05 2022

@author: franc
"""
import pandas as pd
from nltk.probability import FreqDist

def freq_dist(df, collumn):
    
    freq_dist = FreqDist()

    for counter in range(len(df[collumn])):
        for word in df.iloc[counter][collumn]:
            freq_dist[word.lower()]+=1
    
    return(freq_dist)

def dist_freq_plot(fdist, n, name, u_color):
    
    x = fdist.most_common(n)
    df_fdist = pd.DataFrame(x)
    df_fdist.columns = ['Words', 'Frequency']

    ax = df_fdist.plot(x = 'Words', y = 'Frequency', color = u_color, title = name, kind = 'bar', figsize=(15, 10), legend=True, fontsize=12)
    ax.set_xlabel("Words", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    return(ax)

#nltk.help.upenn_tagset()

def freq_dist_tags(df, collumn):
    
    tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS',
            'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 
            'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    
    freq_dist = FreqDist()

    for counter in range(len(df[collumn])):
        for word in df.iloc[counter][collumn]:
            if word[1] in tags:
                freq_dist[word[1]]+=1
                
    return(freq_dist)
                
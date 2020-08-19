# -*- coding: utf-8 -*-

#Dataframes
import pandas as pd

#Math tools
import numpy as np

#Text cleaning
from collections import defaultdict
from nltk.corpus import stopwords
from Preprocess import Prep
import string
from wordcloud import STOPWORDS
from wordcloud import WordCloud

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/NLP_with_ Disaster_Tweets/train.csv', engine='python')
df_test = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/NLP_with_ Disaster_Tweets/test.csv', engine='python')

#Replacing 'target' attribute values for better data visualization
df_train['target'].replace(0, 'Not Real', inplace=True)
df_train['target'].replace(1, 'Real', inplace=True)

#Plot parameters
plt.style.use('ggplot')
colors=["#feb308", "#3778bf"]
sns.set_palette(sns.color_palette(colors))

#Amount of real and not real cases
sns.countplot(x=df_train['target'], alpha=0.7)
plt.ylabel('Count')
plt.xlabel('Target (Disaster)')
plt.title('Real or Not Count')
plt.savefig('RealOrNotCount.png')
plt.show()


#Preprocessing locations names generalizing to countries
df_train['location'].replace({'United States':'USA',
                           'New York':'USA',
                            "London":'UK',
                            "Los Angeles, CA":'USA',
                            "Washington, D.C.":'USA',
                            "Washington, DC":'USA',
                            "California":'USA',
                             "Chicago, IL":'USA',
                             "Chicago":'USA',
                            "New York, NY":'USA',
                            "Sacramento, CA":'USA',
                            "Nashville, TN":'USA',
                            "Pennsylvania, USA":'USA',
                            "Manchester":'England',
                            "California, USA":'USA',
                            "FLorida":'USA',
                            "Nigeria":'Africa',
                            "Kenya":'Africa',
                            "Everywhere":'Worldwide',
                            "San Francisco":'USA',
                            "Florida":'USA',
                            "United Kingdom":'UK',
                            "Los Angeles":'USA',
                            "Toronto":'Canada',
                            "San Francisco, CA":'USA',
                            "Denver, Colorado":'USA',
                            "NYC":'USA',
                            "Seattle":'USA',
                            "US":'USA',
                            "Paterson, New Jersey":'USA',
                            "Oklahoma City, OK":'USA',
                            "Memphis, TN":'USA',
                            "Ireland":'UK',
                            "London, England":'UK',
                            "New York City":'USA',
                            "Texas":'USA',
                            "London, UK":'UK',
                            "Atlanta, GA":'USA',
                            "Mumbai":"India",
                            "Brazil":"Brazil",
                            "Brasil":"Brazil",
                            "São Paulo":"Brazil",
                            "Russia":"Russia",
                            "Germany":"Germany",
                            "Earth":'Worldwide'},inplace=True)

#10 Most frequent locations in tweets
sns.barplot(y=df_train['location'].value_counts()[:10].index, x=df_train['location'].value_counts()[:10], alpha=0.5,
            orient='h')
plt.ylabel('Locations')
plt.xlabel('Count')
plt.title('10 Most Frequent Locations in Tweets')
plt.savefig('TopLocations.png')
plt.show()


#Transforming string to int, if necessary, to aggregate data
df_train['target'].replace('Not Real', 0, inplace=True)
df_train['target'].replace('Real', 1, inplace=True)
#10 Most Frequent Locations in Tweets by Real and Not Real Disasters
df_train_l1 = df_train[df_train['target']==1]['location'].dropna()
df_train_l0 = df_train[df_train['target']==0]['location'].dropna()
#Plot
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)
sns.barplot(x=df_train_l1.value_counts()[:10], y=df_train_l1.value_counts()[:10].index, color="#feb308", alpha=0.5, ax=axes[0])
axes[0].set_title('Real Disaster Tweets', size=10)
axes[0].set_xlabel('Count', size=10, labelpad=20)
axes[0].set_ylabel('Locations', size=10, labelpad=20)
sns.barplot(x=df_train_l0.value_counts()[:10], y=df_train_l0.value_counts()[:10].index, color="#3778bf", alpha=0.5, ax=axes[1])
axes[1].set_title('Not Real Disaster Tweets', size=10)
axes[1].set_xlabel('Count', size=10, labelpad=20)
fig.suptitle('10 Most Frequent Locations in Tweets by Real and Not Real Disasters', size=14, y=1)
plt.savefig('TopLocationsByDisasters.png')
plt.show()


#Missing values of 'keyword' and 'location' attributes in each dataset
missing_cols = ['keyword', 'location']
#Plot
fig, axes = plt.subplots(ncols=2, figsize=(10, 3), dpi=100)
sns.barplot(x=df_train[missing_cols].isnull().sum().index, y=df_train[missing_cols].isnull().sum().values, alpha=0.7, ax=axes[0])
sns.barplot(x=df_test[missing_cols].isnull().sum().index, y=df_test[missing_cols].isnull().sum().values, alpha=0.7, ax=axes[1])
axes[0].set_ylabel('Missing Value Count', size=10, labelpad=20)
axes[0].tick_params(axis='x', labelsize=10)
axes[0].tick_params(axis='y', labelsize=10)
axes[1].tick_params(axis='x', labelsize=10)
axes[1].tick_params(axis='y', labelsize=10)
axes[0].set_title('Training Set', fontsize=8)
axes[1].set_title('Test Set', fontsize=8)
plt.savefig('MissingValues.png')
plt.show()


#Slice of data of real and not real cases by keyword
#Transforming string to int, if necessary, to aggregate data
df_train['target'].replace('Not Real', 0, inplace=True)
df_train['target'].replace('Real', 1, inplace=True)
#Creating a new parameter attribute
df_train['target_mean'] = df_train.groupby('keyword')['target'].transform('mean')
#Transforming int to string for better visualization
df_train['target'].replace(0, 'Not Real', inplace=True)
df_train['target'].replace(1, 'Real', inplace=True)

#Plot
fig, axes = plt.subplots(nrows=3, figsize=(10, 14), dpi=100)
#Most frequent keywords associated with real disasters
sns.countplot(x=df_train.sort_values(by='target_mean', ascending=False)['keyword'].iloc[:400], hue=df_train.sort_values(by='target_mean', ascending=False)['target'], alpha=0.7, ax=axes[0])
#Keywords associated with a mixture of real and not real disasters
sns.countplot(x=df_train.sort_values(by='target_mean', ascending=False)['keyword'].iloc[3000:3400], hue=df_train.sort_values(by='target_mean', ascending=False)['target'], alpha=0.7, ax=axes[1])
#Most frequent keywords associated with not real disasters
sns.countplot(x=df_train.sort_values(by='target_mean', ascending=True)['keyword'].iloc[:400], hue=df_train.sort_values(by='target_mean', ascending=False)['target'], alpha=0.7, ax=axes[2])
axes[0].tick_params(axis='x', labelsize=8, rotation=20)
axes[0].tick_params(axis='y', labelsize=10)
axes[1].tick_params(axis='x', labelsize=8, rotation=20)
axes[1].tick_params(axis='y', labelsize=10)
axes[2].tick_params(axis='x', labelsize=8, rotation=20)
axes[2].tick_params(axis='y', labelsize=10)
plt.title('Most Frequent Keywords in Real and Not Real Disasters', size=10, y=3.45)
plt.savefig('TopKeywords.png')
plt.show()

#Dropping parameter and transforming string to int
df_train.drop(columns=['target_mean'], inplace=True)
df_train['target'].replace('Not Real', 0, inplace=True)
df_train['target'].replace('Real', 1, inplace=True)


#Amount of characters in tweets
df_train_c1 = df_train[df_train['target']==1]['text'].str.len()
df_train_c0 = df_train[df_train['target']==0]['text'].str.len()
#Plot
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)
axes[0].hist(df_train_c1, color='#feb308', alpha=0.5)
axes[0].set_title('Real Disaster Tweets', size=10)
axes[0].set_xlabel('Characters Count', size=10, labelpad=20)
axes[0].set_ylabel('Tweets Count', size=10, labelpad=20)
axes[1].hist(df_train_c0, color='#3778bf', alpha=0.5)
axes[1].set_title('Not Real Disaster Tweets', size=10)
axes[1].set_xlabel('Characters Count', size=10, labelpad=20)
fig.suptitle('Characters Count in Tweets', size=14, y=1)
plt.savefig('CharactersCount.png')
plt.show()


#Amount of words in tweets
df_train_w1 = df_train[df_train['target']==1]['text'].str.split().map(lambda x: len(x))
df_train_w0 = df_train[df_train['target']==0]['text'].str.split().map(lambda x: len(x))
#Plot
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)
axes[0].hist(df_train_w1, color='#feb308', alpha=0.5)
axes[0].set_title('Real Disaster Tweets', size=10)
axes[0].set_xlabel('Words Count', size=10, labelpad=20)
axes[0].set_ylabel('Tweets Count', size=10, labelpad=20)
axes[1].hist(df_train_w0, color='#3778bf', alpha=0.5)
axes[1].set_title('Not Real Disaster Tweets', size=10)
axes[1].set_xlabel('Words Count', size=10, labelpad=20)
fig.suptitle('Words Count in Tweets', size=14, y=1)
plt.savefig('WordsTweets.png')
plt.show()

#Amount of unique words in tweets
df_train_uw1 = df_train[df_train['target']==1]['text'].apply(lambda x: len(set(str(x).split())))
df_train_uw0 = df_train[df_train['target']==0]['text'].apply(lambda x: len(set(str(x).split())))
#Plot
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)
axes[0].hist(df_train_uw1, color='#feb308', alpha=0.5)
axes[0].set_title('Real Disaster Tweets', size=10)
axes[0].set_xlabel('Unique Words Count', size=10, labelpad=20)
axes[0].set_ylabel('Tweets Count', size=10, labelpad=20)
axes[1].hist(df_train_uw0, color='#3778bf', alpha=0.5)
axes[1].set_title('Not Real Disaster Tweets', size=10)
axes[1].set_xlabel('Unique Words Count', size=10, labelpad=20)
fig.suptitle('Unique Words Count in Tweets', size=14, y=1)
plt.savefig('UniqWords.png')
plt.show()


#Distribution of avg. words length in ea tweet
df_train_aw1 = df_train[df_train['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x]) #New dataframe for avg. words length in real cases
df_train_aw0 = df_train[df_train['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x]) #New dataframe for avg. words length in not real cases
#Plot
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)
sns.distplot(df_train_aw1.map(lambda x: np.mean(x)), color='#feb308',ax=axes[0])
axes[0].set_title('Real Disaster Tweets', size=10)
axes[0].set_xlabel('Average Word Length', size=10, labelpad=20)
axes[0].set_ylabel('Ratio of ea. Tweet by Total Tweets', size=10, labelpad=20)
sns.distplot(df_train_aw0.map(lambda x: np.mean(x)), color='#3778bf', ax=axes[1])
axes[1].set_title('Not Real Disaster Tweets', size=10)
axes[1].set_xlabel('Average Word Count', size=10, labelpad=20)
fig.suptitle('Distribution of Avg. Word Length in ea. Tweet', size=14, y=1)
plt.savefig('AvgWordsDist.png')
plt.show()


#Distribution of stop words count in ea tweet
df_train_sw1 = df_train[df_train['target']==1]['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
df_train_sw0 = df_train[df_train['target']==0]['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
#Plot
fig, axes = plt.subplots(ncols=2, figsize=(8, 4), dpi=100)
sns.distplot(df_train_sw1, color='#feb308',ax=axes[0])
axes[0].set_title('Real Disaster Tweets', size=10)
axes[0].set_xlabel('Stop Words Count', size=10, labelpad=20)
axes[0].set_ylabel('Ratio of ea. Tweet by Total Tweets', size=10, labelpad=20)
sns.distplot(df_train_sw0, color='#3778bf', ax=axes[1])
axes[1].set_title('Not Real Disaster Tweets', size=10)
axes[1].set_xlabel('Stop Words Count', size=10, labelpad=20)
fig.suptitle('Distribution of Stop Words Count in ea. Tweet', size=14, y=1)
plt.savefig('StopWordsDist.png')
plt.show()

#Counting the most common stop words in each disaster case
stop=set(stopwords.words('english'))

def create_corpus(target):
    corpus=[]
    
    for x in df_train[df_train['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus

#Not real disaster
corpus=create_corpus(0)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
        
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

x,y=zip(*top)

plt.bar(x,y, alpha=0.5, color='#3778bf')
plt.ylabel('Count')
plt.xlabel('Stop Words')
plt.title('Most Common Stop Words in Not Real Disasters', y=1.05)
plt.savefig('TopStopWords0.png')
plt.show()

#Real disaster
corpus=create_corpus(1)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
    
x,y=zip(*top)

plt.bar(x,y, alpha=0.5, color='#feb308')
plt.ylabel('Count')
plt.xlabel('Stop Words')
plt.title('Most Common Stop Words in Real Disasters', y=1.05)
plt.savefig('TopStopWords1.png')
plt.show()


#Distribution of punctuation count in ea tweet
df_train_p1 = df_train[df_train['target']==1]['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df_train_p0 = df_train[df_train['target']==0]['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
#Plot
fig, axes = plt.subplots(ncols=2, figsize=(10, 4), dpi=100)
sns.distplot(df_train_p1, color='#feb308',ax=axes[0])
axes[0].set_title('Real Disaster Tweets', size=10)
axes[0].set_xlabel('Punctuations Count', size=10, labelpad=20)
axes[0].set_ylabel('Ratio of ea. Tweet by Total Tweets', size=10, labelpad=20)
sns.distplot(df_train_p0, color='#3778bf', ax=axes[1])
axes[1].set_title('Not Real Disaster Tweets', size=10)
axes[1].set_xlabel('Punctuations Count', size=10, labelpad=20)
fig.suptitle('Distribution of Punctuation in ea. Tweet', size=14, y=1)
plt.savefig('PunctuationDist.png')
plt.show()


#Counting the most common punctuation in each disaster case
#Not real
corpus=create_corpus(0)

dic=defaultdict(int)
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1
        
x,y=zip(*dic.items())
#Plot
plt.figure(figsize=(10,5))
plt.bar(x, y, alpha=0.5, color='#3778bf')
plt.ylabel('Count')
plt.xlabel('Punctuation Characters')
plt.title('Most Common Punct. Chars. in Not Real Disasters', y=1)
plt.savefig('TopPunct0.png')
plt.show()


#Real
corpus=create_corpus(1)

dic=defaultdict(int)
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i]+=1
        
x,y=zip(*dic.items())
#Plot
plt.figure(figsize=(10,5))
plt.bar(x, y, alpha=0.5, color='#feb308')
plt.ylabel('Count')
plt.xlabel('Punctuation Characters')
plt.title('Most Common Punct. Chars. in Real Disasters', y=1)
plt.savefig('TopPunct1.png')
plt.show()


#Distribution of punctuation count in ea tweet
df_train_h1 = df_train[df_train['target']==1]['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
df_train_h0 = df_train[df_train['target']==0]['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
#Plot
fig, axes = plt.subplots(ncols=2, figsize=(10, 4), dpi=100)
sns.distplot(df_train_h1, color='#feb308',ax=axes[0])
axes[0].set_title('Real Disaster Tweets', size=10)
axes[0].set_xlabel('Hashtag Count', size=10, labelpad=20)
axes[0].set_ylabel('Ratio of ea. Tweet by Total Tweets', size=10, labelpad=20)
sns.distplot(df_train_h0, color='#3778bf', ax=axes[1])
axes[1].set_title('Not Real Disaster Tweets', size=10)
axes[1].set_xlabel('Hashtag Count', size=10, labelpad=20)
fig.suptitle('Distribution of Punctuation in ea. Tweet', size=14, y=1)
plt.savefig('HashtagDist.png')
plt.show()


#Word clouds for clean text data
df_train = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/NLP_with_ Disaster_Tweets/train.csv', engine='python')
df_train['text'] = df_train['text'].apply(lambda x: Prep.preprocess(x))
#Plot
fig, axes = plt.subplots(ncols=2, figsize=(20, 8))

wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(df_train[df_train['target']==1]['text']))
axes[0].imshow(wordcloud1)
axes[0].set_title('Real Disaster Tweets', fontsize=30, y=1.05)
axes[0].set_xlabel('')
axes[0].set_ylabel('')
axes[0].axis('off')

wordcloud0 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(df_train[df_train['target']==0]['text']))
axes[1].imshow(wordcloud0)
axes[1].set_title('Not Real Disaster Tweets', fontsize=30, y=1.05)
axes[1].set_xlabel('')
axes[1].set_ylabel('')
axes[1].axis('off')
plt.savefig('WordClouds.png')
plt.show()
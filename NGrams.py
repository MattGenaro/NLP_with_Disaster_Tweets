# -*- coding: utf-8 -*-

#Dataframes
import pandas as pd

#Text cleaning
import re
import unicodedata
import nltk
from nltk.corpus import stopwords

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/NLP_with_ Disaster_Tweets/train.csv', engine='python')
df_test = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/NLP_with_ Disaster_Tweets/test.csv', engine='python')

#Simple cleaning function to generate ngrams data
def basic_clean(text):
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]

real = df_train[df_train["target"] == 1]["text"]
not_real = df_train[df_train["target"] == 0]["text"]
words1 = basic_clean(''.join(str(real.tolist())))
words0 = basic_clean(''.join(str(not_real.tolist())))

#Top 20 Unigram list
(pd.Series(nltk.ngrams(words1, 1)).value_counts())[:20]
(pd.Series(nltk.ngrams(words0, 1)).value_counts())[:20]

#Top 20 Bigram list
(pd.Series(nltk.ngrams(words1, 2)).value_counts())[:20]
(pd.Series(nltk.ngrams(words0, 2)).value_counts())[:20]

#Top 20 Trigram list
(pd.Series(nltk.ngrams(words1, 3)).value_counts())[:20]
(pd.Series(nltk.ngrams(words0, 3)).value_counts())[:20]


#Ngrams model visualization
#20 most common unigrams in real disaster tweets
real_1grams_series = (pd.Series(nltk.ngrams(words1, 1)).value_counts())[:20]
notreal_1grams_series = (pd.Series(nltk.ngrams(words0, 1)).value_counts())[:20]

#Plot
colors=["#feb308", "#3778bf"]
sns.set_palette(sns.color_palette(colors))

fig, axes = plt.subplots(ncols=2, figsize=(10, 10), dpi=100)
plt.style.use('ggplot')
plt.tight_layout()

real_1grams_series.sort_values().plot.barh(ax=axes[0], color='#feb308', alpha=0.5)
notreal_1grams_series.sort_values().plot.barh(ax=axes[1], color='#3778bf', alpha=0.5)

axes[0].spines['right'].set_visible(False)
axes[0].set_xlabel('# of Ocurrances')
axes[0].set_ylabel('Unigrams')
axes[0].tick_params(axis='x', labelsize=10)
axes[0].tick_params(axis='y', labelsize=10)

axes[1].spines['right'].set_visible(False)
axes[1].set_xlabel('# of Ocurrances')
axes[1].set_ylabel('')
axes[1].tick_params(axis='x', labelsize=10)
axes[1].tick_params(axis='y', labelsize=10)

axes[0].set_title('20 Most Frequently Occuring Unigrams in Real Disaster Tweets', fontsize=10)
axes[1].set_title('20 Most Frequently Occuring Unigrams in Not Real Disaster Tweets', fontsize=10)

plt.savefig('Top20Unigrams.png')
plt.show()

#20 most common bigrams in real disaster tweets
real_2grams_series = (pd.Series(nltk.ngrams(words1, 2)).value_counts())[:20]
notreal_2grams_series = (pd.Series(nltk.ngrams(words0, 2)).value_counts())[:20]

fig, axes = plt.subplots(ncols=2, figsize=(15, 10), dpi=100)
plt.style.use('ggplot')
plt.tight_layout()

real_2grams_series.sort_values().plot.barh(ax=axes[0], color='#feb308', alpha=0.5)
notreal_2grams_series.sort_values().plot.barh(ax=axes[1], color='#3778bf', alpha=0.5)

axes[0].spines['right'].set_visible(False)
axes[0].set_xlabel('# of Ocurrances')
axes[0].set_ylabel('Bigrams')
axes[0].tick_params(axis='x', labelsize=10)
axes[0].tick_params(axis='y', labelsize=10)

axes[1].spines['right'].set_visible(False)
axes[1].set_xlabel('# of Ocurrances')
axes[1].set_ylabel('')
axes[1].tick_params(axis='x', labelsize=10)
axes[1].tick_params(axis='y', labelsize=10)

axes[0].set_title('20 Most Frequently Occuring Bigrams in Real Disaster Tweets', fontsize=10)
axes[1].set_title('20 Most Frequently Occuring Bigrams in Not Real Disaster Tweets', fontsize=10)

plt.savefig('Top20Bigrams.png')
plt.show()


#20 most common trigrams in real disaster tweets
real_3grams_series = (pd.Series(nltk.ngrams(words1, 3)).value_counts())[:20]
notreal_3grams_series = (pd.Series(nltk.ngrams(words0, 3)).value_counts())[:20]

fig, axes = plt.subplots(ncols=2, figsize=(20, 10), dpi=100)
plt.style.use('ggplot')
plt.tight_layout()

real_3grams_series.sort_values().plot.barh(ax=axes[0], color='#feb308', alpha=0.5)
notreal_3grams_series.sort_values().plot.barh(ax=axes[1], color='#3778bf', alpha=0.5)

axes[0].spines['right'].set_visible(False)
axes[0].set_xlabel('# of Ocurrances')
axes[0].set_ylabel('Trigrams')
axes[0].tick_params(axis='x', labelsize=10)
axes[0].tick_params(axis='y', labelsize=10)

axes[1].spines['right'].set_visible(False)
axes[1].set_xlabel('# of Ocurrances')
axes[1].set_ylabel('')
axes[1].tick_params(axis='x', labelsize=10)
axes[1].tick_params(axis='y', labelsize=10)

axes[0].set_title('20 Most Frequently Occuring Unigrams in Real Disaster Tweets', fontsize=10)
axes[1].set_title('20 Most Frequently Occuring Unigrams in Not Real Disaster Tweets', fontsize=10)

plt.savefig('Top20Trigrams.png')
plt.show()
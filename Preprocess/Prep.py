# -*- coding: utf-8 -*-

#Text cleaning
import nltk
from nltk.corpus import stopwords
import re
import string


#Cleaning data
def clear(text):
    text = text.lower() #lowercase the text 
    text = re.sub('\[.*?\]', '', text) #remove text in square brackets
    text = re.sub('https?://\S+|www\.\S+', '', text) #remove links
    text = re.sub('<.*?>+', '', text) #remove html
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  #remove punctuation
    text = re.sub('\n', '', text) #remove numbers in words
    text = re.sub('\w*\d\w*', '', text)
    return text


#Remove emoji
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


#Tokenization
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


#Remove stopwords from english dictionary
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words


#Aggregate a list of texts into one large text
def aggregate_text(list_of_text):
    agg_text = ' '.join(list_of_text)
    return agg_text


#Function groups all the preprocess operations
def preprocess(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    clean = clear(text)
    remoji = remove_emoji(clean)
    tokenized_text = tokenizer.tokenize(remoji)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    agg_text = ' '.join(remove_stopwords)
    return agg_text



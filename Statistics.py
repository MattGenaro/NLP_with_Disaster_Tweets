# -*- coding: utf-8 -*-

#Data processing
import pandas as pd

#Linear algebra
import numpy as np

df_train = pd.read_csv('/NLP_with_ Disaster_Tweets/train.csv')
df_test = pd.read_csv('/NLP_with_ Disaster_Tweets/test.csv')

#Dimensions of training data
df_train.head(5)
df_train.shape
df_train.index
df_train.columns
df_train.describe()

#Samples of data
df_train[df_train["target"] == 0]["text"].values[1]
df_train[df_train["target"] == 1]["text"].values[1]

#Total missing values by attribute
df_train.isnull().sum()

#Locations values in real and not real disasters
df_train['location'].dropna()
df_train_l1 = df_train[df_train['target']==1]['location'].dropna()
df_train_l0 = df_train[df_train['target']==0]['location'].dropna()

#100 most common words in real and not real disasters
df_train_k1 = df_train[df_train.target==1].keyword.value_counts()[:100]
df_train_k0 = df_train[df_train.target==0].keyword.value_counts()[:100]

#Location Missing values statistics 
a = df_train.location.count()
b = df_train.location.isnull().sum()
print(f'In the train database, the amount of missing values in location column is = {b}')
print(f'The ratio of missing values is = {np.round((b/a)*100, 2)}%')

#Keyword Missing values statistics 
c = df_train.keyword.count()
d = df_train.keyword.isnull().sum()
print(f'In the train database, the amount of missing values in keyword column is = {d}')
print(f'The ratio of missing values is = {np.round((d/c)*100, 2)}%')

#Dimensions of test data
df_test.head(5)
df_test.shape
df_test.index
df_test.columns
df_test.describe()

#Total missing values by attribute
df_test.isnull().sum()

#Missing values statistics
e = df_test.location.count()
f = df_test.location.isnull().sum()
print(f'In the test database, the amount of missing values in location column is = {f}')
print(f'The ratio of missing values is = {np.round((f/e)*100, 2)}%')

g = df_test.keyword.count()
h = df_test.keyword.isnull().sum()
print(f'In the test database, the amount of missing values in keywrod column is = {h}')
print(f'The ratio of missing values is = {np.round((h/g)*100, 2)}%')

#Amount of unique values in 'keyword' attribute
kr = df_train["keyword"].nunique()
ke = df_test["keyword"].nunique()
print(f'Number of unique values in keyword is = {kr} (Train) and {ke} (Test)')

lr = df_train["location"].nunique()
le = df_test["location"].nunique()
print(f'Number of unique values in keyword is = {lr} (Train) and {le} (Test)')


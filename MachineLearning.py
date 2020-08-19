# -*- coding: utf-8 -*-

#Dataframes
import pandas as pd

#Model Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

#Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#Utilities
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from Preprocess import Prep

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Dataframe of work
df_train = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/NLP_with_ Disaster_Tweets/train.csv', engine='python')
df_test = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Projetos/NLP_with_ Disaster_Tweets/test.csv', engine='python')

#Preprocessing the data calling the our preprocess function
df_train['text'] = df_train['text'].apply(lambda x: Prep.preprocess(x))
df_test['text'] = df_test['text'].apply(lambda x: Prep.preprocess(x))

#Dropping not relevant columns with high amount of missing data
df_train.drop(columns=['id',"keyword",'location'], inplace=True, axis=0)
df_train.head()

df_test.drop(columns=['id',"keyword",'location'], inplace=True)
df_test.head()

#Splits data for validation test purposes
X = df_train['text']
Y = df_train['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

#Vectorize the text feature
vec = CountVectorizer()
X_vec = vec.fit_transform(X_train)


#Machine learning modeling 

#Logistic Regression
lr = LogisticRegression(random_state=0)
lr.fit(X_vec, Y_train)
pred_lr = lr.predict(vec.transform(X_test))
#Metrics to evaluate the effectiveness of the model
cmatrix_lr = confusion_matrix(Y_test, pred_lr)
acc_lr = accuracy_score(Y_test, pred_lr)
prec_lr = precision_score(Y_test, pred_lr)
rec_lr = recall_score(Y_test, pred_lr)
f1_lr = f1_score(Y_test, pred_lr)


#Naive Bayes
nb = MultinomialNB(alpha=1)
nb.fit(X_vec, Y_train)
pred_nb = nb.predict(vec.transform(X_test))
#Metrics to evaluate the effectiveness of the model
cmatrix_nb = confusion_matrix(Y_test, pred_nb)
acc_nb = accuracy_score(Y_test, pred_nb)
prec_nb = precision_score(Y_test, pred_nb)
rec_nb = recall_score(Y_test, pred_nb)
f1_nb = f1_score(Y_test, pred_nb)


#Support Vector Machine
svc = SVC(kernel='rbf', random_state=0)
svc.fit(X_vec, Y_train)
pred_svc = nb.predict(vec.transform(X_test))
#Metrics to evaluate the effectiveness of the model
cmatrix_svc = confusion_matrix(Y_test, pred_svc)
acc_svc = accuracy_score(Y_test, pred_svc)
prec_svc = precision_score(Y_test, pred_svc)
rec_svc = recall_score(Y_test, pred_svc)
f1_svc = f1_score(Y_test, pred_svc)


#Xgboost
xgb = XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
xgb.fit(X_vec, Y_train)
pred_xgb = nb.predict(vec.transform(X_test))
#Metrics to evaluate the effectiveness of the model
cmatrix_xgb = confusion_matrix(Y_test, pred_xgb)
acc_xgb = accuracy_score(Y_test, pred_xgb)
prec_xgb = precision_score(Y_test, pred_xgb)
rec_xgb = recall_score(Y_test, pred_xgb)
f1_xgb = f1_score(Y_test, pred_xgb)

#Listing metrics to dataframe
acc_l = [acc_lr, acc_nb, acc_svc, acc_xgb]
prec_l = [prec_lr, prec_nb, prec_svc, prec_xgb]
rec_l = [rec_lr, rec_nb, rec_svc, rec_xgb]
f1_l = [f1_lr, f1_nb, f1_svc, f1_xgb]

#Dataframes for each metric
cf_res_acc = pd.DataFrame({'Scores': acc_l, 'Algorithm':['LogisticRegression', 'NaiveBayes', 'SVC', 'Xgboost']})
cf_res_prc = pd.DataFrame({'Scores': prec_l, 'Algorithm':['LogisticRegression', 'NaiveBayes', 'SVC', 'Xgboost']})
cf_res_rec = pd.DataFrame({'Scores': rec_l, 'Algorithm':['LogisticRegression', 'NaiveBayes', 'SVC', 'Xgboost']})
cf_res_f1 = pd.DataFrame({'Scores': f1_l, 'Algorithm':['LogisticRegression', 'NaiveBayes', 'SVC', 'Xgboost']})


#Plot for every metric scores
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,8), dpi=100)

sns.barplot("Scores", "Algorithm", data=cf_res_acc.sort_values(by='Scores', ascending=False), palette="Greens", ax=axs[0,0])
axs[0,0].set_xlabel("Scores")
axs[0,0].set_title("Accuracy Scores")
sns.barplot("Scores", "Algorithm", data=cf_res_prc.sort_values(by='Scores', ascending=False), palette="Blues", ax=axs[0,1])
axs[0,1].set_xlabel("Scores")
axs[0,1].set_ylabel(" ")
axs[0,1].set_title("Precision Scores")
sns.barplot("Scores", "Algorithm", data=cf_res_rec.sort_values(by='Scores', ascending=False), palette="Oranges", ax=axs[1,0])
axs[1,0].set_xlabel("Scores")
axs[1,0].set_title("Recall Scores")
sns.barplot("Scores", "Algorithm", data=cf_res_f1.sort_values(by='Scores', ascending=False), palette="BuPu", ax=axs[1,1])
axs[1,1].set_xlabel("Scores")
axs[1,1].set_ylabel(" ")
axs[1,1].set_title("F1 Scores")
plt.tight_layout()
plt.savefig('MetricScores.png')
plt.show()


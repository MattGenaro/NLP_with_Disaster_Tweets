# -*- coding: utf-8 -*-

#Data processing
import pandas as pd

#Cross validation tools
from sklearn.model_selection import StratifiedKFold, train_test_split

#Hyperparameter optimization software framework
import optuna

#Machine Learning modeling
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression

#Linear algebra
import numpy as np

#Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Text cleaning
from Preprocess import Prep

#Text to vector
from sklearn.feature_extraction.text import TfidfVectorizer

#Utilities
import os 
import random 
import warnings 
from scipy import sparse

#Visualization
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv('/NLP_with_ Disaster_Tweets/train.csv')
df_test = pd.read_csv('/NLP_with_ Disaster_Tweets/test.csv')

#Preprocessing the data calling the our preprocess function
df_train['text'] = df_train['text'].apply(lambda x: Prep.preprocess(x))
df_test['text'] = df_test['text'].apply(lambda x: Prep.preprocess(x))

#Dropping not relevant columns with high amount of missing data
df_train.drop(columns=['id',"keyword",'location'], inplace=True, axis=0)
df_train.head()

df_test.drop(columns=['id',"keyword",'location'], inplace=True)
df_test.head()


#Creates a strong baseline Naive Bayes Support Vector Machine (NBSVM) model 
class NBSVMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, max_iter=100, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.max_iter = max_iter

    def predict(self, x):
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def fit(self, x, y):
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, 
                                       max_iter=self.max_iter, 
                                       n_jobs=self.n_jobs).fit(x_nb, y)
        return self

#Defining SEED as a constant parameter for random parameters
SEED = 42
#Setting seed
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)

#Splits the data for validation purposes
X_train, X_test, Y_train, Y_test = train_test_split(df_train["text"], df_train["target"],
                                                      test_size=0.2, random_state=SEED,
                                                      stratify=df_train["target"])

#Vectorizer operator
vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, 
                      strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)

#Application of the NBSVM model
def effectv(trial):
    C = trial.suggest_float(name="C", low=1e-3, high=1e3, log=True)
    max_iter = trial.suggest_discrete_uniform(name="max_iter", low=50, high=500, q=50)
    nbsvm = NBSVMClassifier(C=C, max_iter=max_iter)
    
    train_term_doc = vec.fit_transform(X_train)
    valid_term_doc = vec.transform(X_test)
    nbsvm.fit(train_term_doc, Y_train)
    
    prediction = nbsvm.predict(valid_term_doc)
    prediction[prediction>=0.5] = 1
    prediction[prediction<0.5] = 0
    
    acc_score = accuracy_score(Y_test, prediction)
    return acc_score

#Optuna hyperparameter tunning
optuna.logging.set_verbosity(0)
warnings.filterwarnings("ignore")

NUM_TRIALS = 100
study = optuna.create_study(direction="maximize")
study.optimize(effectv, n_trials=NUM_TRIALS, show_progress_bar=True)

print(f"Best Value: {study.best_trial.value}")
print(f"Best Params: {study.best_params}")

#Set the best parameters to a variable
kwargs = study.best_params

#Concatenates both dataframes to a single one
df_test["target"] = -1
df = pd.concat([df_train, df_test])

train = df[df['target']!=-1]
test = df[df['target']==-1]

#Kfold cross validation
NUM_SPLITS = 10
final_preds = np.zeros((len(test)))
kfold = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=SEED)
#Scores lists to create dataframe
accuracy_list = [] 
precision_list = []
recall_list = []
f1_list = []

for fold, (train_index, valid_index) in enumerate(kfold.split(train["text"], train["target"])):
    
    X_train = train.iloc[train_index, :].reset_index(drop=True)
    X_valid = train.iloc[valid_index, :].reset_index(drop=True)
    
    y_train = X_train['target']
    y_valid = X_valid['target']
    
    train_term_doc = vec.fit_transform(X_train["text"])
    valid_term_doc = vec.transform(X_valid["text"])
    test_term_doc = vec.transform(test["text"])
    
    # using best hyperparameters selected above
    nbsvm = NBSVMClassifier(**kwargs)
    nbsvm.fit(train_term_doc, y_train)
    
    valid_preds = nbsvm.predict(valid_term_doc)
    accuracy_list.append(accuracy_score(y_valid, valid_preds))
    precision_list.append(precision_score(y_valid, valid_preds))
    recall_list.append(recall_score(y_valid, valid_preds))
    f1_list.append(f1_score(y_valid, valid_preds))
    
    test_preds = nbsvm.predict(test_term_doc)
    final_preds += test_preds

#Creates a dataframe to save information about each fold 
folds_col=range(1, NUM_SPLITS+1)
folds_results = pd.DataFrame({'Folds':folds_col,
                            'Accuracy': accuracy_list,
                            'Precision': precision_list,
                            'Recall': recall_list,
                            'F1': f1_list})
folds_results.max() #max value found in each fold


#Cross Validation Scores for each fold in kfold
plt.figure(figsize=(10,5))
sns.lineplot(x='Folds', y='value', hue='variable', 
             data=pd.melt(folds_results, ['Folds']))
plt.ylabel('Scores')
plt.title('Cross Validation Scores', size=16, y=1)
plt.savefig('CrossValidScores.png')
plt.show()


#Final predictions to column
columns=['target']
final_pred = pd.DataFrame(columns=columns)
final_pred['target'] = final_preds/NUM_SPLITS
final_pred['target'] = final_pred["target"].apply(lambda x: 1 if x>=0.5 else 0)
final_pred.head(10)

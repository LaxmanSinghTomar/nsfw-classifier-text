#  Importing Libraries & Packages

import pandas as pd
import numpy as np

from varname.helpers import Wrapper

import os

# CountVectorizer will help calculate word counts
from sklearn.feature_extraction.text import CountVectorizer

# Import the string dictionary that we'll use to remove punctuation
import string

import nltk
if os.path.isdir("/home/laxman/nltk_data/"):
    pass
else:
    nltk.download('punkt')
    nltk.download('stopwords')

import pickle
from rich.console import Console

import warnings
warnings.filterwarnings("ignore")

console = Console()
train = pd.read_csv('data/raw/check.csv')
train[train['comment_text'].isna()]

train.dropna(inplace = True)
train[train['comment_text'].isna()]

nontoxic_tr = train[train['toxic'] == 0]
toxic_tr = train[train['toxic'] == 1]

cv = CountVectorizer(max_df=0.95, min_df=2,
                                     max_features=10000,
                                     stop_words='english')

final_cv = cv.fit_transform(train['comment_text'])

X_nontoxic = cv.transform(nontoxic_tr['comment_text'])
X_toxic = cv.transform(toxic_tr['comment_text'])

nontoxic_final_count_df = pd.DataFrame(X_nontoxic.toarray(), columns=cv.get_feature_names())
toxic_final_count_df = pd.DataFrame(X_toxic.toarray(), columns=cv.get_feature_names())

nontoxic_words = {}
toxic_words = {}

for k in cv.get_feature_names():
    nontoxic = nontoxic_final_count_df[k].sum()
    toxic = toxic_final_count_df[k].sum()
    
    nontoxic_words[k] = nontoxic/(nontoxic_tr.shape[0])
    toxic_words[k] = toxic/(toxic_tr.shape[0])
    
    
nontoxic_words_adj = {}
toxic_words_adj = {}

for key, value in nontoxic_words.items():
    nontoxic_words_adj[key] = nontoxic_words[key] - (toxic_words[key])
    
for key, value in toxic_words.items():
    toxic_words_adj[key] = toxic_words[key] - (nontoxic_words[key])


# saving toxic and nontoxic words
def save_dict(dict_name):
    file = open(f"data/intermediate/{dict_name.name}.pkl", "wb")
    pickle.dump(dict_name.value, file)
    console.print(f"{dict_name.name} Dumped!", style = "bold orange_red1")
    file.close()    

toxic_words = Wrapper(toxic_words_adj)
nontoxic_words = Wrapper(nontoxic_words_adj)

save_dict(toxic_words)
save_dict(nontoxic_words)

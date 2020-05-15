import pandas as pd
import numpy as np

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string
import json

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

import re
from collections import Counter


punctuations = string.punctuation
nlp = spacy.load('en_core_web_sm')
parser = English()


def remove_bracketed(song):
    text = re.sub("\[.*?\]", '', song)
    text = re.sub("\(.*?\)", '', text)
    text = re.sub("\{.*?\}", '', text)
    
    return text


def spacy_tokenizer(text, use_stopwords=True, custom_stopwords=set()):
    text = remove_bracketed(text)
    mytokens = parser(text)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else "-PRON-" for word in mytokens ]
    # mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    if use_stopwords:
        mytokens = [word for word in mytokens if word not in (stop_words | custom_stopwords) and word not in punctuations ]
    else:
        mytokens = [word for word in mytokens if word not in punctuations ]

    return mytokens


def tokenize_column(dataframe, column_name, use_stopwords=True, custom_stopwords=set()):
    df_ = dataframe.reset_index(drop=True)
    token_list = []
    for song in df_[column_name]:
        token_list.append(spacy_tokenizer(song, use_stopwords, custom_stopwords))
    
    return token_list




df = pd.read_json('lyrics_cleaned.json')

df.reset_index(drop=True,inplace=True)

df = df.drop(['character_count', 'word_count'], axis=1)

lyric_stopwords = set(['hey', 'baby', 'babe', 'woo', 'ha', 'like', 'oh', 'ooh', 'woah', 'yeah'])
all_stop_words = lyric_stopwords.union(STOP_WORDS)

top_artist = df.groupby(by='artist').count().sort_values(by='title',ascending=False).loc[:,'title']
artist_list = list(top_artist[:50].index)
working_df = df[df['artist'].isin(artist_list)]
song_is_artist = pd.get_dummies(working_df['artist'], prefix='is')

aggr_feat_imp_dict = {}

for artist in artist_list:
    is_artist = 'is_{}'.format(artist)
    y = song_is_artist.iloc[:,song_is_artist.columns.get_loc(is_artist)]

    X_train, X_test, y_train, y_test = train_test_split(working_df['lyrics'], 
                                                        y, 
                                                        test_size=.40)

    vectorizer = TfidfVectorizer(stop_words=all_stop_words, 
                                 ngram_range=(1,3),
    #                             , max_df=.8, 
    #                              min_df=.2, 
                                 max_features=10000
                                )
    vectorizer.fit(X_train)
    X_train_vec = vectorizer.transform(X_train)
    features = vectorizer.get_feature_names()

    imbrf = BalancedRandomForestClassifier(n_estimators=5000,
                                           max_features='auto', 
                                           sampling_strategy=0.5
                                          ).fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)

    y_pred = imbrf.predict(X_test_vec)

    for score, term in zip(imbrf.feature_importances_, features):
        if term not in aggr_feat_imp_dict:
            aggr_feat_imp_dict[term] = score
        else:
            aggr_feat_imp_dict[term] += score

with open('feat_ranks_dict.json', 'w') as fp:
    json.dump(aggr_feat_imp_dict, fp)
import numpy as np
import pandas as pd
import re
import spacy
from spacy.lang.en import English
punctuations = string.punctuation
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
parser = English()


def remove_bracketed(song):
    text = re.sub("\[.*?\]", '', song)
    text = re.sub("\(.*?\)", '', text)
    text = re.sub("\{.*?\}", '', text)
    
    return text


def spacy_tokenizer(text, use_stopwords=True, custom_stopwords=set()):
    text = remove_bracketed(text)
    mytokens = parser(text)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
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

def tf(term, document_tokens):
    term_occ = document_tokens.count(term)
    total_tokens = len(document_tokens)
    return term_occ / total_tokens



def get_tf(document_tokens):
    term_freqs = {}
    for token in document_tokens:
        if token not in term_freqs:
            term_freqs[token] = tf(token, document_tokens)
    return term_freqs
    

def get_idf_dict(corpus):
    occ_dict = {}
    for doc in corpus:
        for token in doc:
            if token not in occ_dict:
                occ_dict[token] = 1
            else:
                occ_dict[token] += 1
    return occ_dict

def get_doc_freq_dict(corpus):
    doc_occs = Counter([word for row in corpus for word in set(row)])
    doc_freq = {k: (v / float(len(docs))) for k, v in doc_occs.items()}
    return doc_freq


def vectorize_tokens(pandas_series):
    docs = pandas_series.to_numpy()
    doc_freq = get_doc_freq_dict(working_df['lyrics_tokens'])
    vocabulary = [k for k,v in doc_freq.items()]
    vectors = np.zeros((len(docs),len(vocabulary)))

    for i in range(len(docs)):
        for j in range(len(vocabulary)):
            term     = vocabulary[j]
            term_tf  = tf(term, docs[i])   # 0.0 if term not found in doc
            term_idf = np.log(1 + (1 / doc_freq[term])) # smooth formula
            vectors[i,j] = term_tf * term_idf
    return vectors, vocabulary


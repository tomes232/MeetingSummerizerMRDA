import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim import corpora, models
from os import listdir
import nltk
from pprint import pprint

nltk.download('wordnet')
np.random.seed(2018)


def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def read_data(transcript_dir='./ami-transcripts'):
    files = [f for f in listdir(transcript_dir)]
    meeting_dict = {}
    for filename in files[0:1]:
        f = open(transcript_dir+"/"+filename, "r")
        dialog = []
        for line in f:
            line = line.replace('\n', '')
            content_list = line.split('|')
            dialog.append({'speaker':'', 'dialog':preprocess(content_list[0]), 'da_tags':''})
        meeting_dict[filename[0:-1]] = dialog
    return meeting_dict

def lda_bag_of_words(meeting_dict, verbose=False):

    processed_doc = []
    if(verbose):
        print(str(count) + '\t' + key)
    for dialog in meeting_dict:
        if len(dialog['dialog']) != 0:
            processed_doc.append(dialog['dialog'])

    word_dict = gensim.corpora.Dictionary(processed_doc)

    if verbose:
        count = 0
        for k, v in word_dict.iteritems():
            print(k, v)
            count += 1
            if count > 10:
                break

    bow_corpus = [word_dict.doc2bow(doc) for doc in processed_doc]
    if(verbose):
        bow_doc_4 = bow_corpus[4]
        for i in range(len(bow_doc_4)):
            print("Word {} (\"{}\") appears {} time.".format(bow_doc_4[i][0],
                                                    word_dict[bow_doc_4[i][0]],
                                                    bow_doc_4[i][1]))
    return word_dict, bow_corpus

def td_idf(bow_corpus, verbose=False):
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    if verbose:
        for doc in corpus_tfidf:
            pprint(doc)
            break
    return corpus_tfidf



def bag_of_words(transcript):
    full_text = []
    for dialog in transcript:
        if len(dialog['dialog']) != 0:
            full_text.append(dialog['dialog'])
    word_dict = gensim.corpora.Dictionary(full_text)

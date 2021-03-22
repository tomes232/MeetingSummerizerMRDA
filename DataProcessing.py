import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from os import listdir
import nltk
nltk.download('wordnet')
np.random.seed(2018)


def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def read_data(transcript_dir='./train'):
    files = [f for f in listdir(transcript_dir)]
    meeting_dict = {}
    for filename in files[0:2]:
        f = open(transcript_dir+"/"+filename, "r")
        dialog = []
        for line in f:
            line = line.replace('\n', '')
            content_list = line.split('|')
            result = []
            for token in gensim.utils.simple_preprocess(content_list[1]):
                if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                    result.append(lemmatize_stemming(token))
            dialog.append({'speaker':content_list[0], 'dialog':result, 'da_tags':content_list[2:]})
        meeting_dict[filename[0:-4]] = dialog
    return meeting_dict


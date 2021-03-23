import pandas as pd
from DataProcessing import *
import pprint
import gensim

meeting_dict = read_data()  


count = 0
for key in meeting_dict:
    print("Meeting {} topic modeling:\n".format(key))
    dictionary, corpus_bow = lda_bag_of_words(meeting_dict[key])
    corpus_tfidf = td_idf(corpus_bow)

    print("LDA TD-IDF topic modeling with 10 topics")
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))
    print("\n")

    print("LDA Bag of Words topic modeling with 10 topics")
    lda_model = gensim.models.LdaMulticore(corpus_bow, num_topics=10, id2word=dictionary, passes=2, workers=2)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
    print("\n\n\n")


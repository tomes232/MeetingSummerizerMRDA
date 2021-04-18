import nltk
import statistics
import numpy as np
import gensim
from gensim.corpora.dictionary import Dictionary
import summarize
import operator

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def segment_tokens_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    text = text.replace(u'\xa0', u' ')
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def segment_tokens(lst, n):
    segments = []
    count = 0
    new_segment = []
    for item in lst:
        if(count < n):
            count += 1
            new_segment.append(item)
        else:
            count = 0
            segments.append(new_segment)
            new_segment = []
    segments.append(new_segment)
    return segments


def build_topic_prob_dict(lda, topic_id, words):
    topic_dict = {}
    for word in words:
        # topic_probs = lda.get_document_topics(gensim.corpora.dictionary.doc2bow(word))
        topic_probs = lda.get_document_topics(lda.id2word.doc2bow([word]))
        #print("topic_probs: ", topic_probs)
        # find topic id in the generated list
        topic_found = False
        for topic in topic_probs:
            if(topic[0] == topic_id):
               # print("FOUND TOPIC with id: ", topic_id)
                # we found the topic, add the probability to the set.
                topic_found = True
                topic_dict[word] = topic[1]
            
        if not topic_found:
            topic_dict[topic_id] = 0
    #print("len(topic_dict): ", len(topic_dict))
    return topic_dict


# text: document input, w: length of the pseudosentences, topic_prob_dict
# dict from a word to its set of (topic, prob) sets
def texttile(text, w, lda):
    # begin by tokenizing the text so it's easier to work with
    print("tokenizing")
    # text_tokens = nltk.word_tokenize(text)
    sentences = segment_tokens_sentences(text)
    #print(sentences)

    segments = []
    x = 0
    deleted_sentences = []
    for segment in sentences:
        text_tokens = []
        if len(nltk.word_tokenize(segment)) > 5:
            for token in gensim.utils.simple_preprocess(segment):
                if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                    text_tokens.append(token)

            segments.append(text_tokens)

        else:
            deleted_sentences.append(x)

        x += 1
        

    for index in sorted(deleted_sentences, reverse=True):
        del sentences[index]

    print("segmenting tokens")
    #print(segments)
    # segment the text
    #segments = segment_tokens(text_tokens, w)

    # compute the gap scores (Section 3.2 of guiding paper)

    print("computing gap scores")
    gap_scores = []
    for i in range(len(segments) - 1):
        # find the gap score for segments i and i + 1
        # find the topic intersection
        # TODO  look into getting the segments in bow format
        #print(segments[i])
        seg1_topics = lda.get_document_topics(lda.id2word.doc2bow(segments[i]))
        seg2_topics = lda.get_document_topics(lda.id2word.doc2bow(segments[i+1]))

        topic_id_set_1 = set()
        for topic in seg1_topics:
            topic_id_set_1.add(topic[0])

        topic_id_set_2 = set()
        for topic in seg2_topics:
            topic_id_set_2.add(topic[0])

        topic_id_set = topic_id_set_1.intersection(topic_id_set_2)
        #print("len(topic_id_set): ", len(topic_id_set))

        seg1_topic_prob_vec = []
        seg2_topic_prob_vec = []

        # top of pseudocode for Algo 1 in paper
        sim_score = 0.0
        for topic_id in topic_id_set:
            # for all words in segment 1, add the probabilites the word connects to the topic
            seg1_topic_dict = build_topic_prob_dict(lda, topic_id, segments[i])
            seg2_topic_dict = build_topic_prob_dict(lda, topic_id, segments[i+1])
            #print("seg1_topic_dict: ", seg1_topic_dict)

            #print("seg1_topic_dict.values(): ", seg1_topic_dict.values())
            avg_seg1_prob = statistics.mean(seg1_topic_dict.values())
            avg_seg2_prob = statistics.mean(seg2_topic_dict.values())

            # seg1_topic_prob_vec.append(avg_seg1_prob)
            # seg2_topic_prob_vec.append(avg_seg2_prob)
            #print("avg_seg1: ", avg_seg1_prob)
            #print("product = ", avg_seg1_prob * avg_seg2_prob)
            #print("sim_score before: ", sim_score)
            sim_score += avg_seg1_prob * avg_seg2_prob
            #print("sim_score after: ", sim_score)

        gap_scores.append(sim_score)
        # Find the cosine similarity of the vecs
        # gap_scores.append(np.dot(seg1_topic_prob_vec, seg2_topic_prob_vec))
    print("gap scores")
    for gap_score in gap_scores:
        print(gap_score)

    #print("finding peaks")
    # find the peaks
    peaks = []
    print("number of gap scores: ", len(gap_scores))
    for i in range(len(gap_scores)):
        if(i > 0 and i < (len(gap_scores) - 1)):
            if(gap_scores[i] > gap_scores[i-1] and gap_scores[i] > gap_scores[i+1]):
                #print("peak found at index: ", str(i))
                peaks.append(i)
        elif(i == 0):
            if(gap_scores[i] > gap_scores[i+1]):
                #print("peak found at index: ", str(i))
                peaks.append(i)
        elif(gap_scores[i] > gap_scores[i-1]):
            #print("peak found at index: ", str(i))
            peaks.append(i)

    print("computing depth scores")
    # compute depths between peaks
    depths = []
    print("number of peaks: ", len(peaks))
    for i in range(len(peaks) - 1):
        # check all the points between the peaks and find the lowest one.
        lowest_score = gap_scores[peaks[i]]
        lowest_point = 0
        for j in range(peaks[i] + 1, peaks[i + 1]):
            if gap_scores[j] < lowest_score:
                lowest_point = j
                lowest_score = gap_scores[j]
        depth_score = (peaks[i] - lowest_score) + (peaks[i + 1] - lowest_score)
        #print("local min at index: ", lowest_point)
        depths.append((lowest_point, depth_score))
    
    text_segments = []
    begin = 0
    for index, score in depths:
        text_segments.append(sentences[begin:index])
        begin = index
    text_segments.append(sentences[depths[-1][0]:])



    print("len(depth_scores): ", len(depths))
    print("partitioning")
    final_partition = []
    prev = 0
    for depth in depths:
        partition = []
        for i in range(prev, depth[0]):
            partition = partition + segments[i]
        prev = depth[0]
        final_partition.append(partition)
    
    if(depths[-1][0] != (len(gap_scores) - 1)):
        partition = []
        for i in range(prev, len(gap_scores)):
            partition = partition + segments[i]
        final_partition.append(partition)

    return final_partition, text_segments
    # # compute the depth scores
    # depth_scores = []
    # for i in range(1, len(gap_scores) - 1):
    #     depth_scores.append((gap_scores[i - 1] - gap_scores[i]) + (gap_scores[i + 1] - gap_scores[i]))

#def partition_document(text, partition):
#    for text 
def run_texttile(file):
    file = open(file, "r")
    print("reading in the file")
    text = file.read()
    print("Loading the model")
    lda = gensim.models.LdaModel.load('lda.model')
    print("Calling texttile")
    partitioning, text_segments = texttile(text, 20, lda)

    print("partitioning length: ", len(partitioning))
    titles = []
    for segment in partitioning:
        dictionary = Dictionary()
        sentence_bow = dictionary.doc2bow(segment, allow_update=True)
        topics = lda.get_document_topics(sentence_bow)
        topics.sort(key = operator.itemgetter(1))
        topic_term_list = lda.show_topic(topics[0][0], topn=1000)
        terms = []
        for term in topic_term_list:
            terms.append(term[0])
        titles.append(terms)

    return titles, text_segments


def main():
    print("Opening file")
    #file = open("file.txt", "r")
    file = open("ami-transcripts/ES2014c.transcript.txt", "r")
    print("reading in the file")
    text = file.read()
    print("Loading the model")
    lda = gensim.models.LdaModel.load('lda.model')
    print("Calling texttile")
    partitioning, text_segments = texttile(text, 20, lda)

    print("partitioning length: ", len(partitioning))
    titles = []
    for segment in partitioning:
        dictionary = Dictionary()
        sentence_bow = dictionary.doc2bow(segment, allow_update=True)
        topics = lda.get_document_topics(sentence_bow)
        topics.sort(key = operator.itemgetter(1))
        topic_term_list = lda.show_topic(topics[0][0], topn=1000)
        terms = []
        for term in topic_term_list:
            terms.append(term[0])
        titles.append(terms)


    print("Summarize:")
    print(len(text))
    print(len(summarize.summary(titles, text_segments)))
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    summary = summarize.summary(titles, text_segments)
    print(summary)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print(summarize.lex_summary_summary(summary))

    print("Score Summary:")
    sum_file = open("ami-summary/ES2014c.extsumm.txt", "r")
    text_sum = sum_file.read()
    print(text_sum)
    print(summarize.score(summary,text_sum))

    


if __name__ == "__main__":
    main()
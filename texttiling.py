import nltk
import statistics
import numpy as np
import gensim
from gensim.corpora.dictionary import Dictionary

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
        topic_probs = lda.get_term_topics(word)
        # find topic id in the generated list
        topic_found = False
        for topic in topic_probs:
            if(topic[0] == topic_id):
                # we found the topic, add the probability to the set.
                topic_found = True
                topic1_dict[topic_id] = topic[1]
            
        if not topic_found:
            topic_dict[topic_id] = 0
    
    return topic_dict


# text: document input, w: length of the pseudosentences, topic_prob_dict
# dict from a word to its set of (topic, prob) sets
def texttile(text, w, lda):
    # begin by tokenizing the text so it's easier to work with
    print("tokenizing")
    # text_tokens = nltk.word_tokenize(text)
    text_tokens = gensim.utils.simple_preprocess(text)

    print("segmenting tokens")
    # segment the text
    segments = segment_tokens(text_tokens, w)

    # compute the gap scores (Section 3.2 of guiding paper)

    print("computing gap scores")
    gap_scores = []
    for i in range(len(segments) - 1):
        # find the gap score for segments i and i + 1
        # find the topic intersection
        # TODO  look into getting the segments in bow format
        print(segments[i])
        seg1_topics = lda.get_document_topics(lda.id2word.doc2bow(segments[i]))
        seg2_topics = lda.get_document_topics(lda.id2word.doc2bow(segments[i+1]))
        
        topic_id_set = set()
        for topic in seg1_topics + seg2_topics:
            topic_id_set.add(topic[0])

        seg1_topic_prob_vec = []
        seg2_topic_prob_vec = []

        # top of pseudocode for Algo 1 in paper
        sim_score = 0
        for topic_id in topic_id_set:
            # for all words in segment 1, add the probabilites the word connects to the topic
            seg1_topic_dict = build_topic_prob_dict(lda, topic_id, segments[i])
            seg2_topic_dict = build_topic_prob_dict(lda, topic_id, segments[i+1])

            avg_seg1_prob = statistics.mean(seg1_topic_dict.values())
            avg_seg2_prob = statistics.mean(seg2_topic_dict.values())

            seg1_topic_prob_vec.append(avg_seg1_prob)
            seg2_topic_prob_vec.append(avg_seg2_prob)

            sim_score += avg_seg1_prob * avg_seg2_prob

        gap_scores.append(sim_score)
        # Find the cosine similarity of the vecs
        # gap_scores.append(np.dot(seg1_topic_prob_vec, seg2_topic_prob_vec))

    print("finding peaks")
    # find the peaks
    peaks = []
    for i in range(len(gap_scores)):
        if(i > 0 and i < (len(gap_scores) - 1)):
            if(gap_scores[i] > gap_scores[i-1] and gap_scores[i] > gap_scores[i+1]):
                peaks.append(i)
        elif(i == 0):
            if(gap_scores[i] > gap_scores[i+1]):
                peaks.append(i)
        elif(gap_scores[i] > gap_scores[i-1]):
            peaks.append(i)

    print("computing depth scores")
    # compute depths between peaks
    depths = []
    for i in range(len(peaks) - 1):
        # check all the points between the peaks and find the lowest one.
        lowest_score = gap_scores[peaks[i]]
        lowest_point = 0
        for j in range(peaks[i] + 1, peaks[i + 1]):
            if gap_scores[j] < lowest_point:
                lowest_point = j
                lowest_score = gap_scores[j]
        depth_score = (peaks[i] - lowest_score) + (peaks[i + 1] - lowest_score)
        depths.append((lowest_point, depth_score))


    print("partitioning")
    final_partition = []
    prev = 0
    for depth in depths:
        partition = []
        for i in range(prev, depth[0]):
            partition + segments[i]
        prev = depth[0]
        final_partition.append(partition)
    
    if(depths[-1][0] != (len(gap_scores) - 1)):
        partition = []
        for i in range(prev, len(gap_scores)):
            partition + segments[i]
        final_partition.append(partition)

    return final_partition
    # # compute the depth scores
    # depth_scores = []
    # for i in range(1, len(gap_scores) - 1):
    #     depth_scores.append((gap_scores[i - 1] - gap_scores[i]) + (gap_scores[i + 1] - gap_scores[i]))



def main():
    print("Opening file")
    file = open("file.txt", "r")
    print("reading in the file")
    text = file.read()
    print("Loading the model")
    lda = gensim.models.LdaModel.load('lda.model')
    print("Calling texttile")
    partitioning = texttile(text, 20, lda)

    for segment in partitioning:
        string = ""
        for word in segment:
            string = string + " " + word
        print(string)
        print("NEXT SEGMENT")


if __name__ == "__main__":
    main()
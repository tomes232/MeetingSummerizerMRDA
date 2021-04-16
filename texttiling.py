import nltk
import statistics
import numpy as np

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


def build_topic_prob_dict(lda, topic_id, segment_topics):
    topic_dict = {}
    for word in topics_1:
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
    text_tokens = nltk.word_tokenize(text)

    # segment the text
    segments = segment_tokens(text_tokens, w)

    # compute the gap scores (Section 3.2 of guiding paper)
    gap_scores = []
    for i in range(len(segments - 1)):
        # find the gap score for segments i and i + 1
        # find the topic intersection
        # TODO  look into getting the segments in bow format
        seg1_topics = lda.get_document_topics(lda.doc2bow(segments[i]))
        seg2_topics = lda.get_document_topics(lda.doc2bow(segments[i + 1]))
        
        topic_id_set = {} 
        for topic in seg1_topics + seg2_topics:
            topic_id_set.add(topic[0])

        seg1_topic_prob_vec = []
        seg2_topic_prob_vec = []

        # top of pseudocode for Algo 1 in paper
        sim_score = 0
        for topic_id in topic_id_set:
            # for all words in segment 1, add the probabilites the word connects to the topic
            seg1_topic_dict = build_topic_prob_dict(lda, topic_id, seg1_topic_dict)
            seg2_topic_dict = build_topic_prob_dict(lda, topic_id, seg2_topic_dict)

            avg_seg1_prob = statistics.mean(seg1_topic_dict.values())
            avg_seg2_prob = statistics.mean(seg2_topic_dict.values())

            seg1_topic_prob_vec.append(avg_seg1_prob)
            seg2_topic_prob_vec.append(avg_seg2_prob)

            sim_score += avg_seg1_prob * avg_seg2_prob

        gap_scores.append(sim_score)
        # Find the cosine similarity of the vecs
        # gap_scores.append(np.dot(seg1_topic_prob_vec, seg2_topic_prob_vec))

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

    
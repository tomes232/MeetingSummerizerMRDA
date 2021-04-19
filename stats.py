from os import listdir
from os.path import join
import summarize
import texttiling

import numpy as np

import pandas as pd

def get_transcripts_stats(transcript_dir, summary_dir):
    transcript_files = [f.split(".")[0] for f in listdir(transcript_dir)]
    #print("transcript files: ", transcript_files)
    summary_files = [f.split(".")[0] for f in listdir(summary_dir)]
    #print("summary files: ", summary_files)
    both_files = list(set(transcript_files) & set(summary_files))
    #print("both files: ", len(both_files))


    transcript_files = [join(transcript_dir,f) for f in listdir(transcript_dir) if f.split(".")[0] in both_files]
    #print("transcript files: ", len(transcript_files))

    summary_files = [join(summary_dir, f) for f in listdir(summary_dir) if f.split(".")[0] in both_files]
    #print("summary files: ", len(summary_files))

    l = []
    m = []
    h = []
    precision = []
    recall = []
    fmeasure = []
    both_files.sort()
    transcript_files.sort()
    summary_files.sort()
    for bf, tf, sf in zip(both_files, transcript_files, summary_files):
        titles, text_segments = texttiling.run_texttile(tf)
        summary = summarize.summary(titles, text_segments)

        sum_file = open(sf, "r")
        ami_sum = sum_file.read()

        tran_file = open(tf, "r")
        ami_tran = tran_file.read()

        rouge_stats = summarize.score(ami_sum, summary)
        l.append(len(texttiling.segment_tokens_sentences(ami_tran)))
        m.append(len(texttiling.segment_tokens_sentences(summary)))
        h.append(len(texttiling.segment_tokens_sentences(ami_sum)))
        precision.append(rouge_stats["rouge1"][0])
        recall.append(rouge_stats["rouge1"][1])
        fmeasure.append(rouge_stats["rouge1"][2])


        #tran_dict[bf] = {"L": len(tex.tiling.segment_token_sentences(ami_tran)),
        #                "M":len(tex.tiling.segment_token_sentences(summary)) ,
        #                "H":len(tex.tiling.segment_token_sentences(ami_sum)),
        #                "precision":rouge_stats["'rouge1'"][0],
        #                "recall":"precision":rouge_stats["'rouge1'"][1],
        #                "fmeasure":"precision":rouge_stats["'rouge1'"][2]}

    data = {"transcipt": both_files, "L": l, "M": m, "H":h, "Precision": precision, "Recall":recall, "Fmeasure":fmeasure}
    df = pd.DataFrame.from_dict(data)

    df.loc['mean'] = df.mean()

    return df

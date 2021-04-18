from textteaser.textteaser import TextTeaser

import sumy

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from sumy.summarizers.luhn import LuhnSummarizer

from rouge_score import rouge_scorer


def summary(titles, segments):
    #print(type(segments))
    summary_list = []
    tt = TextTeaser()
    for segment, title in zip(segments, titles):
        summary = tt.summarize(" ".join(title), " ".join(segment))
        summary_list.append(summary[0])
    #print(summary_list)

    return ' '.join(summary_list)


def lex_summary(segments):
    summary_list = []
    #print(len(segments))
    for segment in segments:
        parser = PlaintextParser.from_string(document,Tokenizer("english"))
        # Using LexRank
        summarizer = LexRankSummarizer()
        #Summarize the document with 1 sentences
        summary = summarizer(parser.document, 1)
        summary_list.append(summary[0])

    return summary_list

def lex_summary_summary(segment):
    parser = PlaintextParser.from_string(segment,Tokenizer("english"))
    # Using LexRank
    summarizer = LexRankSummarizer()
    #Summarize the document with 1 sentences
    summary = summarizer(parser.document, 15)
    return summary


def luhn_summary(segments):
    summary_list = []
    #print(len(segments))
    for segment in segments:
        parser = PlaintextParser.from_string(document,Tokenizer("english"))
        # Using LexRank
        summarizer_luhn = LuhnSummarizer()
        summary =summarizer_luhn(parser.document,2)
        #Summarize the document with 1 sentences
        summary_list.append(summary[0])

    return summary_list

def score(generated_summary, actual_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(generated_summary,actual_summary)
    return scores


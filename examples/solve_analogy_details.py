# -*- coding: utf-8 -*-

"""
solve_analogy_details.py
~~~~~~~~~~~~~~~~~~~~~~~

- This script tests an input embedding model with Google analogy questions
- and returns how model predicted on each question,
- using functionalities of kudkudak/word-embeddings-benchmarks:

    * GitHub: https://github.com/kudkudak/word-embeddings-benchmarks
    * Reference:
        How to evaluate word embeddings? On importance of data efficiency and simple supervised tasks
        (Stanisław Jastrzebski, Damian Leśniak, Wojciech Marian Czarnecki)
        https://arxiv.org/abs/1702.02170

    Usage:
        $ python3 solve_analogy_details.py --file <embedding>
        $ python3 solve_analogy_details.py --file <embedding> | tee -a "<log-filename>"

Yejin Cho (scarletcho@gmail.com)
Last updated: 2017-09-18
"""

import os
import logging
from optparse import OptionParser
from web.datasets.analogy import fetch_google_analogy
from web.embeddings import fetch_SG_GoogleNews, fetch_GloVe, load_embedding
from web.datasets.utils import _get_dataset_dir

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="Path to the file with embedding. If relative will load from data directory.",
                  default=None)

parser.add_option("-p", "--format", dest="format",
                  help="Format of the embedding, possible values are: word2vec, word2vec_bin, dict and glove.",
                  default=None)

parser.add_option("-o", "--output", dest="output",
                  help="Path where to save results.",
                  default=None)

parser.add_option("-c", "--clean_words", dest="clean_words",
                  help="Clean_words argument passed to load_embedding function. If set to True will remove"
                       "most of the non-alphanumeric characters, which should speed up evaluation.",
                  default=False)

parser.add_option("--case", dest="case",
                  help="Lowercase or uppercase vocabulary in the given embedding",
                  default='lower')

parser.add_option("-v", "--verbose", dest="verbose",
                  help="Whether to print questions, answers, and model predictions",
                  default=False)

def append2txt(mystrvar, fname):
    with open(fname, 'a') as out:
        out.write('{}\n'.format(mystrvar))


# Option & argument parsing
(options, args) = parser.parse_args()
fname = options.filename
case = options.case
verbose = options.verbose

# Restate input embedding (model) filename
print('Model filename: ' + fname)


if not fname:
    # If input embedding is not given as argument, fetch pre-trained GloVe online
    w = fetch_GloVe(corpus="wiki-6B", dim=300)
else:
    # Access input embedding model provided as argument
    if not os.path.isabs(fname):
        fname = os.path.join(_get_dataset_dir(), fname)

    format = options.format

    if not format:
        _, ext = os.path.splitext(fname)
        if ext == ".bin":
            format = "word2vec_bin"
        elif ext == ".txt":
            format = "glove"
        elif ext == ".pkl":
            format = "dict"

    assert format in ['word2vec_bin', 'word2vec', 'glove', 'bin'], "Unrecognized format"

    load_kwargs = {}
    if format == "glove":
        vocab_size = sum(1 for line in open(fname))
        dim = len(next(open(fname)).split()) - 1
        load_kwargs = {"vocab_size": vocab_size, "dim": dim}

    # Load embedding
    w = load_embedding(fname, format=format, normalize=True, lower=False,
                        clean_words=options.clean_words, load_kwargs=load_kwargs)


# Fetch analogy dataset
data = fetch_google_analogy()
os.system('touch {semantic_,syntactic_,}pred_list.txt')

# Counts
syn_missing_cnt = 0
sem_missing_cnt = 0
syn_cnt = 0
sem_cnt = 0

score = 0
syn_score = 0
sem_score = 0
syn_score_sum = 0
sem_score_sum = 0

# Begin solving analogy questions
for id in range(0, len(data.X)):
    w1, w2, w3 = data.X[id][0], data.X[id][1], data.X[id][2]
    if data.category_high_level[id] == 'syntactic':
        syn_cnt += 1
    else:
        sem_cnt += 1

    try:
        if case == 'upper':
            w1, w2, w3 = w1.upper(), w2.upper(), w3.upper()

        # Get nearest neighbors as model prediction
        y_pred = w.nearest_neighbors(w[w2] - w[w1] + w[w3], exclude=[w1, w2, w3])[0].lower()

        if y_pred == data.y[id]:
            score = 1
            if data.category_high_level[id] == 'syntactic':
                syn_score = 1
            else:
                sem_score = 1
        else:
            score = 0
            if data.category_high_level[id] == 'syntactic':
                syn_score = 0
            else:
                sem_score = 0

        syn_score_sum += syn_score
        sem_score_sum += sem_score

        # Create a line of analogy question, answer, model prediction, and score gained (1: correct, 0: incorrect)
        line = "{}  {}  {}  {}  {}  {}".format(w1.lower(), w2.lower(), w3.lower(), data.y[id], y_pred, str(score))

        if verbose:
            print("Question: {} is to {} as {} is to ?".format(w1, w2, w3))
            print("Answer: " + data.y[id])
            print("Predicted: " + " ".join(y_pred))
        
        append2txt(line, data.category_high_level[id] + '_pred_list.txt')
        append2txt(line, 'pred_list.txt')


    except KeyError:
    # If any of query words is not found in vocabulary, add missing count
        print('Missing word in analogy question')
        if data.category_high_level[id] == 'syntactic':
            syn_missing_cnt += 1
        else:
            sem_missing_cnt += 1

    if id % 100 == 0:
        print('Item #: ' + str(id) + ' / ' + str(len(data.X)))

score_sum = syn_score_sum + sem_score_sum
missing_cnt = syn_missing_cnt + sem_missing_cnt
questioned_cnt = len(data.X) - missing_cnt
syn_questioned_cnt = syn_cnt - syn_missing_cnt
sem_questioned_cnt = sem_cnt - sem_missing_cnt


# Print results from analogy test
print('================================================')
print('>> General statistics')
print('Missing questions: {} / {}'.format(missing_cnt, len(data.X)))
print('Correct answers: {} / {}'.format(score_sum, questioned_cnt))
print('Accuracy: {0:.2f} %'.format(float(score_sum)/float(questioned_cnt)*100))
print('================================================')
print('>> Syntactic questions')
print('Missing questions: {} / {}'.format(syn_missing_cnt, syn_cnt))
print('Correct answers: {} / {}'.format(syn_score_sum, syn_questioned_cnt))
print('Accuracy: {0:.2f} %'.format(float(syn_score_sum)/float(syn_questioned_cnt)*100))
print('================================================')
print('>> Semantics questions')
print('Missing questions: {} / {}'.format(sem_missing_cnt, sem_cnt))
print('Correct answers: {} / {}'.format(sem_score_sum, sem_questioned_cnt))
print('Accuracy: {0:.2f} %'.format(float(sem_score_sum)/float(sem_questioned_cnt)*100))
print('================================================')


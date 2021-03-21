import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import pickle
import sys
import os

sys.path.append(os.getcwd())

def preprocess(raw_text):
    # keep only words
    letters_only_text = re.sub("[^a-zA-ZА-Яа-я]", " ", raw_text)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("russian"))
    meaningful_words = [w for w in words if w not in stopword_set]

    # stemmed words
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in meaningful_words]

    # join the cleaned words in a list
    cleaned_word_list = " ".join(stemmed_words)

    return cleaned_word_list


def clean(comment):
    """prepares row data to use in model
    return: prepared data"""

    comment = preprocess(comment)
    tfidf = pickle.load(open('utils/tfidf', 'rb'))
    comment = tfidf.transform([comment]).astype(dtype=np.float32)

    return comment



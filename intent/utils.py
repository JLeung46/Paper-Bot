import numpy as np
import pickle
import csv
import re
import requests

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def tfidf_features(X_train, X_test, vectorizer_path=None):
    """Performs TF-IDF transformation and dumps the model."""

    # Train a vectorizer on X_train data.
    # Transform X_train and X_test data.

    # Pickle the trained vectorizer to 'vectorizer_path'
    # Don't forget to open the file in writing bytes mode.

    tfidf_vectorizer = TfidfVectorizer()
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    if vectorizer_path not None:
        with open(vectorizer_path, 'wb') as fin:
            pickle.dump(tfidf_vectorizer, fin)
    return X_train, X_test


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def unpickle_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENT_RECOGNIZER = os.path.join(BASE_DIR, 'intent/intent_classifier.pkl')
TFIDF_VECTORIZER = os.path.join(BASE_DIR, 'intent/tfidf_vectorizer.pkl')

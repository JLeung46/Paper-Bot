"""
Creates a DialogueManager object that loads the intent model and chatbot model.
The object additionally reads user input and returns a response based on the predicted intent.
"""

import sys
import os
import numpy as np
import settings

INTENT_PATH = settings.INTENT_RECOGNIZER
TFIDF_PATH = settings.TFIDF_VECTORIZER
CHATBOT_PATH = settings.BASE_DIR

sys.path.append(os.path.join(settings.BASE_DIR, 'intent'))

from chatbot import chatbot
from chatbot import text_data
from text_parser import TextParser
from utils import unpickle_file, text_prepare


class DialogueManager:
    """
    This class loads both the intent model and chatbot model
    and predicts the intent of the user input.
    """
    def __init__(self):
        """
        The constructor for DialogueManager class.
        """
        print("Loading resources...")

        self.intent_recognizer = unpickle_file(INTENT_PATH)
        self.tfidf_vectorizer = unpickle_file(TFIDF_PATH)
        self.text_parser = TextParser()

    def create_bot(self):
        """
        Creates a Chatbot object.
        """
        vocab = text_data.TextData()
        vocab.load_data()
        self.chatbot = chatbot.Chatbot(text_data=vocab)
        self.chatbot.load_model()

    def get_answer(self, text):
        """
        Pre-processes the input from user and predicts the intent.

        Parameters:
            text (String): The user's input.

        Returns:
            response (String): A response from the chatbot.
        """

        prepared_text = text_prepare(text)
        features = self.tfidf_vectorizer.transform(np.array([prepared_text]))
        intent = self.intent_recognizer.predict(features)

        if intent == 'dialogue':
            response = self.chatbot.evaluate(text)
            return response

        response = self.text_parser.get_response(text)
        return response

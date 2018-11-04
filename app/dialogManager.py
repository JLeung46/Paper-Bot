import sys
import os
import settings

intent_path = settings.INTENT_RECOGNIZER
tfidf_path = settings.TFIDF_VECTORIZER
chatbotPath = settings.BASE_DIR

sys.path.append(os.path.join(settings.BASE_DIR, 'intent'))

from chatbot import chatbot
from textParser import TextParser
from utils import *


class DialogueManager(object):
    def __init__(self):
        print("Loading resources...")

        self.intent_recognizer = unpickle_file(intent_path)
        self.tfidf_vectorizer = unpickle_file(tfidf_path)
        self.textParser = TextParser()

    def createBot(self):

        self.chatBot = chatbot.Chatbot()
        self.chatBot.main(['--modelTag', 'server', '--test',
                          'daemon', '--rootDir', chatbotPath])

    def getAnswer(self, text):

        prepared_text = text_prepare(text)
        features = self.tfidf_vectorizer.transform(np.array([prepared_text]))
        intent = self.intent_recognizer.predict(features)

        if intent == 'dialogue':
            response = self.chatBot.daemonPredict(text)
            return response

        else:
            response = self.textParser.getResponse(text)
            return response

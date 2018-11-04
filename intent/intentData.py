import os
import numpy as np
import pandas as pd
from newsData import NewsData
from utils import *


class IntentData:

    def __init__(self):
        self.dialogueData = None
        self.newsData = None

        self.BASE_DIR = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')

        self.fullDataX = None # Dataset containing both news and dialogue data
        self.fullDataY = None # Contains both 'news' and 'dialogue' labels

        self.sample_size = 800  # Number of samples for dialogue data
        self.state = 0  # random state

    def create_dataset(self):
        """
        Merge news and dialog dataset
        Sort training data and labels
        """
        self.fullDataX = np.concatenate([self.dialogueData['text'].values,
                                        self.newsData['text'].values])
        self.fullDataY = ['dialogue'] * self.dialogueData.shape[0] + \
            ['news'] * self.newsData.shape[0]

    def loadData(self):
        """
        Load news and dialogue data
        """
        news = NewsData(self.DATA_DIR)
        self.newsData = news.get_newsDataset()
        self.dialogueData = pd.read_csv(self.DATA_DIR + os.sep + 'dialogue' +
                                        os.sep + 'dialogues.tsv', sep='\t')
        .sample(self.sample_size, random_state=self.state)

    def cleanData(self):
        """
        Perform text cleaning
        """
        self.newsData['text'] = self.newsData['text']
        .map(lambda x: text_prepare(x))
        self.dialogueData['text'] = self.dialogueData['text']
        .map(lambda x: text_prepare(x))

    def prepareData(self):
        """
        Prepare dataset for training and testing
        """
        self.loadData()
        self.cleanData()
        self.create_dataset()

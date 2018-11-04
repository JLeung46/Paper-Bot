import os
import csv
import json
import pandas as pd


class NewsData:

    def __init__(self, dirName):
        """
        Args:
            dirName (string): directory where data is located
        """
        self.dataset = []
        self.samples = []
        self.categories = []
        self.publishers = []

        self.categories = self.load_file(os.path.join(
            dirName, "news/newsCategories.csv"))
        self.publishers = self.load_file(os.path.join(
            dirName, "news/newsPublishers.csv"))
        self.countries = self.load_file(os.path.join(
            dirName, "news/newsCountries.csv"))
        self.samples = self.load_samples(os.path.join(
            dirName, "news/newsSamples.json"))

    def load_file(self, filename):
        """
        Args:
            fileName (str): file to load
        Return:
            data (list): data from file
        """

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = [line[0] for line in reader]
        return data

    def load_samples(self, filename):
        """
        Args:
            fileName (str): file to load
        Return:
            samples (dict): sample sentences used to generate the dataset
        """
        with open(filename) as f:
            samples = json.load(f)
        return samples

    def generate_dataset(self):
        """
        Fills sample sentences with possible options from news api
        Convert to dataframe and add label
        """
        for k, v in self.samples['samples'].items():

            if k == 'categories':
                for phrase in v:
                    for category in self.categories:
                        self.dataset.append(phrase.replace(
                            '<fill>', category[0]))

            elif k == 'countries':
                for phrase in v:
                    for country in self.countries:
                        self.dataset.append(phrase.replace(
                            '<fill>', country[0]))

            elif k == 'publishers':
                for phrase in v:
                    for publisher in self.publishers:
                        self.dataset.append(phrase.replace(
                            '<fill>', publisher[0]))

        self.dataset = pd.DataFrame({'text': self.dataset})
        self.dataset['tag'] = 'news'

    def get_newsDataset(self):
        self.generate_dataset()
        return self.dataset

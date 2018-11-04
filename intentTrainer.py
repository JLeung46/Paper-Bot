import sys
import os
from sklearn.model_selection import train_test_split

sys.path.append("intent")


from intentData import IntentData
from intentModel import IntentModel
from utils import *


class Trainer:

    def __init__(self):

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X_train_tfidf = None
        self.X_test_tfidf = None

        self.model = None

        self.BASE_DIR = os.getcwd()
        self.modelDir = os.path.join(self.BASE_DIR, 'intent')
        self.tfidfDir = os.path.join(self.BASE_DIR, 'intent')

        self.test_size = 0.1

    def mainTrain(self):
        """
        Load data, split into train/test,
        featurize using tfidf, train model, save it.
        """
        intentDataset = IntentData()
        intentDataset.prepareData()
        print('***********************')
        print('Intent Dataset Loaded!')
        print()

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(intentDataset.fullDataX, intentDataset.fullDataY,
                             test_size=self.test_size)

        self.X_train_tfidf, self.X_test_tfidf = tfidf_features(
            X_train=self.X_train, X_test=self.X_test,
            vectorizer_path=self.tfidfDir + os.sep + 'tfidf_vectorizer.pkl')

        print('Training Intent Model...')
        print()
        self.model = IntentModel()
        self.model.train(self.X_train_tfidf, self.y_train)

        preds = self.model.predict(self.X_test_tfidf)
        acc = self.model.get_accuracy(self.y_test, preds)

        pickle.dump(self.model, open(self.modelDir + os.sep
                    + 'intent_classifier.pkl', 'wb'))

        print()
        print('Model Saved!')


if __name__ == '__main__':
    modelTrainer = Trainer()
    modelTrainer.mainTrain()

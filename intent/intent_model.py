from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class IntentModel:
    def __init__(self):

        self.clf = LogisticRegression(penalty='l2', C=1, random_state=0)

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def get_accuracy(self, y_test, y_pred):
        acc = accuracy_score(y_test, y_pred)
        print('Test accuracy = {}'.format(acc))
        return acc

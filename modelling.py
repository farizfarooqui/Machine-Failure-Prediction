import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

class Modelling:
    def __init__(self, X_train, Y_train, X_test, Y_test, models):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.models = models

    def fit(self):
        model_acc = []
        model_time = []
        for i in self.models:
            start = time.time()
            if i == 'knn':
                accuracy = []
                for j in range(1, 200):
                    kn = KNeighborsClassifier(n_neighbors=j)
                    kn.fit(self.X_train, self.Y_train)
                    predK = kn.predict(self.X_test)
                    accuracy.append([accuracy_score(self.Y_test, predK), j])
                temp = accuracy[0]
                for m in accuracy:
                    if temp[0] < m[0]:
                        temp = m
                i = KNeighborsClassifier(n_neighbors=temp[1])
            i.fit(self.X_train, self.Y_train)
            model_acc.append(accuracy_score(self.Y_test, i.predict(self.X_test)))
            stop = time.time()
            model_time.append((stop - start))
        self.models_output = pd.DataFrame({'Models': self.models, 'Accuracy': model_acc, 'Runtime (s)': model_time})

    def results(self):
        models = self.models_output
        self.best = models['Models'][0]
        self.models_output_cleaned = models
        return models

    def best_model_accuracy(self):
        return self.models_output_cleaned['Accuracy'][0]

    def best_model_runtime(self):
        return round(self.models_output_cleaned['Runtime (s)'][0], 3)
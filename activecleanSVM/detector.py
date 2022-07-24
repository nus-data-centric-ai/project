import numpy as np
from sklearn.linear_model import SGDClassifier

class Detector:
    def __init__(self, clf):
        self.clf = clf

    '''remove dirty samples if the probability of dirty is more than a threshold'''
    def ec_filter(self, dirtyex, full_data, t=0.90):
        if self.clf != None:
            pred = self.clf.predict_proba(full_data[dirtyex, :])
            return [j for i, j in enumerate(dirtyex) if pred[i][0] < t]

        print("All data has been cleaned")
        return dirtyex


    '''get the probability of an example being dirty'''
    def get_error_prob(self, dirtyex, full_data):
        if self.clf != None:
            pred = self.clf.predict_proba(full_data[dirtyex, :])
            return [pred[i][0] for i, j in enumerate(dirtyex)]
        else:
            return None

    def update_classifier(self, total_labels, full_data):
        # (indice, True if indice is clean_indice) in total_labels
        indices = [i[0] for i in total_labels]
        # 1 if clean, 0 if dirty
        labels = [int(i[1]) for i in total_labels]
        if np.sum(labels) < len(labels): # not all data is clean
            clf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=200, fit_intercept=True)
            clf.fit(full_data[indices, :], labels)
            self.clf = clf
        else:
            self.clf = None


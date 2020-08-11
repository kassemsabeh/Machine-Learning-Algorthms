import numpy as np

class MeanNormalizer():
    """Normalizes column in the features X by subtracting the mean and dividing by the standard deviation (std)"""

    def __init__(self):
        print("Initialized a Mean Normalizer Model...")
    
    def fit(self, X):
        """Fit on the training dataset X

        Args:
            X (numpy array): Features matrix of training dataset.
        """

        self.means = []
        self.std = []

        for i in range(X.shape[1]):
            self.means.append(np.mean(X[:, i]))
            self.std.append(np.std(X[:, i]))
        
    
    def transform(self, X):
        """Transform dataset to normalized feature matrix"""
        X_ = np.zeros((X.shape))
        for i in range(len(self.means)):
            X_[:, i] = (X[:, i] - self.means[i]) / self.std[i]
        
        return X_

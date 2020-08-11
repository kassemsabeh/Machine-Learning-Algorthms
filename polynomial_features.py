import numpy as np

class PolynomialFeatures():
    def __init__(self, n_dimentions):
        self.dim = n_dimentions
        print(f"Initialized Polynomial Features with {self.dim} dimentions")
    
    def transform(self, X):
        X_pol = X
        for i in range(X_pol.shape[1]):
            for j in range (1, self.dim):
                feature = (X[:, i]) ** j
                X_pol = np.c_[X_pol, feature]
        num = X_pol.shape[1]
        for i in range(num - 1):
            for j in range (i+1, num):
                feature = X_pol[:, i].reshape(X_pol.shape[0], 1) * X_pol[:, j].reshape(X_pol.shape[0], 1)
                X_pol = np.c_[X_pol, feature]
        print(f"Returned Vector with {X_pol.shape[1]} features")
        return X_pol
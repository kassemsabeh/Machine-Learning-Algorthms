import numpy as np

class LogisticRegression():
    def __init__(self):
        print("Initialized Logistic Regression Model..")
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        
        return (- y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + (self.lam / 2 * y.size) * (np.sum(self.theta * self.theta))
    
    def fit(self, X, y, num_iterations=1000, alpha=0.01, lam=100):
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        self.lam = lam
        self.coef_ = np.zeros(X.shape[1] - 1)
        self.intercept_ = 0
        self.theta = np.zeros(X.shape[1])
        self.iterations = num_iterations
        self.losses_ = []
        for _ in range(num_iterations):
            z = X.dot(self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            #reg = 1 - ((alpha * self.lam) / y.size)
            self.theta = self.theta - alpha * gradient
            self.losses_.append(self.__loss(h, y))
        print(self.__loss(h, y))
    
    def predict_prob(self, vector):
        ones = np.ones((vector.shape[0], 1))
        vector = np.concatenate((ones, vector), axis=1)
        return self.__sigmoid(np.dot(vector, self.theta))
    
    def predict(self, vector, threshold=0.5):
        return self.predict_prob(vector) >= threshold


class LinearRegression():
    def __init__(self):
        print("Initialized Linear Model")
    
    def __add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.c_[ones, X]
    
    def __loss(self, h, y):
        reg = (self.lam / (2 * y.shape[0])) * (self.theta[1:] ** 2).sum()
        return (1 / (2 * y.shape[0])) * ((h - y) ** 2).sum() + reg
    
    def fit(self, X, y, alpha=0.001, lam=1, epochs=1000):
        X = self.__add_bias(X)
        self.lam = lam
        self.theta = np.zeros((X.shape[1], 1))
        self.epochs = epochs
        self.losses=[]
        gradient = np.zeros((self.theta.shape))
        
        for _ in range(epochs):
            h = X.dot(self.theta)
            gradient[0] = X[:, 0].T.dot(h - y) / y.shape[0]
            gradient[1:] = (X[:, 1:].T.dot(h - y) / y.shape[0]) + (self.lam / y.shape[0]) * self.theta[1:]
            #gradient = X.T.dot(h - y) / y.shape[0]
            self.theta = self.theta - alpha * gradient
            self.losses.append(self.__loss(h, y))
        print(self.__loss(h, y))
    
    def predict(self, vector):
        vector = self.__add_bias(vector)
        return vector.dot(self.theta)
    
    def score(self, X, y):
        pred = self.predict(X)
        return self.__loss(pred, y)
        
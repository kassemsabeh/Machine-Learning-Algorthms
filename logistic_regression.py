import numpy as np

class GradientDescentClassifier():

    def __init__(self):
        print("Initialized Gradient Descent model..")
    
    def compute_loss(self, X, y):
        h = self.sigmoid(X.dot(self.theta))
        m = len(y)
        return (-1 / m) * np.sum((y * np.log(h)) + (1 - y) * np.log(1 - h))


    def sigmoid(self, X):
        return 1 / (1 + np.exp( - X ))   

    def fit(self, X, y, alpha=0.01, iterations=100):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_b = y.reshape(len(y), 1)
        m = len(y)
        
        self.theta = np.zeros((X_b.shape[1], 1))
        self.losses = []
        self.iterations = iterations

        for _ in range(iterations):
            h = self.sigmoid(X_b.dot(self.theta))
            gradients = (1 / m) * X_b.T.dot(h - y_b)
            self.theta = self.theta - alpha * gradients
            self.losses.append(self.compute_loss(X_b, y_b))
        print("Successfully fitted to training data")
        print(f"loss: {self.compute_loss(X_b, y_b)}")
    
    def predict_prob(self, vector):
        ones = np.ones((vector.shape[0], 1))
        vector = np.concatenate((ones, vector), axis=1)
        return self.sigmoid(np.dot(vector, self.theta))
    
    
    def predict(self, vector, threshold=0.5):
        return self.predict_prob(vector) >= threshold

class LogisticRegression():
    def __init__(self):
        print("Initialized Logistic Regression Model..")
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (- y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y, num_iterations=1000, alpha=0.01):
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        self.theta = np.zeros(X.shape[1])
        self.iterations = num_iterations
        self.losses_ = []
        for _ in range(num_iterations):
            z = X.dot(self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta = self.theta - alpha * gradient
            self.losses_.append(self.__loss(h, y))
        print(self.__loss(h, y))
    
    def predict_prob(self, vector):
        ones = np.ones((vector.shape[0], 1))
        vector = np.concatenate((ones, vector), axis=1)
        return self.__sigmoid(np.dot(vector, self.theta))
    
    def predict(self, vector, threshold=0.5):
        return self.predict_prob(vector) >= threshold
        
        
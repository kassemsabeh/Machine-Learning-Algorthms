import numpy as np

class NeuralNetwork():
    def __init__(self,):
        print("Initialized Neural Network...")
    
    def __loss(self, y):
        #reg = (self.lam / (2 * y.shape[0])) * (((self.theta1[:, 1:-1]) ** 2).sum() + ((self.theta2[:, 1:-1]) ** 2).sum())
        return (- y * np.log(self.A3) - (1 - y) * np.log(1 - self.A3)).sum() / (y.shape[0])
    def __sigmoid(self, z):
         return 1 / (1 + np.exp(-z))
    
    def __add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.c_[ones, X]
        
    def fit(self, X, y, alpha=0.0001, epochs=40):
        self.theta1 = np.random.uniform(low=-0.12, high=0.12, size=(25, 401))
        self.theta2 = np.random.uniform(low=-0.12, high=0.12, size=(10, 26))
        self.losses = []
        self.epochs = epochs
        for _ in range (epochs):
            #print(f"Epoch {epoch}:")
            #1-forward propagation
            self.A1 = self.__add_bias(X)
            z2 = self.theta1.dot(self.A1.T)
            self.A2 = self.__sigmoid(z2).T

            self.A2 = self.__add_bias(self.A2)
            z3 = self.theta2.dot(self.A2.T)
            self.A3 = self.__sigmoid(z3).T

            #2 - Add delta
            delta3 = self.A3 - y
            delta2 = delta3.dot(self.theta2) * (self.A2 * (1 - self.A2))

            #3 - Accumalate delta's
            Delta2 = delta3.T.dot(self.A2)
            Delta1 = delta2[:, 1:].T.dot(self.A1)

            #4 - Calculate gradients
            gradient1 = Delta1.sum() / X.shape[0]
            gradient2 = Delta2.sum() / X.shape[0]

            #5 - update gradients
            self.theta1 = self.theta1 - alpha * gradient1
            self.theta2 = self.theta2 - alpha * gradient2
    
            #6 - print loss
            #print(loss(y_encoded))
            self.losses.append(self.__loss(y))
    
    def score(self, X):
        #self.theta1 = np.zeros((layer_1, X.shape[1]))
        #self.alpha = alpha
        #self.lam = lam
        #Compute first layer
        self.A1 = self.__add_bias(X)
        z2 = self.theta1.dot(self.A1.T)
        self.A2 = self.__sigmoid(z2).T
        
        #Compute last layer
        self.A2 = self.__add_bias(self.A2)
        z3 = self.theta2.dot(self.A2.T)
        self.A3 = self.__sigmoid(z3).T
        #print(self.__loss(y))
        return self.A3
    
    #Dummy function
    def predict(self, int):
        value = self.A3[int].argmax() + 1
        if(value == 10):
            return 0
        else: return value
        
        
        
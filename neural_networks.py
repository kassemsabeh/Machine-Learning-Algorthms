class NeuralNetwork():
    def __init__(self, theta1, theta2):
        print("Initialized Neural Network...")
        self.theta1 = theta1
        self.theta2 = theta2
    
    def __sigmoid(self, z):
         return 1 / (1 + np.exp(-z))
    
    def __add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.c_[ones, X]
        
    def fit(self, X, y):
        #self.theta1 = np.zeros((layer_1, X.shape[1]))
        
        #Compute first layer
        self.A1 = self.__add_bias(X)
        z2 = self.theta1.dot(self.A1.T)
        self.A2 = sigmoid(z2).T
        
        #Compute last layer
        self.A2 = self.__add_bias(self.A2)
        z3 = self.theta2.dot(A2.T)
        self.A3 = sigmoid(z3).T
        return self.A3
    
    #Dummy function
    def predict(self, int):
        value = self.A3[int].argmax() + 1
        if(value == 10):
            return 0
        else: return value
        
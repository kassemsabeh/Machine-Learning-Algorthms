import numpy as np
import matplotlib.pyplot as plt

class GradientDescent():

    """Gradient Descent is the process of minimizing a function by following the gradients of the cost function. 
    You can use gradient descent to minimize the error of a model on our training data. Error is measured by the square error function
    """

    def __init__(self):
        """Initialize a gradient descent model to fit on a training dataset. Linear model h = theta_0*X_0 + theta_1*X_1 + ... + theta_n*X_n where n is the number of features."""
        print("Initialized Gradient Descent model...")
    
    def compute_loss(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X_b = np.c_[ones, X]
        y_b = y.reshape(len(y), 1)
        h = X_b.dot(self.theta)
        m = len(y)
        loss = h - y_b
        return (1 / (2 * m)) * np.sum(loss ** 2)

    def fit(self, X, y, alpha=0.01, iterations=1000):
        """Fit the model on the training set.

        Args:
            X (float numpy array): Feature matrix of n*m dimentions where n is number of features and m is number of training examples.
            y (float numpy array): Traget vector with m*1 dimentions.
            alpha (float, optional): Learning rate. Defaults to 0.01.
            iterations (int, optional): Number of iterations to train. Defaults to 1000.
        """
        ones = np.ones((X.shape[0], 1))
        X_b = np.c_[ones, X]
        y_b = y.reshape(len(y), 1)
        m = len(y)
        
        self.theta = np.zeros((X_b.shape[1], 1))
        self.losses = []
        self.iterations = iterations

        for _ in range(iterations):
            h = X_b.dot(self.theta)
            gradients = (1 / m) * X_b.T.dot(h - y_b)
            self.theta = self.theta - alpha * gradients
            self.losses.append(self.compute_loss(X, y))
        print("Successfully fitted to training data")
        print(f"loss: {self.compute_loss(X, y)}")
    
    def predict(self, vector):
        """ Predict value on model"""
        ones = np.ones((vector.shape[0], 1))
        vec_b = np.c_[ones, vector]
        predictions = vec_b.dot(self.theta)
        return predictions


    #def visualize_loss(self, figsize = (12, 5)):
      #  """Visualize the loss in the cost function as a function of number of iterations.

      #  Args:
       #     figsize (tuple of int): Size of figure. Example (12, 5)
       # """
        #ax, fig = plt.subplots(figsize)
        #ax.plot(range(self.iterations), self.losses)
        #ax.set_xlabel("Cost function J")
        #ax.set_ylabel("Number of iterations")
        #ax.set_title("Gradient Descent Convergence")
        #plt.show()
    
    def score(self, X, y):
        """Score using mean squared error"""
        self.compute_loss(X, y)


class NormalEquation():
    """Compute linear model directly using normal equation
    """

    def __init__(self):
        """Initialize a model based on the Normal Equation"""
        print("Successfully initialized a model using the normal equation...")
    
    def compute_loss(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X_b = np.c_[ones, X]
        y_b = y.reshape(len(y), 1)
        h = X_b.dot(self.theta)
        m = len(y)
        loss = h - y_b
        return (1 / (2 * m)) * np.sum(loss ** 2)
    
    def fit(self, X, y):
        """Fit model to training set"""
        ones = np.ones((X.shape[0], 1))
        X_b = np.c_[ones, X]
        y_b = y.reshape(len(y), 1)
        self.theta = np.linalg.inv(X_b.transpose().dot(X_b)).dot(X_b.transpose().dot(y_b))
        print("Successfully fitted to training data")
        print(f"Loss on training data: {self.compute_loss(X, y)}")
    
    def predict(self, vector):
        """ Predict value on model """
        ones = np.ones((vector.shape[0], 1))
        vec_b = np.c_[ones, vector]
        predictions = vec_b.dot(self.theta)
        return predictions

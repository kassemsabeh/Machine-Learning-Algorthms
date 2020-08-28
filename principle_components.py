import numpy as np
class PCA():
    
    def __init__(self, k):
        self.k = k
        print("Make sure that input data is normalized")
    
    def fit(self, X):
        
        sigma = (1 / X.shape[0]) * (X.T.dot(X)) 
        self.U, _, _ = np.linalg.svd(sigma)
        self.U_red = self.U[:, :self.k]
    
    def transform(self, X):
        return X.dot(self.U_red)
    
    def recover(self, Z):
        assert Z.shape[1] == self.k, 'Incompatible dimensions'
        return Z.dot(self.U_red.T)
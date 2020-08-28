import numpy as np
import  random

#Define a cluster d.s
class Cluster():
    def __init__(self, centroid):
        self.centroid = centroid
        self.points = []
    
    def add_point(self, point):
        self.points.append(point)
    
    def remove_points(self):
        self.points=[]

class KMeans():
    def __init__(self, k):
        self.k = k
    
    def __random_point(self, X):
        limit = X.shape[0]
        number = random.randint(0, limit)
        return X[number]
    
    def __update_points(self, X):
        
        #Remove all points
        for cluster in self.clusters:
            cluster.remove_points()
        
        for point in X:
            dist = 100
            for i in range(len(self.clusters)):
                if (dist > np.linalg.norm(point - self.clusters[i].centroid)):
                    dist = np.linalg.norm(point - self.clusters[i].centroid)
                    max_cluster = i
            self.clusters[max_cluster].add_point(point)
        
    def __update_centroids(self, X):
        
        for cluster in self.clusters:
            cluster.centroid = np.mean(cluster.points, axis=0)
                

    
    def fit(self, X, iterations=30):
        #1 - Random initialization
        self.clusters = []
        for _ in range(self.k):
            random_point = self.__random_point(X)
            self.clusters.append(Cluster(random_point))
        for _ in range(iterations):
            
            #2.1 - Assign data points to clusters
            self.__update_points(X)
        
            #2.2 - Assign new centroids from mean 
            self.__update_centroids(X)
        
        for cluster in self.clusters:
            cluster.points = np.array(cluster.points)
    

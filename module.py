from sklearn.datasets import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random

class KMeans:
    K=3
    dataset= "iris"
    observations = 150
    features = 4
    centroid = []
    distance = []
    clusterAssign = []
    epochNumber = 100
    tol = 0.0001
    normalize = "false"
    elbowAnalysis = "false"
    
    def setDataset(self,file):
        self.dataset=file
    
    def setK(self, k):
        self.K = k
     
    def initialize(self, d):
        self.centroid.clear()
        self.distance.clear()
        self.clusterAssign.clear()
        
        #print(type(d.X))
        
        sampleList = d.X.sample(n = self.K)
        #print(sampleList)
        
        self.centroid = sampleList.values.tolist()
            
        #print(self.centroid)
        
        
    
    def setObservations(self,n):
        self.observations = n
        
    def setFeatures(self, n):
        self.features = n
        
    def setEpochNumber(self,en):
        self.epochNumber=int(en)
        
    def setNormalize(self, string):
        self.normalize = string.lower()
    
    def setElbowAnalysis(self, string):
        self.elbowAnalysis = string.lower()
        if self.elbowAnalysis == "true":
            print("Elbow Analysis is set. Given K value (if any) will be ignored.")
        
class Dataset:
    
    X = ""

    filenames = {"iris":load_iris,"boston":load_boston,"breast_cancer":load_breast_cancer,"diabetes":load_diabetes,
                "wine": load_wine, "digits":load_digits}
    
    def convertToDataFrame(self, file):
       
        df = pd.DataFrame(file.data, columns=file.feature_names)
        #df['target'] = pd.Series(file.target) [target variable not needed in clustering]
        return df
        
    def setX(self,X):
        self.X=X


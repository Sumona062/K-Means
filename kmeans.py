import sys
import pandas as pd
import module
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

def findMax(series):
    max = series[0]
    for i in range(len(series)-1):
        if(series[i+1] > max):
            max = series[i+1]

    #print(max)
    return max

def findMin(series):
    min = series[0]
    for i in range(len(series)-1):
        if(series[i+1] < min):
            min = series[i+1]

    #print(min)
    return min
   
def normalization(d):
    
    #print(type(d.X))
    
    cols = d.X.shape[1]
    #print(cols)
    
    for c in range(cols):
        series = d.X.iloc[:,c]
        max = findMax(series)
        min = findMin(series)
        for i in range(len(series)):
            d.X.iloc[i,c] = ((d.X.iloc[i,c]) - min)/ (max - min)
            #print(d.X.iloc[i,c])

def encode(file):
    
    categorical_cols=[]
    for column_name in file.columns:
        if file[column_name].dtype == object:
            categorical_cols.append(column_name) 
            
    file=pd.get_dummies(file,columns=categorical_cols)
        
    return file

def labelEncode(file):
    
    for column_name in file.columns:
        if file[column_name].dtype == object:
            labelEncoder = preprocessing.LabelEncoder()
            labelEncoder.fit_transform(file[column_name])
    
    #print('inside label encode')
    return file
    
def preprocess(m,d):
    
    if m.dataset in d.filenames: # built-in files
        data=d.filenames[m.dataset]()
        file = d.convertToDataFrame(data)
        
    else: # user-given files
        file=pd.read_csv(m.dataset+".csv")
        
    #print(file)
    n=len(file.columns)
    #d.setX(file.iloc[:,:n-1])
    d.setX(labelEncode(file))
    
    
    #print('After Encoding and Normalization - X: ')
    #print(d.X)
    m.setObservations(file.shape[0])
    m.setFeatures(file.shape[1])
    
def setValue(m,arg,j,d):
    
    if arg== "-d":
        m.setDataset(sys.argv[j+1])
        preprocess(m,d) 
    elif arg=="-k":
        m.setK(int(sys.argv[j+1]))
        
    elif arg== "-en":
        m.setEpochNumber(sys.argv[j+1])
            
    elif arg=="-normalize":
        m.setNormalize(sys.argv[j+1])
        
    elif arg=="-elbow":
        m.setElbowAnalysis(sys.argv[j+1])
        
    else:
        print(arg)
        print("Wrong argument given. The accepted flags are: -d, -k, -en, -normalize and -elbow.");

def findMinArg(series):
    min = series[0]
    minArg = 0
    for i in range(len(series)-1):
        if(series[i+1] < min):
            min = series[i+1]
            minArg = i+1
    return minArg

def squaredSum (a, b, m):
    sum = 0
    for i in range(m.features):
        sum += (a[i] - b[i])**2
    return sum

def euclidean(a, b, m):
    temp = squaredSum(a, b, m)
    return temp**0.5

def clusterAssignment(m, d):
    m.clusterAssign.clear()
    for j in range(m.observations):
        series = []
        for i in range(m.K): 
            series.append(m.distance[i][j])
        selectCluster = findMinArg(series)
        m.clusterAssign.append(selectCluster)

def SSECalculation(m,d):
    sse = 0
    for i in range(m.observations):
        sse += squaredSum(d.X.iloc[i], m.centroid[m.clusterAssign[i]], m)
        #print(sse)
    return sse

def findMean(series, m):
    point = []
    for j in range(m.features):
        sum = 0
        for i in range(len(series)):
            #print( i, j)
            sum += series[i][j]
        
        if len(series) != 0:
            point.append(sum/len(series))
        
    return point
       
def centroidUpdate(m, d):
    m.centroid.clear()
    for i in range(m.K):
        series = []
        for j in range(m.observations):
            if m.clusterAssign[j] == i:
                series.append(d.X.iloc[j].values.tolist())
        
        print(series)
        m.centroid.append(findMean(series, m))

def equalCentroid(previous, current):
    flag = True
    for i in range(m.K):
        for j in range(m.features):
            if current[i][j] <= previous[i][j] + m.tol and current[i][j] >= previous[i][j] - m.tol:
                continue
            else:
                flag = False
                break
    
    return flag
    
def clusterCalculation(m, d):
    
    en = 1
    dist = []
    sse = 0
    while(True):
        if(en > m.epochNumber):
            break
        
        print("Epoch no.: ", en)
        m.distance.clear()
        for i in range(len(m.centroid)):
            dist = []
            for j in range(m.observations):                    
                dist.append(euclidean(m.centroid[i],d.X.iloc[j].values.tolist(), m))
            #print(dist)
            m.distance.append(dist)
        
        print("Distance Matrix ... ")
        for p in range(len(m.centroid)):
            for q in range(m.observations):
                print("%.2f"%m.distance[p][q], end = " ")
            print()
        
        
        clusterAssignment(m, d)
        print("Cluster Assignment ... ")
        print(m.clusterAssign)
        
        sse = SSECalculation(m,d)
        print("Sum of Squared Distance", sse)
        
        # keep previous centroid for checking
        previousCentroid = m.centroid.copy()
        centroidUpdate(m,d)
        
        print("Updated Centroids ...  ")
        print(m.centroid)
        
        #print(previousCentroid)
        #print(m.centroid)
        flag = equalCentroid(previousCentroid, m.centroid)
        
        if flag == True:
            break
        
        en = en+1 #increasing epoch
        
    return sse
        
def printCluster(m, d):
    for i in range(m.K):
        print("Cluster # ", i+1)
        print("Centroid: [", end = " ")
        for j in range(m.features):
            if j==m.features-1:
                print("%.2f"%m.centroid[i][j], end = " ")
            else:
                print("%.2f"%m.centroid[i][j], end = ", ")

        print("]")
            
        
        for j in range(m.observations):
            if m.clusterAssign[j] == i:
                print("Point # ", j , ": " ,d.X.iloc[j].values.tolist())
    
def printElbowAnalysis(k, SSE):
    print()
    print("k and SSE: ELbow ANalysis")
    print("---------------------------")
    for i in range(len(k)):
        print("k = ", k[i], " : SSE = ", SSE[i])
        
def plotCurve(k, SSE):
    plt.scatter(k, SSE, c = 'blue')
    plt.plot(k, SSE, "b--")
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.title("SSE vs. k (Elbow Analysis)")
    plt.show()
    
if __name__=="__main__":
    
    print("Program name: ",sys.argv[0].replace('py',''))
    print("Program name(with type): ",sys.argv[0])
    print("Element number of program: ",len(sys.argv))
    print("Argument list:", sys.argv)
    NumofParam= len(sys.argv)
    #print("Num of Params= ",NumofParam)
    list=sys.argv
     
    
    m=module.KMeans()
    d=module.Dataset()
    preprocess(m,d)
    
    for i in range(1,NumofParam,2):
        argument=list[i].replace(' ','')
        setValue(m,argument,i,d)
    
    print("------------------------")
    print("Parameters Description")
    print("------------------------")
    print("Dataset: ",m.dataset)
    print("K: ", m.K)
    print("Number of Observations: ", m.observations)
    print("Number of Features: ", m.features)
    print("Epoch Number: ",m.epochNumber)
    print("Normalization: ", m.normalize)
    print("Elbow Analysis: ", m.elbowAnalysis)
    
   
    print("------------------------")
    
    print(m.normalize)
    if m.normalize == "true":
        normalization(d)
        
    print("After encoding and Normalization - X")
    print(d.X)
        
    
    
    
    
    
    if m.elbowAnalysis == "true":
        k = []
        SSE = []
        for i in range (2, 11, 1):
            
            m.K = i
            m.initialize(d)
            print("---------------------------")
            print("Number of CLusters, k = ", i);
            print("---------------------------")
            print("Initial Centroids: ", m.centroid)
            
            sse = clusterCalculation(m,d)
    
            print("Final Cluster Formation ... ")
            printCluster(m, d)
            k.append(i)
            SSE.append(sse)
            
        printElbowAnalysis(k, SSE)
        plotCurve(k, SSE)
        
    else:
        m.initialize(d)
        print("Initial Centroids: ", m.centroid)
        clusterCalculation(m,d)
        print("Final Cluster Formation ... ")
        printCluster(m, d)
    
    
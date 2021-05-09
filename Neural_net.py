
# coding: utf-8

# In[221]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


class NeuralNet:
    def __init__(self, feats, labels, nInp, nHid, nOut, miu=0.1, nItr = 10):
        self.feats= feats
        self.labels=labels
        self.ni = nInp
        self.nh = nHid
        self.no = nOut
        self.miu = miu
        self.nItr = nItr
        
        self.iNodes = np.zeros(self.ni)
        self.hNodes = np.zeros(self.nh)
        self.oNodes = np.zeros(self.no)

        self.ihWeights = np.random.randn(self.ni, self.nh)#np.zeros(shape=[self.ni, self.nh], dtype=np.float32)
        self.hoWeights = np.random.randn(self.nh, self.no)#np.zeros(shape=[self.nh, self.no], dtype=np.float32)

        self.hBiases = np.ones(self.nh)*0.05
        self.oBiases = np.ones(self.no)*0.05

        
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
    
    def forward(self, x):
        self.iNodes=x
        
        vInp=np.dot(self.iNodes,self.ihWeights)
        self.hNodes=self.sigmoid(vInp)+self.hBiases
        
        vHid=np.dot(self.hNodes,self.hoWeights)
        self.oNodes=self.sigmoid(vHid)+self.oBiases
    
    def sigmoidPrime(self, s):
        return s * (1 - s)
    
    def backward(self):
        sumhDel = 0
        sumoDel = 0
        for (x,y) in zip(self.feats,self.labels):
            self.forward(x)
            self.oError = self.oNodes-y
            self.oDelta = self.oError*self.sigmoidPrime(self.oNodes)

            self.hError = np.dot(self.oDelta, self.hoWeights.T)
            self.hDelta = self.hError*self.sigmoidPrime(self.hNodes)

            sumoDel += np.dot(np.array([self.hNodes]).T, np.array([self.oDelta])) #np.dot(np.array([y_r-1]).T,np.array([delta]))
            sumhDel += np.dot(np.array([self.iNodes]).T, np.array([self.hDelta]))

        self.hoWeights += -self.miu*sumoDel
        self.ihWeights += -self.miu*sumhDel

    def train(self, iterate=False):
        if iterate:
            for i in range(self.nItr):
                self.backward()
        else:
            loop_count = 0
            conv_count = 0
            div_len = 10

            prev_acc = self.accuracy(self.feats, self.labels)
            while True:

                self.backward()
                loop_count += 1
                accuracy = self.accuracy(self.feats, self.labels)
                print(accuracy)
                if accuracy > 70.0 and abs(accuracy - prev_acc) <= div_len:
                    conv_count += 1
                    prev_acc = accuracy
                else:
                    conv_count = 0
                if conv_count == 20:
                    # if accuracy==100.0:
                    print("Converges after: " + str(loop_count) + " iterations")
                    print("accuracy: " + str(self.accuracy(self.feats, self.labels)))
                    break
                if loop_count == 200:
                    print("accuracy so far: " + str(self.accuracy(self.feats, self.labels)))
                    print(str(loop_count) + " iterations")
                    break

    def accuracy(self, feats, labels):
        count =0
        for (x,y) in zip(feats,labels):
            self.forward(x)
            if np.argmax(y)==np.argmax(self.oNodes):
                count+=1
        return count/len(self.labels)*100
               


# In[279]:


# infile = open('./datasets/Train.txt', 'r')
# nfeat = infile.read(1)
# infile.read(1)
# nClass = infile.read(1)

# print(nfeat)
# print(nClass)
dataset = pd.read_table('./Supplied/trainNN.txt', sep='\s+', skiprows=[0], header=None)
# dataset = pd.read_table('./datasets/Train.txt', sep='\s+', skiprows=[0], header=None)
# dataset = pd.read_table('./Perceptron/trainLinearlyNonSeparable.txt', sep='\s+', header=None)
array = dataset.values
train_x = array[:, :-1]
train_y = array[:, -1]

feats = MinMaxScaler().fit_transform(train_x)
labels = OneHotEncoder().fit_transform(train_y.reshape(-1, 1)).toarray()
nFeat=np.shape(feats)[1]
nClass=np.shape(labels)[1]


# In[ ]:


model = NeuralNet(feats,labels,nFeat,10,nClass)

# model.forward(x)
# print(model.iNodes)
# print(model.hNodes)
# print(model.oNodes)
model.train()
# print(model.oError)



# In[ ]:

dataset = pd.read_table('./Supplied/testNN.txt', sep='\s+', skiprows=[0], header=None)
# dataset = pd.read_table('./datasets/Test.txt', sep='\s+', skiprows=[0], header=None)
# dataset = pd.read_table('./Perceptron/testLinearlyNonSeparable.txt', sep='\s+', header=None)
array = dataset.values
train_x = array[:, :-1]
train_y = array[:, -1]


feats = MinMaxScaler().fit_transform(train_x)
labels = OneHotEncoder().fit_transform(train_y.reshape(-1, 1)).toarray()
print(model.accuracy(feats,labels))


# In[201]:


# print(np.random.randint(low=1, high=2, size=[4,5]))


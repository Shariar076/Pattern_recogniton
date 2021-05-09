
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


# In[94]:


class NeuralNet:
    def __init__(self, feats, labels, hidShape, miu=0.1, nItr = 10):
        self.feats= feats
        self.labels=labels
        self.ni = np.shape(feats)[1]
        self.no = np.shape(labels)[1]
        self.hShape = hidShape
        self.miu = miu
        self.nItr = nItr
        
        self.iNodes = np.zeros(self.ni)
        self.oNodes = np.zeros(self.no)
        
        self.ihWeights = np.random.randn(self.ni, self.hShape[0])
        
        self.oBiases = np.ones(self.no)*0.05

        self.hWeights = {}
        for i in range(len(self.hShape)):
            
            if i==len(self.hShape)-1:
                self.hWeights[i]=np.random.randn(self.hShape[i], self.no)
            else:
                self.hWeights[i]=np.random.randn(self.hShape[i], self.hShape[i+1])
        
        self.hNodes = {}
        self.hBiases = {}
        for i in range(len(self.hShape)):
            self.hNodes[i]=np.ones(self.hShape[i])*0.05
            self.hBiases[i]=np.ones(self.hShape[i])*0.05
        
        
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
    
    def forwardpass(self, x):
        self.iNodes=x
        
        vHid=np.dot(self.iNodes, self.ihWeights)
        
        for i in range(len(self.hShape)):
            self.hNodes[i] = self.sigmoid(vHid)+self.hBiases[i]
            vHid=np.dot(self.hNodes[i],self.hWeights[i])

        self.oNodes=self.sigmoid(vHid)+self.oBiases
    
    def delSigmoid(self, s):
        return s * (1 - s)
    
    def backwardpass(self):
        sumihDel = np.zeros(shape=[self.ni, self.hShape[0]])
        sumhDel = {}
        for i in range(len(self.hShape)):
            if (i==len(self.hShape)-1):

                sumhDel[i]= np.zeros(shape=[self.hShape[i], self.no])
            else:
                sumhDel[i]= np.zeros(shape=[self.hShape[i], self.hShape[i+1]])
        for (x, y) in zip(self.feats,self.labels):
            self.forwardpass(x)
            self.oError = self.oNodes-y
            self.oDelta = self.oError*self.delSigmoid(self.oNodes)
             
            self.hError = {}
            self.hDelta = {}
            
            for i in range(len(self.hShape)-1, -1, -1):
                if i == len(self.hShape)-1:
                    self.hError[i] = np.dot(self.oDelta, self.hWeights[i].T)
                    self.hDelta[i] = self.hError[i]*self.delSigmoid(self.hNodes[i])
                    sumhDel[i] += np.dot(np.array([self.hNodes[i]]).T, np.array([self.oDelta]))
                else:
                    self.hError[i] = np.dot(self.hDelta[i+1], self.hWeights[i].T)
                    self.hDelta[i] = self.hError[i]*self.delSigmoid(self.hNodes[i])
                    sumhDel[i] += np.dot(np.array([self.hNodes[i]]).T, np.array([self.hDelta[i+1]]))
            
            sumihDel += np.dot(np.array([self.iNodes]).T, np.array([self.hDelta[0]]))

        for i in range(len(self.hShape)):
            self.hWeights[i] += -self.miu*sumhDel[i]
        self.ihWeights += -self.miu*sumihDel

    def train(self, iterate=False):
        if iterate:
            for i in range(self.nItr):
                self.backwardpass()
        else:
            loop_count = 0
            conv_count = 0
            div_len = 1.0

            prev_acc = self.accuracy(self.feats, self.labels)[0]
            # print(prev_acc)
            while True:

                self.backwardpass()
                loop_count += 1
                accuracy = self.accuracy(self.feats, self.labels)[0]
                print(accuracy)
                # if accuracy>90.0 and abs(accuracy-prev_acc)<=div_len:
                if abs(accuracy - prev_acc) <= div_len:
                    conv_count += 1
                else:
                    conv_count = 0
                prev_acc = accuracy
                if conv_count == 60:
                    print("Converges after: " + str(loop_count) + " iterations")
                    print("accuracy: " + str(self.accuracy(self.feats, self.labels)[0]))
                    break
                # if loop_count==300:
                #     print("accuracy so far: "+str(self.accuracy(self.feats,self.labels)[0]))
                #     print(str(loop_count)+" iterations")
                #     break

    def accuracy(self, feats, labels):
        count =0
        for (x,y) in zip(feats,labels):
            self.forwardpass(x)
            if np.argmax(y)==np.argmax(self.oNodes):
                count+=1
        return count/len(self.labels)*100, len(self.labels)-count
               


# In[96]:



dataset = pd.read_table('./Supplied/trainNN.txt', sep='\s+', skiprows=[0], header=None)
# dataset = pd.read_table('./Perceptron/trainLinearlyNonSeparable.txt', sep='\s+', header=None)
array = dataset.values
train_x = array[:, :-1]
train_y = array[:, -1]

feats = MinMaxScaler().fit_transform(train_x)
labels = OneHotEncoder().fit_transform(train_y.reshape(-1, 1)).toarray()

model = NeuralNet(feats,labels,[4,5,4,5])
model.train()

dataset = pd.read_table('./Supplied/testNN.txt', sep='\s+', skiprows=[0], header=None)
# dataset = pd.read_table('./Perceptron/testLinearlyNonSeparable.txt', sep='\s+', header=None)
array = dataset.values
train_x = array[:, :-1]
train_y = array[:, -1]

feats = MinMaxScaler().fit_transform(train_x)
labels = OneHotEncoder().fit_transform(train_y.reshape(-1, 1)).toarray()
print("(accuracy, missclassified): "+str(model.accuracy(feats,labels)))




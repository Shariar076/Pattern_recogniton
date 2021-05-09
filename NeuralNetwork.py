import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


class NeuralNet:
    def __init__(self, feats, labels, hidShape, miu=0.1, nItr = 10, activaion=1):
        self.feats= feats
        self.labels=labels
        self.ni = np.shape(feats)[1]
        self.no = np.shape(labels)[1]
        self.hShape = hidShape
        self.miu = miu
        self.nItr = nItr
        self.activation=activaion

        self.iNodes = np.zeros(self.ni)
        self.oNodes = np.zeros(self.no)
        
        self.ihWeights = np.random.randn(self.ni, self.hShape[0]) #np.zeros(shape=[self.ni, self.hShape[0]])
        
        self.oBiases = np.ones(self.no)*0.05

        self.hWeights = {}
        for i in range(len(self.hShape)):
            
            if i==len(self.hShape)-1:
                self.hWeights[i]=np.random.randn(self.hShape[i], self.no) #np.zeros(shape=[self.hShape[i], self.no])
            else:
                self.hWeights[i]=np.random.randn(self.hShape[i], self.hShape[i+1]) #np.zeros(shape=[self.hShape[i], self.hShape[i+1]])
        
        self.hNodes = {}
        self.hBiases = {}
        for i in range(len(self.hShape)):
            self.hNodes[i]=np.ones(self.hShape[i])*0.05
            self.hBiases[i]=np.ones(self.hShape[i])*0.05
        print("Number of Neurons in input layer:"+str(self.ni))
        print("Number of Neurons in output layer:"+str(self.no))

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def hypertan(self, s):
            return np.tanh(s/2)

    def forwardpass(self, x):
        self.iNodes=x
        
        vHid=np.dot(self.iNodes, self.ihWeights)
        
        for i in range(len(self.hShape)):
            if self.activation==1:
                self.hNodes[i] = self.sigmoid(vHid)+self.hBiases[i]
            else:
                self.hNodes[i] = self.hypertan(vHid)+self.hBiases[i]
            vHid=np.dot(self.hNodes[i],self.hWeights[i])

        if self.activation == 1:
                self.oNodes = self.sigmoid(vHid)+self.oBiases
        else:
            self.oNodes = self.hypertan(vHid)+self.oBiases

    def delSigmoid(self, s):
        return s * (1 - s)

    def delHypertan(self, s):
        return (1 - s) * (1 + s)

    def backwardpass(self):
        sumihDel = np.zeros(shape=[self.ni, self.hShape[0]])
        sumhDel = {}
        for i in range(len(self.hShape)):
            if i == len(self.hShape) - 1:
                sumhDel[i]= np.zeros(shape=[self.hShape[i], self.no])
            else:
                sumhDel[i]= np.zeros(shape=[self.hShape[i], self.hShape[i+1]])
        for (x, y) in zip(self.feats, self.labels):
            self.forwardpass(x)

            self.oError = self.oNodes-y
            if self.activation==1:
                self.oDelta = self.oError*self.delSigmoid(self.oNodes)
            else:
                self.oDelta = self.oError * self.delHypertan(self.oNodes)

            # self.oBiases=self.oDelta
            self.hError = {}
            self.hDelta = {}
            for i in range(len(self.hShape)-1, -1, -1):
                if i == len(self.hShape)-1:
                    self.hError[i] = np.dot(self.oDelta, self.hWeights[i].T)

                    if self.activation == 1:
                        self.hDelta[i] = self.hError[i]*self.delSigmoid(self.hNodes[i])
                    else:
                        self.hDelta[i] = self.hError[i]*self.delHypertan(self.hNodes[i])

                    sumhDel[i] += np.dot(np.array([self.hNodes[i]]).T, np.array([self.oDelta]))

                else:
                    self.hError[i] = np.dot(self.hDelta[i+1], self.hWeights[i].T)
                    if self.activation == 1:
                        self.hDelta[i] = self.hError[i]*self.delSigmoid(self.hNodes[i])
                    else:
                        self.hDelta[i] = self.hError[i]*self.delHypertan(self.hNodes[i])

                    sumhDel[i] += np.dot(np.array([self.hNodes[i]]).T, np.array([self.hDelta[i+1]]))

                # self.hBiases[i] = self.hDelta[i]
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
            div_len = .1
            
            prev_acc=self.accuracy(self.feats,self.labels)[0]
            # print(prev_acc)
            while True:
                if prev_acc<100.0:
                    self.backwardpass()
                loop_count+=1
                accuracy = self.accuracy(self.feats,self.labels)[0]
                print(accuracy)
                if accuracy>95.0 and abs(accuracy-prev_acc) <= div_len:
                    conv_count+=1
                else:
                    conv_count=0
                prev_acc = accuracy
                if conv_count==100:
                    print("Converges after: "+str(loop_count-100)+" iterations")
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


# dataset = pd.read_table('./Perceptron/trainLinearlyNonSeparable.txt', sep='\s+', header=None)
# dataset = pd.read_table('./datasets/Train.txt', sep='\s+', header=None)
# dataset = pd.read_table('./Supplied/trainNN.txt', sep='\s+',  header=None)
dataset = pd.read_table('./evaluation/trainNN.txt', sep='\s+',  header=None)
print("Training size:"+str(len(dataset)))
array = dataset.values
train_x = array[:, :-1]
train_y = array[:, -1]

feats = MinMaxScaler().fit_transform(train_x)
labels = OneHotEncoder().fit_transform(train_y.reshape(-1, 1)).toarray()

model = NeuralNet(feats,labels,[5,4,6,6],miu=0.01,activaion=1)  #miu=0.01 for act=1, miu=0.001 for act=2

model.train()

# dataset = pd.read_table('./Perceptron/testLinearlyNonSeparable.txt', sep='\s+', header=None)
# dataset = pd.read_table('./datasets/Test.txt', sep='\s+', header=None)
# dataset = pd.read_table('./Supplied/testNN.txt', sep='\s+', header=None)
dataset = pd.read_table('./evaluation/testNN.txt', sep='\s+', header=None)
print("Testing size:"+str(len(dataset)))
array = dataset.values
train_x = array[:, :-1]
train_y = array[:, -1]

feats = MinMaxScaler().fit_transform(train_x)
labels = OneHotEncoder().fit_transform(train_y.reshape(-1, 1)).toarray()
print("(accuracy, missclassified): "+str(model.accuracy(feats,labels)))




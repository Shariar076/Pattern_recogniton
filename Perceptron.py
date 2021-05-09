import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, p=0.01, itr=10):
        self.p = p                #learning rate
        self.itr = itr            #num of iteration

    def fit(self, features, labels):
        self.k =len(np.unique(labels))  #num of classes

        row_size=(len(features[0])+1)*self.k
        col_size=len(features)*(self.k-1)

        feat_space=np.zeros([col_size,row_size])

        self.lbl_names=np.arange(self.k)

        i = 0
        for feat,label in zip(features, labels):
            pos = int(label * (len(feat) + 1))
            arr = np.delete(self.lbl_names, label)
            for j in range(self.k-1):
                oth_pos= arr[j]*(len(feat) + 1)
                feat_space[i + j, oth_pos + 1:oth_pos + len(feat) + 1] = np.negative(feat)
                feat_space[i + j][oth_pos] = -1
                feat_space[i + j, pos + 1:pos + len(feat) + 1] = feat
                feat_space[i + j][pos] = 1
            i += (self.k-1)

        self.w = np.zeros((1 + features.shape[1])*self.k)

        for i in self.lbl_names:
            self.w[i*(len(features[0])+1)]=1

        for t in range(self.itr):
            for feat in feat_space:
                wx = np.dot(feat, self.w)
                if wx <= 0:
                    self.w += feat*self.p
        return self

    def predict(self, feature):
        w=np.array([])
        feature = np.insert(feature, 0, 1)
        for i in range(self.k):
            pos=int(self.lbl_names[i] * len(feature))
            pred=np.dot(feature, self.w[pos: pos+len(feature)])
            w=np.append(w, pred)
        return np.argmax(w)


# dataset = pd.read_csv('iris.csv', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
# dataset = pd.read_table('./kesler/Train.txt', sep='\s+', names=['f1', 'f2', 'f3', 'label'])
dataset =pd.read_table('./Perceptron/trainLinearlyNonSeparable.txt', sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'f5', 'label'])
# dataset = pd.read_table('sample.txt', sep='\s+', names=['f1', 'f2', 'label'])


dataset[dataset.iloc[:, -1].name] = dataset[dataset.iloc[:, -1].name] \
    .replace({1 : 0, 2: 1, 3: 2})

dataset = dataset.sample(frac=1).reset_index(drop=True)

array = dataset.values
train_x = array[:, :-1]
train_y = array[:, -1]

# dataset = pd.read_table('./kesler/Test.txt', sep='\s+', names=['f1', 'f2', 'f3', 'label'])
dataset = pd.read_table('./Perceptron/testLinearlyNonSeparable.txt', sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'f5', 'label'])

dataset[dataset.iloc[:, -1].name] = dataset[dataset.iloc[:, -1].name] \
    .replace({1 : 0, 2: 1, 3: 2})
array = dataset.values
test_x = array[:, :-1]
test_y = array[:, -1]

model = Perceptron(0.01,100)
model = model.fit(features=train_x, labels=train_y)
count = 0

for xi, yi in zip(test_x, test_y):
    predict = model.predict(feature=xi)
    if predict == yi:
        count += 1

print(float(count)*100/float(len(test_x)))

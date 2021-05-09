import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, p=0.01, itr=10):
        self.p = p
        self.itr = itr

    def fit(self, features, labels):
        self.w = np.zeros(1 + features.shape[1])
        self.errors = []
        for t in range(self.itr):
            errors = 0
            for feat, label in zip(features, labels):
                update = self.p * float((label - self.predict(feat))/2) #p * +1  when pred = -1 lbl=1 and -1 otherwise
                self.w[1:] += update * feat #+/- p  * x
                self.w[0] += update
                if update != 0:
                    errors += 1
            self.errors.append(errors)
        return self

    def predict(self, features):
        p=np.dot(features, self.w[1:]) + self.w[0]
        return np.where(p >= 0.0, 1, -1)


class_to_drop = 3
# dataset = pd.read_csv('iris.csv', names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])
# dataset = pd.read_table('Train.txt', sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'f5', 'label'])
dataset = pd.read_table('./Perceptron/trainLinearlyNonSeparable.txt', sep='\s+', names=['f1', 'f2', 'f3', 'f4', 'f5', 'label'])
dataset = dataset.sample(frac=1).reset_index(drop=True)
# dataset = dataset.where(dataset['label'] != class_to_drop).dropna().reset_index(drop=True)

dataset[dataset.iloc[:, -1].name] = dataset[dataset.iloc[:, -1].name] \
    .replace({1 : 1, 2: -1})

array = dataset.values
train_x = array[:, :-1]
train_y = array[:, -1]

dataset = pd.read_table('./Perceptron/testLinearlyNonSeparable.txt', sep='\s+', names=['f1', 'f2', 'f3','f4','f5', 'label'])
# dataset = dataset.where(dataset['label'] != class_to_drop).dropna().reset_index(drop=True)
dataset[dataset.iloc[:, -1].name] = dataset[dataset.iloc[:, -1].name] \
    .replace({1 : 1, 2: -1})
array = dataset.values
test_x = array[:, :-1]
test_y = array[:, -1]

model = Perceptron(0.001,2)
model = model.fit(features=train_x, labels=train_y)
count = 0

for xi, yi in zip(test_x, test_y):
    predict = model.predict(features=xi)

    if predict == yi:
        count += 1

print(float(count)*100/float(len(test_x)))

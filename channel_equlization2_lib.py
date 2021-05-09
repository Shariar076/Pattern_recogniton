import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
class Channel:
	def __init__(self, data, A, B, l, noise):
		self.data=data
		self.A = A
		self.B = B
		self.nOmega = 2**(l+1)
		self.noise = noise
		self.cls_data = {}
		self.miu = np.zeros(shape=[self.nOmega,2])
		self.sigma = np.zeros(shape=[self.nOmega, 2, 2])
		self.count = np.zeros(self.nOmega, dtype=int)
		self.classes = np.zeros(self.nOmega, dtype=int)
		self.classes[:int(self.nOmega/2)] = 0
		self.classes[int(self.nOmega/2):] = 1

	def calc(self, I):
		cl = int(I, base=2)
		vec = np.zeros(2)
		vec[0] = self.A * int(I[0]) + self.B * int(I[1]) + np.random.normal(0.0, self.noise)
		vec[1] = self.A * int(I[1]) + self.B * int(I[2]) + np.random.normal(0.0, self.noise)

		if cl not in self.cls_data:
			self.cls_data[cl] = []
		self.cls_data[cl].append(vec)

	def train(self):
		for i in range(len(self.data)-2):
			self.calc(self.data[i:i+3])

		for i in range(self.nOmega):
			self.miu[i] = np.mean(np.array(self.cls_data[i]).T,axis=1)
			self.sigma[i] = np.cov(np.array(self.cls_data[i]).T)

	def predict_seq(self, data):
		x = [0]
		for i in range(len(data) - 1):
			I = data[i:i + 2]
			xk = self.A*int(I[0]) + self.B*int(I[1])
			x.append(xk)

		dp = np.ones(shape=[len(x), self.nOmega])*(-99999)
		parent = np.zeros(shape=[len(x), self.nOmega], dtype=int)

		dp[0][:] = np.log(0.25)

		for i in range(1, len(data)):
			for j in range(self.nOmega):
				if j < 4:
					from1 = 2 * j
					from2 = 2 * j + 1
				else:
					from1 = (j - 4) * 2
					from2 = (j - 4) * 2 + 1
				d=np.log(multivariate_normal.pdf([x[i], x[i - 1]], self.miu[j], self.sigma[j]))

				if dp[i][j] <= (dp[i - 1][from1] + np.log(0.5) + d):
					dp[i][j] = dp[i - 1][from1] + np.log(0.5) + d
					parent[i][j] = from1

				if dp[i][j] <= (dp[i - 1][from2] + np.log(0.5) + d):
					dp[i][j] = dp[i - 1][from2] + np.log(0.5) + d
					parent[i][j] = from2
		dp = np.array(dp)

		last = np.argmax(dp[99])

		pred = str(self.classes[last])

		for i in range(99, 0, -1):
			pred = str(self.classes[parent[i][last]]) + pred
			last = parent[i][last]

		return pred




# filename = './channel_data/train.txt'
filename = './Evaluation/train.txt'
file = open(filename)
train_size = 1000000
train_data=file.read(train_size)
# train_data=file.read()
print(len(train_data))
# filename = './channel_data/test.txt'
filename = './Evaluation/test.txt'
file = open(filename)
test_data=file.read()


# model = Channel(train_data, A=2, B=3, l=2, noise=10)
# # print(model.cls_data)
# model.train()
# # plt.scatter(model.miu[:,0],model.miu[:,1])
# # plt.show()
#
# print(model.miu)
# print(model.sigma)
#
# model.predict_seq(test_data)

cases = [
	[2, 3, 1],
	[1, 0.6, .1],
	[1, 0.6, .2],
	[1, 0.6, .5],
	[1, 0.6, 1.0],
	[1, 0.6, 8.5],
	[1, 6, 2],

	# [1, 0.6, .3],
	# [1, 0.6, 30]
	]
for case in cases:
	print("A:", case[0], " B:", case[1], " noise:", case[2])
	model = Channel(train_data, A=case[0], B=case[1], l=2, noise=case[2])
	model.train()
	pred_seq = model.predict_seq(test_data)
	# text_file = open("./channel_data/out.txt", "w")
	# text_file.write(pred_seq)
	# text_file.close()
	count = 0
	for i in range(min(len(test_data),len(pred_seq))):
		if pred_seq[i]==test_data[i]:
			count+=1

	print("accuracy", (count/(len(test_data))*100))

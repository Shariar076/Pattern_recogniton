import numpy as np
import matplotlib.pyplot as plt
class Channel:
	def __init__(self, data, A, B, l, noise):
		self.data=data
		self.A=A
		self.B=B
		self.nOmega=2**(l+1)
		self.noise =noise
		self.omega = np.zeros(shape=[self.nOmega,2])
		self.count = np.zeros(self.nOmega)
		self.classes = np.zeros(self.nOmega, dtype=int)
		self.classes[:int(self.nOmega/2)] = 0
		self.classes[int(self.nOmega/2):] = 1

	def calc(self, I):
		cl= int(I,base=2)

		self.omega[cl][0]+= self.A*int(I[0]) + self.B*int(I[1]) + np.random.normal(0.0, self.noise)
		self.omega[cl][1]+= self.A*int(I[1]) + self.B*int(I[2]) + np.random.normal(0.0, self.noise)
		self.count[cl] += 1

	def train(self):
		print("train")
		for i in range(len(self.data)-2):
			self.calc(self.data[i:i+3])
		print("end")

		for i in range(self.nOmega):
			self.omega[i][0] /= self.count[i]
			self.omega[i][1] /= self.count[i]

	def distance(self, x1, y1, x2, y2):
		return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

	def predict(self, I):
		x0 = self.A * int(I[0]) + self.B * int(I[1]) + np.random.normal(0.0, self.noise)
		x1 = self.A * int(I[1]) + self.B * int(I[2]) + np.random.normal(0.0, self.noise)

		cl = self.classes[np.argmin(np.array([self.distance(x0,x1,x[0],x[1]) for x in self.omega]))]
		return cl

	def predict_seq(self, data):
		seq = ''
		for i in range(len(data)-2):
			bit = self.predict(data[i:i+3])
			seq+=str(bit)
		return seq +'00'


filename = './channel_data/train.txt'
file = open(filename)
train_size = 10000
train_data=file.read(train_size)

filename = './channel_data/test.txt'
file = open(filename)
test_data=file.read(-1)

cases = [
	[10, 15, 1],
	[10, 15, 2],
	[10, 15, 3],
	[10, 15, 4],
	[0.5, 0, 0],
	[0.5, 0, .1],
	[0.5, 0, .3],
	[10, 15, .1]
	]
for case in cases:
	print("A:", case[0], " B:", case[1], " noise:", case[2])
	model = Channel(train_data, A=case[0], B=case[1], l=2, noise=case[2])
	model.train()
	# plt.scatter(model.omega[:,0],model.omega[:,1])
	# plt.show()
	pred_seq = model.predict_seq(test_data)
	text_file = open("./channel_data/out.txt", "w")
	text_file.write(pred_seq)
	text_file.close()
	count = 0
	for i in range(len(test_data)-2):
		if pred_seq[i]==test_data[i]:
			count+=1

	print("accuracy", (count/(len(test_data)-2))*100)

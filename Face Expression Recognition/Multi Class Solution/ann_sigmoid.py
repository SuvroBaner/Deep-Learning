import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getBinaryData, sigmoid, sigmoid_cost, error_rate, relu

class ANN(object):
	def __init__(self, M):
		self.M = M # number of hidden units of this model

	def fit(self, X, Y, learning_rate = 5*10e-7, reg = 1.0, epochs = 10000, show_fig = False):
		### dividing X and Y in train and test
		X, Y = shuffle(X, Y)
		Xvalid, Yvalid = X[-1000:], Y[-1000:]  # last 1000 points
		X, Y = X[:-1000], Y[:-1000]

		N, D = X.shape

		### initializing the weights-
		self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)   # input to hidden weight; 
		self.b1 = np.zeros(self.M) # bias is of value 0 of size M
		self.W2 = np.random.randn(self.M) / np.sqrt(self.M) # hidden to output weights
		self.b2 = 0

		costs = []
		best_validation_error = 1

		for i in range(epochs):
			# forward propagation
			pY, Z = self.forward(X)  # probability of Y|X and value at hidden layer (it will be used for gradient descent)

			# Gradient Descent step-
			pY_Y = pY - Y # prediction - target
			self.W2 -= learning_rate*(Z.T.dot(pY - Y) + reg*self.W2)   # first update hidden to output weights
			self.b2 -= learning_rate*((pY_Y).sum() + reg*self.b2)  # update the bias between output and hidden nodes

			# backpropagation between input to hidden weights-
			#dZ = np.outer(pY_Y, self.W2) * (Z > 0)  # derivative of relu
			dZ = np.outer(pY_Y, self.W2) * (1 - Z*Z)  # derivative of tanh
			# now, with this variable we can update input to hidden weights-
			self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
			self.b1 -= learning_rate*(np.sum(dZ, axis = 0) + reg*self.b1)

			if i%20 == 0:
				pYvalid, _ = self.forward(Xvalid)
				c = sigmoid_cost(Yvalid, pYvalid)
				costs.append(c)
				e = error_rate(Yvalid, np.round(pYvalid))
				print("i:", i, "cost:", c, "error:", e)

				if e < best_validation_error:
					best_validation_error = e
		print("Best Validation Error:", best_validation_error)

		if show_fig:
			plt.plot(costs)
			plt.show()

	def forward(self, X):
		#Z = relu(X.dot(self.W1) + self.b1) # for relu
		Z = np.tanh(X.dot(self.W1) + self.b1) # for tanh
		return sigmoid(Z.dot(self.W2) + self.b2), Z

	def predict(self, X):
		pY, _ = self.forward(X)
		return np.round(pY)

	def score(self, X, Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)


def main():
	X, Y = getBinaryData()
	
	# We increase the 1-sample, since Y == 1 's are much less

	X0 = X[Y == 0, :]
	X1 = X[Y == 1, :]
	X1 = np.repeat(X1, 9, axis = 0)
	X = np.vstack([X0, X1])
	Y = np.array([0]*len(X0) + [1]*len(X1))

	model = ANN(100)  # hidden layer size to 100 i.e. one hidden layer has 100 hidden units.
	model.fit(X, Y, show_fig = True)



if __name__ == '__main__':
	main()

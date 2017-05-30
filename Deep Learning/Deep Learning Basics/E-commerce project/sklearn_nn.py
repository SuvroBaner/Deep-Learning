from process import get_data

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

# Get the data-
X, Y = get_data()

# Split into train and test-
X, Y = shuffle(X, Y)
Ntrain = int(0.7*len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

# Create a Neural Network-
model = MLPClassifier(hidden_layer_sizes = (20, 20), max_iter = 2000)  # 2 hidden layers each of size = 20 and backpropagation will be 2000

# Train the neural network
model.fit(Xtrain, Ytrain)

# Print the train and test accuracy-
train_accuracy = model.score(Xtrain, Ytrain)
test_accuracy = model.score(Xtest, Ytest)

print("Train Accuracy: ", train_accuracy, "Test Accuracy: ", test_accuracy)
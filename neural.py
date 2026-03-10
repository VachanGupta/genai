Welcome to Newton Book
Powered By Edison
Newton Book is interactive web-based tool for data analysis, research, and teaching. Write code, create visualizations, and blend it with text to tell data-driven stories. Enjoy exploring and creating with Newton book!

print("Hello, Begin Your Data Journey")
!pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_data():
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=3, n_redundant=0, random_state=42)
    X = MinMaxScaler().fit_transform(X)
    y = y.reshape(-1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

class ActivationFunctions:
    def __init__(self, activation='sigmoid'):
        self.activation = activation

    def activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Unsupported activation function.")

    def derivative(self, x):
        if self.activation == 'sigmoid':
            sig = self.activate(x)
            return sig * (1 - sig)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError("Unsupported activation function.")
class CustomNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, activation='sigmoid', learning_rate=0.1):
         self.lr = learning_rate
         self.activation = ActivationFunctions(activation)
         self.W1 = np.random.randn(input_dim, hidden_dim)
         self.b1 = np.zeros((1, hidden_dim))
         self.W2 = np.random.randn(hidden_dim, 1)
         self.b2 = np.zeros((1, 1))

         self.losses = []

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation.activate(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))

        return self.a2
         
        

    def compute_loss(self, y_true, y_pred):
         return np.mean((y_true - y_pred) ** 2)
    def backward(self, X, y):
         m = X.shape[0]

         dz2 = self.a2 - y
         dW2 = np.dot(self.a1.T, dz2) / m
         db2 = np.sum(dz2, axis=0, keepdims=True) / m

         dz1 = np.dot(dz2, self.W2.T) * self.activation.derivative(self.z1)
         dW1 = np.dot(X.T, dz1) / m
         db1 = np.sum(dz1, axis=0, keepdims=True) / m

         self.W2 -= self.lr * dW2
         self.b2 -= self.lr * db2
         self.W1 -= self.lr * dW1
         self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=1000):
        for _ in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            self.backward(X, y)

        return self.losses

    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)
def train_and_compare(X_train, X_test, y_train, y_test, activation='sigmoid'):
    model = CustomNeuralNetwork(input_dim=X_train.shape[1], hidden_dim=6, activation=activation)
    losses = model.fit(X_train, y_train, epochs=1000)
    y_pred_custom = model.predict(X_test)
    custom_acc = accuracy_score(y_test, y_pred_custom)

    sklearn_activation = 'logistic' if activation == 'sigmoid' else activation
    sklearn_model = MLPClassifier(hidden_layer_sizes=(6,), activation=sklearn_activation, max_iter=1000)
    sklearn_model.fit(X_train, y_train.ravel())
    y_pred_sklearn = sklearn_model.predict(X_test)
    sklearn_acc = accuracy_score(y_test, y_pred_sklearn)

    return model.losses, custom_acc, sklearn_acc
def plot_loss(losses, activation):
    plt.plot(losses)
    plt.title(f"Loss vs Iterations ({activation})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
acc_custom =0
acc_sklearn =0
X_train, X_test, y_train, y_test = load_data()
losses_sigmoid, acc_custom, acc_sklearn = train_and_compare(X_train, X_test, y_train, y_test, activation='sigmoid')
plot_loss(losses_sigmoid, 'sigmoid')
acc_custom
acc_sklearn

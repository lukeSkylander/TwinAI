import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def cross_entropy(pred, label):
    return -np.log(pred[label] + 1e-9)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(hidden_size, 64) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros(64)

        self.W3 = np.random.randn(64, output_size) * np.sqrt(2 / 64)
        self.b3 = np.zeros(output_size)

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = softmax(self.z3)

        return self.a3

    def backward(self, x, y, learning_rate):
        # Output layer error (softmax + cross-entropy derivative)
        output_error = self.a3.copy()
        output_error[y] -= 1

        # Layer 3 gradients
        dW3 = np.outer(self.a2, output_error)
        db3 = output_error

        # Layer 2 error
        hidden2_error = self.W3 @ output_error
        hidden2_error *= relu_derivative(self.z2)

        # Layer 2 gradients
        dW2 = np.outer(self.a1, hidden2_error)
        db2 = hidden2_error

        # Layer 1 error
        hidden1_error = self.W2 @ hidden2_error
        hidden1_error *= relu_derivative(self.z1)

        # Layer 1 gradients
        dW1 = np.outer(x, hidden1_error)
        db1 = hidden1_error

        # Update weights and biases
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load(self, path):
        data = np.load(path)

        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.W3 = data["W3"]
        self.b3 = data["b3"]

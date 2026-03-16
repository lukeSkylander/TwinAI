from data_loader import load_mnist, debug_show_samples
from nn import NeuralNetwork
from draw_gui import DrawGUI

import numpy as np
import os

MODEL_PATH = "model/mnist_model.npz"
DATA_PATH = "./data/kaggle_mnist"

os.makedirs("model", exist_ok=True)


print("Loading MNIST dataset...")

X_train, y_train, X_test, y_test = load_mnist(DATA_PATH)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

debug_show_samples(X_train, y_train, 10)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)


if os.path.exists(MODEL_PATH):

    print("Loading saved model...")
    nn.load(MODEL_PATH)

else:

    print("Training model...")

    epochs = 20
    learning_rate = 0.01

    batch_size = 64

    for epoch in range(epochs):
        # Shuffle the training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        print(f"Epoch {epoch + 1}/{epochs}")

        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(X_train_shuffled), batch_size):
            X_batch = X_train_shuffled[i : i + batch_size]
            y_batch = y_train_shuffled[i : i + batch_size]

            batch_loss = 0
            for x, y in zip(X_batch, y_batch):
                pred = nn.forward(x)
                loss = -np.log(pred[y] + 1e-9)  # cross entropy loss
                batch_loss += loss
                nn.backward(x, y, learning_rate)

            epoch_loss += batch_loss
            num_batches += 1

            # Print progress every 100 batches
            if num_batches % 100 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Batch {num_batches}, Average Loss: {avg_loss:.4f}")

        # Print epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

        # Evaluate on a small subset every few epochs
        if (epoch + 1) % 5 == 0:
            correct = 0
            test_samples = min(1000, len(X_test))
            for i in range(test_samples):
                pred = nn.forward(X_test[i])
                if np.argmax(pred) == y_test[i]:
                    correct += 1
            accuracy = correct / test_samples
            print(f"  Validation accuracy after epoch {epoch + 1}: {accuracy*100:.2f}%")

    print("Saving model...")
    nn.save(MODEL_PATH)


# evaluation
correct = 0

for x, y in zip(X_test, y_test):

    pred = nn.forward(x)

    if np.argmax(pred) == y:
        correct += 1


accuracy = correct / len(X_test)

print(f"Accuracy: {accuracy*100:.2f}%")


print("Starting GUI...")
gui = DrawGUI(nn)
gui.run()

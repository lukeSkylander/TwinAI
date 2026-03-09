from data_loader import load_dataset
from nn import NeuralNetwork, cross_entropy
import numpy as np
import matplotlib.pyplot as plt
import os
from draw_gui import DrawGUI

os.makedirs("plots", exist_ok=True)

X_train, y_train = load_dataset("Reduced MNIST Data/Reduced Trainging data")
X_test, y_test = load_dataset("Reduced MNIST Data/Reduced Testing data")

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))
print("Vector:", X_train.shape)
print("Image vector size:", X_train.shape[1])

nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=64, output_size=10)

epochs = 20
learning_rate = 0.01

loss_history = []
accuracy_history = []

for epoch in range(epochs):

    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    total_loss = 0

    for x, y in zip(X_train, y_train):
        pred = nn.forward(x)
        total_loss += cross_entropy(pred, y)
        nn.backward(x, y, learning_rate)

    avg_loss = total_loss / len(X_train)
    loss_history.append(avg_loss)

    # calculate accuracy after each epoch
    correct = 0
    for x, y in zip(X_test, y_test):
        pred = nn.forward(x)
        if np.argmax(pred) == y:
            correct += 1

    accuracy = correct / len(X_test)
    accuracy_history.append(accuracy)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy*100:.2f}%")

# plot training loss
plt.figure()
plt.plot(range(1, epochs+1), loss_history, marker='o')
plt.xticks(range(1, epochs+1))
plt.title("Training Loss über Epochen")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("plots/training_loss.png")
plt.close()

# plot accuracy
plt.figure()
plt.plot(range(1, epochs+1), accuracy_history, marker='o')
plt.xticks(range(1, epochs+1))
plt.title("Testgenauigkeit über Epochen")
plt.xlabel("Epoche")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("plots/accuracy.png")
plt.close()

# final confusion matrix
confusion = np.zeros((10, 10), dtype=int)

for x, y in zip(X_test, y_test):
    pred = nn.forward(x)
    predicted = np.argmax(pred)
    confusion[y][predicted] += 1

plt.figure(figsize=(8,6))
plt.imshow(confusion)
plt.title("Confusion Matrix")
plt.xlabel("Vorhergesagte Ziffer")
plt.ylabel("Tatsächliche Ziffer")
plt.colorbar()
plt.savefig("plots/confusion_matrix.png")
plt.close()

# accuracy per digit
digit_accuracy = confusion.diagonal() / confusion.sum(axis=1)

plt.figure()
plt.bar(range(10), digit_accuracy)
plt.title("Genauigkeit pro Ziffer")
plt.xlabel("Ziffer")
plt.ylabel("Accuracy")
plt.savefig("plots/digit_accuracy.png")
plt.close()

print("Plots gespeichert in /plots")

print("Starting drawing GUI...")

gui = DrawGUI(nn)
gui.run()

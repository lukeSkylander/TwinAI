import numpy as np
import struct
import matplotlib.pyplot as plt


def debug_show_samples(images, labels, count=10):

    indices = np.random.choice(len(images), count, replace=False)

    plt.figure(figsize=(10, 3))

    for i, idx in enumerate(indices):

        img = images[idx].reshape(28, 28)

        plt.subplot(1, count, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(str(labels[idx]))
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def load_images(path):

    with open(path, "rb") as f:

        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))

        data = np.frombuffer(f.read(), dtype=np.uint8)

        images = data.reshape(num, rows * cols)

        images = images.astype(np.float32) / 255.0

        return images


def load_labels(path):

    with open(path, "rb") as f:

        magic, num = struct.unpack(">II", f.read(8))

        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels


def load_mnist(base_path):

    X_train = load_images(base_path + "/train-images.idx3-ubyte")
    y_train = load_labels(base_path + "/train-labels.idx1-ubyte")

    X_test = load_images(base_path + "/t10k-images.idx3-ubyte")
    y_test = load_labels(base_path + "/t10k-labels.idx1-ubyte")

    return X_train, y_train, X_test, y_test

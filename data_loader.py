from PIL import Image
import os
import numpy as np


def load_dataset(folder):
    X = []
    y = []

    for label in range(10):
        label_folder = os.path.join(folder, str(label))

        for filename in os.listdir(label_folder):
            if not filename.lower().endswith(".jpg"):
                continue

            path = os.path.join(label_folder, filename)
            img = Image.open(path).convert("L")
            img = np.array(img) / 255.0

            X.append(img.flatten())
            y.append(label)

    return np.array(X), np.array(y)

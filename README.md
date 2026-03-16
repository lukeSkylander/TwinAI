# MNIST Neural Network AI

A complete implementation of a neural network for MNIST digit recognition with an interactive drawing GUI.

## Features

- **Custom Neural Network**: 3-layer feedforward network implemented from scratch
- **MNIST Dataset Support**: Loads and processes the MNIST handwritten digit dataset
- **Interactive GUI**: Draw digits and see real-time predictions
- **Model Persistence**: Save and load trained models
- **Training Visualization**: Real-time loss tracking and validation accuracy

## Components

- `nn.py`: Neural network implementation with ReLU activation and softmax output
- `data_loader.py`: MNIST dataset loading and preprocessing
- `draw_gui.py`: Interactive Tkinter GUI for drawing and prediction
- `main.py`: Main training and evaluation script
- `test_system.py`: System verification tests

## Quick Start

1. **Run the system**:
   ```bash
   python main.py
   ```

2. **Test the system**:
   ```bash
   python test_system.py
   ```

## Architecture

- **Input Layer**: 784 neurons (28x28 pixels)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with softmax activation (digits 0-9)

## Training

The network trains for 20 epochs with:
- Batch size: 64
- Learning rate: 0.01
- Stochastic Gradient Descent optimizer
- Cross-entropy loss function

## GUI Usage

1. Draw a digit in the black canvas using your mouse
2. See real-time predictions and probability distributions
3. Use "Clear" to reset the canvas
4. Use "Random Noise" to test with random input

## Dataset

The system expects MNIST data in `./data/kaggle_mnist/` directory with files:
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## Results

The trained model typically achieves 95%+ accuracy on the MNIST test set.

Auftrag:

📝 Auftrag: Einführung in Künstliche Neuronale Netze (KNN) – Programmierprojekt in Python

🎯 Lernziele
	• Sie verstehen, wie ein künstliches Neuron funktioniert (Inputs, Gewichte, Aktivierungsfunktion)
	• Sie können ein einfaches neuronales Netz in Python selbst implementieren
	• Sie können Daten vorbereiten und in das Netz einspeisen
	• Sie können Trainingsprozesse (Gradientenabstieg, Fehlerberechnung) nachvollziehen
	• Sie können das Modell evaluieren und visualisieren
	
📚 Projektbeschreibung
Sie programmieren in kleinen Gruppen ein eigenes künstliches neuronales Netz ohne externe ML-Bibliotheken (kein TensorFlow, PyTorch usw.). Zulässig sind numpy und matplotlib.
Das Netz soll ein kleines Klassifikationsproblem lösen – Erkennung von handgeschriebenen Ziffern (MNIST, reduzierte Version)

📦 Abgabeformat
	w Python-Projekt (Ordner mit allen .py-Dateien)
	w 1–2-seitige Dokumentation (PDF)
	w Grafiken/Plots
	w Kurze Präsentation (ca. 5-10 Minuten) in der vierten Doppellektion (gleichzeitg Abgabe)
	
🧭 Bewertungskriterien
	Kriterium	Gewichtung
	Funktionierender Code	40%
	Dokumentation & Verständlichkeit	25%
	Korrekte Anwendung von KNN-Prinzipien	25%
	Präsentation	10%

Hier der Link auf MNIST Test- und Trainingsdaten:
<https://www.kaggle.com/datasets/mohamedgamal07/reduced-mnist>

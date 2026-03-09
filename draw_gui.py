import tkinter as tk
import numpy as np


class DrawGUI:
    def __init__(self, model):
        self.model = model

        self.size = 28
        self.scale = 10
        self.canvas_size = self.size * self.scale

        self.root = tk.Tk()
        self.root.title("Digit Recognition")

        self.canvas = tk.Canvas(
            self.root, width=self.canvas_size, height=self.canvas_size, bg="black"
        )
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT)

        self.predict_btn = tk.Button(
            self.button_frame, text="Predict", command=self.predict
        )
        self.predict_btn.pack(side=tk.LEFT)

        self.label = tk.Label(self.root, text="Draw a digit", font=("Arial", 16))
        self.label.pack()

        self.image = np.zeros((self.size, self.size))

    def draw(self, event):
        x = event.x // self.scale
        y = event.y // self.scale

        if 0 <= x < self.size and 0 <= y < self.size:
            self.image[y][x] = 1.0

            x0 = x * self.scale
            y0 = y * self.scale
            x1 = x0 + self.scale
            y1 = y0 + self.scale

            self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="white")

    def clear(self):
        self.canvas.delete("all")
        self.image = np.zeros((self.size, self.size))
        self.label.config(text="Draw a digit")

    def predict(self):
        vector = self.image.flatten()
        pred = self.model.forward(vector)
        digit = np.argmax(pred)

        self.label.config(text=f"Prediction: {digit}")

    def run(self):
        self.root.mainloop()

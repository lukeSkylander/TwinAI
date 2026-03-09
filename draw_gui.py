import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DrawGUI:
    def __init__(self, model):
        self.model = model

        # Parameters
        self.size = 28  # MNIST image size
        self.scale = 10  # how big each pixel appears
        self.canvas_size = self.size * self.scale

        # Tkinter root
        self.root = tk.Tk()
        self.root.title("Digit Recognition - Live Prediction")

        # Canvas for drawing
        self.canvas = tk.Canvas(
            self.root, width=self.canvas_size, height=self.canvas_size, bg="black"
        )
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        # Buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()
        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT)

        # Internal image
        self.image = np.zeros((self.size, self.size))

        # Matplotlib figure for live probabilities
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.bars = self.ax.bar(range(10), np.zeros(10), color="skyblue")
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks(range(10))
        self.ax.set_xticklabels(range(10))
        self.ax.set_xlabel("Digit")
        self.ax.set_ylabel("Probability")
        self.ax.set_title("Live Prediction Probabilities")
        self.fig.tight_layout()  # avoid clipping
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_fig.get_tk_widget().pack()

        # initialize the plot
        self.update_plot()

    def draw(self, event):
        x = event.x // self.scale
        y = event.y // self.scale

        if 0 <= x < self.size and 0 <= y < self.size:
            self.image[y][x] = 1.0

            # Draw a rectangle on the canvas
            x0 = x * self.scale
            y0 = y * self.scale
            x1 = x0 + self.scale
            y1 = y0 + self.scale
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="white")

        # Update probabilities live
        self.update_plot()

    def clear(self):
        self.canvas.delete("all")
        self.image = np.zeros((self.size, self.size))
        self.update_plot()

    def update_plot(self):
        vector = self.image.flatten()
        pred = self.model.forward(vector)

        for i, bar in enumerate(self.bars):
            bar.set_height(pred[i])

        self.fig.canvas.draw_idle()

    def run(self):
        self.root.mainloop()

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DrawGUI:

    def __init__(self, model):

        self.model = model

        self.size = 28
        self.scale = 12
        self.brush = 1

        self.canvas_size = self.size * self.scale

        self.root = tk.Tk()
        self.root.title("MNIST Digit Recognizer")

        # DRAW CANVAS
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)

        # IMAGE BUFFER
        self.image = np.zeros((self.size, self.size))

        # BUTTONS
        controls = tk.Frame(self.root)
        controls.pack()

        tk.Button(controls, text="Clear", width=10, command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Random Noise", width=12, command=self.noise).pack(side=tk.LEFT, padx=5)

        # PREDICTION LABEL
        self.pred_label = tk.Label(self.root, text="Draw a digit", font=("Arial", 18))
        self.pred_label.pack(pady=5)

        # MATPLOTLIB CHART
        self.fig, self.ax = plt.subplots(figsize=(7, 3))

        self.bars = self.ax.bar(range(10), np.zeros(10))

        self.ax.set_ylim(0, 1)
        self.ax.set_xticks(range(10))
        self.ax.set_xlabel("Digit")
        self.ax.set_ylabel("Probability")
        self.ax.set_title("Prediction Distribution")

        self.fig.tight_layout()

        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_fig.get_tk_widget().pack()

        self.update_plot()

    # DRAW WITH BRUSH
    def draw(self, event):
        x = event.x // self.scale
        y = event.y // self.scale

        for dx in range(-2, 3):
            for dy in range(-2, 3):

                px = x + dx
                py = y + dy

                if 0 <= px < self.size and 0 <= py < self.size:

                    dist = np.sqrt(dx*dx + dy*dy)

                    intensity = max(0, 1 - dist/3)

                    self.image[py][px] = min(1, self.image[py][px] + intensity * 0.3)

                    gray = int(self.image[py][px] * 255)

                    color = f'#{gray:02x}{gray:02x}{gray:02x}'

                    x0 = px * self.scale
                    y0 = py * self.scale
                    x1 = x0 + self.scale
                    y1 = y0 + self.scale

                    self.canvas.create_rectangle(
                        x0, y0, x1, y1,
                        fill=color,
                        outline=color
                    )

        self.update_plot()

    # CLEAR CANVAS
    def clear(self):

        self.canvas.delete("all")
        self.image = np.zeros((self.size, self.size))
        self.pred_label.config(text="Draw a digit")

        self.update_plot()

    # RANDOM TEST BUTTON
    def noise(self):

        self.image = np.random.rand(28, 28)

        self.canvas.delete("all")

        for y in range(self.size):
            for x in range(self.size):

                val = self.image[y][x]

                if val > 0.5:

                    x0 = x * self.scale
                    y0 = y * self.scale
                    x1 = x0 + self.scale
                    y1 = y0 + self.scale

                    self.canvas.create_rectangle(
                        x0, y0, x1, y1,
                        fill="white",
                        outline="white"
                    )

        self.update_plot()

    # UPDATE PREDICTION
    def update_plot(self):

        def center_digit(img):

            coords = np.argwhere(img > 0.1)

            if len(coords) == 0:
                return img

            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)

            digit = img[y0:y1+1, x0:x1+1]

            padded = np.zeros((28,28))

            h, w = digit.shape

            start_y = (28 - h) // 2
            start_x = (28 - w) // 2

            padded[start_y:start_y+h, start_x:start_x+w] = digit

            return padded

        img = center_digit(self.image)

        vector = img.flatten()
        pred = self.model.forward(vector)

        predicted_digit = np.argmax(pred)

        vector = vector / (vector.max() + 1e-8)

        self.pred_label.config(
            text=f"Prediction: {predicted_digit} ({pred[predicted_digit]*100:.1f}%)"
        )

        for i, bar in enumerate(self.bars):

            bar.set_height(pred[i])

            if i == predicted_digit:
                bar.set_color("orange")
            else:
                bar.set_color("skyblue")

        self.fig.canvas.draw_idle()

    def run(self):
        self.root.mainloop()


# RUN GUI WITHOUT TRAINING
if __name__ == "__main__":

    from nn import NeuralNetwork

    MODEL_PATH = "model/mnist_model.npz"

    nn = NeuralNetwork(784, 128, 10)
    nn.load(MODEL_PATH)

    gui = DrawGUI(nn)
    gui.run()

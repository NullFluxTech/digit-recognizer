"""Draw a digit with the mouse and watch the MLP's top-3 guesses update live.

    uv run draw.py

Mouse: draw.   Keys 0-9: set the label.   Enter: save to samples/<label>_<nn>.png.
Esc or Backspace: clear.
The small preview shows the 28x28 image the model actually sees, after preprocessing.
"""

import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image

from digits.inference import SAMPLES_DIR, load_model, next_sample_path, predict_probs, top_k
from digits.preprocess import to_mnist

CANVAS = 280        # drawing area in screen pixels (10x the MNIST resolution)
BRUSH = 10          # brush radius in pixels: digits ~200 px tall get strokes ~20 px wide, the
                    # same stroke-to-height ratio as MNIST (~2.5 px strokes in a 20 px box)
CELL = 5            # preview: each of the 28x28 pixels drawn as a 5x5 square


def stamp_segment(ink: np.ndarray, a: tuple[float, float], b: tuple[float, float], radius: float) -> None:
    """Paint (in place) every pixel within `radius` of the segment a-b, as (row, col) points.

    For each pixel p, the closest point of the segment is a + t (b - a), with
        t = clip( (p - a) . (b - a) / |b - a|^2 , 0, 1 )      (projection onto the segment)
    and p is ink if |p - closest| <= radius. Only the bounding box of the segment is touched.
    """
    h, w = ink.shape
    r0 = max(0, int(min(a[0], b[0]) - radius))
    r1 = min(h, int(max(a[0], b[0]) + radius) + 2)
    c0 = max(0, int(min(a[1], b[1]) - radius))
    c1 = min(w, int(max(a[1], b[1]) + radius) + 2)
    if r0 >= r1 or c0 >= c1:
        return
    rows, cols = np.mgrid[r0:r1, c0:c1]                    # (h',w') each: row / col index of every pixel
    dr, dc = b[0] - a[0], b[1] - a[1]                      # segment vector d = b - a
    length2 = dr * dr + dc * dc                            # |d|^2
    if length2 == 0:
        t = np.zeros(rows.shape)                           # a single point (a click)
    else:
        t = ((rows - a[0]) * dr + (cols - a[1]) * dc) / length2  # (h',w') (p - a) . d / |d|^2
        t = np.clip(t, 0.0, 1.0)                           # (h',w') stay on the segment
    dist2 = (rows - (a[0] + t * dr)) ** 2 + (cols - (a[1] + t * dc)) ** 2  # (h',w') |p - closest|^2
    ink[r0:r1, c0:c1][dist2 <= radius * radius] = 1.0      # boolean mask selects the pixels to paint


class DrawApp:
    def __init__(self, root: tk.Tk, model, samples_dir: Path = SAMPLES_DIR):
        self.root, self.model, self.samples_dir = root, model, samples_dir
        self.ink = np.zeros((CANVAS, CANVAS), dtype=np.float32)   # (280,280), 1 = ink
        self.last = None              # previous mouse position (row, col) while drawing
        self.pending = False          # a prediction update is already scheduled
        self.label = tk.StringVar(value="?")
        root.title("Digit recognizer: draw a digit")

        self.canvas = tk.Canvas(root, width=CANVAS, height=CANVAS, bg="white", cursor="pencil",
                                highlightthickness=1, highlightbackground="#c3c2b7")
        self.canvas.grid(row=0, column=0, rowspan=3, padx=12, pady=12)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.schedule_update())

        side = tk.Frame(root)
        side.grid(row=0, column=1, sticky="n", padx=(0, 12), pady=12)
        tk.Label(side, text="What the model sees (28x28):").pack(anchor="w")
        self.preview = tk.Canvas(side, width=28 * CELL, height=28 * CELL, bg="white",
                                 highlightthickness=1, highlightbackground="#c3c2b7")
        self.preview.pack(anchor="w", pady=(2, 10))
        self.cells = [[self.preview.create_rectangle(c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL,
                                                     width=0, fill="white")
                       for c in range(28)] for r in range(28)]
        tk.Label(side, text="Top 3:").pack(anchor="w")
        self.guess_labels = [tk.Label(side, text="", font=("Helvetica", 14 if i == 0 else 11), anchor="w")
                             for i in range(3)]
        for lab in self.guess_labels:
            lab.pack(anchor="w", fill="x")

        bottom = tk.Frame(root)
        bottom.grid(row=3, column=0, columnspan=2, sticky="we", padx=12, pady=(0, 12))
        tk.Label(bottom, text="Label (press 0-9):").pack(side="left")
        tk.Label(bottom, textvariable=self.label, font=("Helvetica", 14, "bold"), width=2).pack(side="left")
        tk.Button(bottom, text="Save (Enter)", command=self.save).pack(side="left", padx=6)
        tk.Button(bottom, text="Clear (Esc)", command=self.clear).pack(side="left")
        self.status = tk.Label(root, text=self.progress_text(), anchor="w", fg="#52514e")
        self.status.grid(row=4, column=0, columnspan=2, sticky="we", padx=12, pady=(0, 10))

        root.bind("<Key>", self.on_key)
        self.show_prediction(None)

    # --- drawing ------------------------------------------------------------------------------
    def on_press(self, event) -> None:
        self.last = (event.y, event.x)                         # Tk gives (x, y); arrays are (row, col)
        self.paint(self.last, self.last)

    def on_move(self, event) -> None:
        if self.last is None:
            return
        here = (event.y, event.x)
        self.paint(self.last, here)
        self.last = here
        self.schedule_update()

    def paint(self, a, b) -> None:
        stamp_segment(self.ink, a, b, BRUSH)
        self.canvas.create_line(a[1], a[0], b[1], b[0], width=2 * BRUSH, fill="black",
                                capstyle=tk.ROUND, joinstyle=tk.ROUND)

    def clear(self) -> None:
        self.ink[...] = 0.0
        self.canvas.delete("all")
        self.last = None
        self.show_prediction(None)

    # --- prediction ---------------------------------------------------------------------------
    def schedule_update(self) -> None:
        if not self.pending:                                   # at most one update every 80 ms
            self.pending = True
            self.root.after(80, self.update_prediction)

    def update_prediction(self) -> None:
        self.pending = False
        if self.ink.max() == 0:
            self.show_prediction(None)
            return
        x = to_mnist(self.ink)                                 # (280,280) -> (784,)
        self.show_prediction(x)

    def show_prediction(self, x: np.ndarray | None) -> None:
        img = np.zeros((28, 28)) if x is None else x.reshape(28, 28)
        for r in range(28):
            for c in range(28):
                v = int(255 * (1.0 - img[r, c]))               # ink shown dark on white
                self.preview.itemconfigure(self.cells[r][c], fill=f"#{v:02x}{v:02x}{v:02x}")
        if x is None:
            for lab in self.guess_labels:
                lab.config(text="")
            self.guess_labels[0].config(text="(draw a digit)")
            return
        for lab, (digit, p) in zip(self.guess_labels, top_k(predict_probs(self.model, x))):
            lab.config(text=f"{digit}   {p:6.1%}")

    # --- labels and saving ----------------------------------------------------------------------
    def on_key(self, event) -> None:
        if event.char.isdigit():
            self.label.set(event.char)
        elif event.keysym == "Return":
            self.save()
        elif event.keysym in ("Escape", "BackSpace"):
            self.clear()

    def save(self) -> Path | None:
        if not self.label.get().isdigit():
            self.status.config(text="Press a digit key 0-9 to set the label first.")
            return None
        if self.ink.max() == 0:
            self.status.config(text="Nothing to save: the canvas is empty.")
            return None
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        path = next_sample_path(int(self.label.get()), self.samples_dir)
        paper = np.round((1.0 - self.ink) * 255).astype(np.uint8)  # (280,280) dark ink on white, like paper
        Image.fromarray(paper).save(path)
        self.clear()
        self.status.config(text=f"Saved {path.name}.   " + self.progress_text())
        return path

    def progress_text(self) -> str:
        counts = [len(list(self.samples_dir.glob(f"{d}_*.png"))) if self.samples_dir.exists() else 0
                  for d in range(10)]
        return "Samples so far: " + "  ".join(f"{d}:{n}" for d, n in enumerate(counts))


def main() -> None:
    model = load_model()
    root = tk.Tk()
    root.resizable(False, False)
    DrawApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()

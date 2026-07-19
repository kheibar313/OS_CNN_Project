"""
GUI application (Dark Theme Version).

Features
--------
- Draw digit on canvas
- Predict drawing
- Predict image file
- Random MNIST sample
- Train model with configurable workers
- Dark theme UI
- Log panel + image preview
"""

from __future__ import annotations

import random
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk

import numpy as np
from tensorflow.keras.datasets import mnist

from model.predictor import Predictor
from trainer.trainer import Trainer

import queue


class DrawApp:

    CANVAS_SIZE = 280
    IMAGE_SIZE = 28

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.log_queue = queue.Queue()
        self.root.after(100, self._process_log_queue)

        self.root.title("OS CNN Project")
        self.root.configure(bg="#121212")

        self.predictor = Predictor()

        # ---------------- Drawing state ----------------
        self.image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.drawer = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        # ---------------- UI ----------------
        self._create_widgets()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _create_widgets(self):

        # ===== Top Bar =====
        top = tk.Frame(self.root, bg="#121212")
        top.pack(pady=10)

        tk.Label(
            top,
            text="Workers / Layer:",
            bg="#121212",
            fg="white"
        ).pack(side=tk.LEFT)

        self.worker_var = tk.IntVar(value=1)

        tk.Spinbox(
            top,
            from_=1,
            to=20,
            width=5,
            textvariable=self.worker_var,
            bg="#1e1e1e",
            fg="white",
            insertbackground="white",
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            top,
            text="Train",
            command=self.train_model,
            bg="#333333",
            fg="white",
            activebackground="#555555",
        ).pack(side=tk.LEFT, padx=5)

        # ===== Canvas =====
        self.canvas = tk.Canvas(
            self.root,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
            highlightbackground="gray"
        )
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)

        # ===== Buttons =====
        btn_frame = tk.Frame(self.root, bg="#121212")
        btn_frame.pack(pady=10)

        def btn(text, cmd):
            return tk.Button(
                btn_frame,
                text=text,
                command=cmd,
                bg="#222",
                fg="white",
                width=18,
                activebackground="#444"
            )

        btn("Predict Drawing", self.predict_canvas).grid(row=0, column=0, padx=5)
        btn("Predict Image", self.predict_image).grid(row=0, column=1, padx=5)
        btn("Random MNIST", self.random_mnist).grid(row=0, column=2, padx=5)
        btn("Clear", self.clear_canvas).grid(row=1, column=0, pady=5)

        # ===== Result =====
        self.result = tk.Label(
            self.root,
            text="Prediction:",
            font=("Arial", 14),
            bg="#121212",
            fg="white"
        )
        self.result.pack(pady=10)

        # ===== Image preview =====
        self.preview_label = tk.Label(self.root, bg="#121212")
        self.preview_label.pack(pady=5)

        # ===== Log =====
        tk.Label(
            self.root,
            text="Log",
            bg="#121212",
            fg="white",
            font=("Arial", 12, "bold")
        ).pack()

        self.log = tk.Text(
            self.root,
            width=90,
            height=15,
            bg="#1e1e1e",
            fg="white",
            insertbackground="white"
        )
        self.log.pack(padx=10, pady=5)

    # ---------------------------------------------------------
    # Log
    # ---------------------------------------------------------

    def add_log(self, text: str):
        self.log_queue.put(text)

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def train_model(self):

        workers = self.worker_var.get()

        self.add_log("=" * 50)
        self.add_log(f"Training started | workers per layer = {workers}")

        trainer = Trainer()
        trainer.logger = self.add_log
        trainer.thread_pool.logger = self.add_log
        trainer.set_epochs(workers)
        trainer.run()

        self.add_log("Training finished.")
        messagebox.showinfo("Done", "Training Completed")

    # ---------------------------------------------------------
    # Drawing
    # ---------------------------------------------------------

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="white",
            width=18,
            capstyle=tk.ROUND,
            smooth=True,
        )

        self.drawer.line(
            (self.last_x, self.last_y, event.x, event.y),
            fill=255,
            width=18,
        )

        self.last_x, self.last_y = event.x, event.y

    def clear_canvas(self):

        self.canvas.delete("all")

        self.image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.drawer = ImageDraw.Draw(self.image)

        self.result.config(text="Prediction:")
        self.preview_label.config(image="")

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict_canvas(self):

        img = self.image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
        img.save("_temp.png")

        digit, conf, t = self.predictor.predict("_temp.png")

        self.result.config(
            text=f"Digit: {digit} | Confidence: {conf:.2%} | {t*1000:.2f} ms"
        )

        self.add_log(f"Canvas -> {digit} ({conf:.2%})")

    def predict_image(self):

        path = filedialog.askopenfilename()
        if not path:
            return

        digit, conf, t = self.predictor.predict(path)

        self.result.config(
            text=f"Digit: {digit} | Confidence: {conf:.2%}"
        )

        self.add_log(f"Image -> {digit} ({conf:.2%})")

        self._show_image(path)

    # ---------------------------------------------------------
    # MNIST
    # ---------------------------------------------------------

    def random_mnist(self):

        (_, _), (x_test, y_test) = mnist.load_data()

        idx = random.randint(0, len(x_test) - 1)

        img = Image.fromarray(x_test[idx])
        img.save("_mnist.png")

        digit, conf, t = self.predictor.predict("_mnist.png")

        self.result.config(
            text=f"True={y_test[idx]} | Pred={digit} | {conf:.2%}"
        )

        self.add_log(f"MNIST #{idx} -> True={y_test[idx]} Pred={digit}")

        self._show_image("_mnist.png")

    # ---------------------------------------------------------
    # Show image in UI
    # ---------------------------------------------------------

    def _show_image(self, path: str):

        img = Image.open(path).resize((120, 120))
        img = ImageTk.PhotoImage(img)

        self.preview_label.configure(image=img)
        self.preview_label.image = img

    # ---------------------------------------------------------

    def _process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log.insert("end", msg + "\n")
            self.log.see("end")

        self.root.after(100, self._process_log_queue)


    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    def run(self):
        self.root.mainloop()
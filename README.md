# 🧠 OS CNN Project

> Operating Systems Final Project - CNN Training with Multithreading & Synchronization

![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

This project was developed as the final project for the **Operating Systems** course.

Instead of focusing only on machine learning, the project demonstrates how operating system concepts such as:

- Threads
- Scheduling
- Synchronization
- Semaphores
- Thread Pools
- Shared Resources

can be integrated into the training process of a neural network.

The CNN architecture itself remains identical to the reference implementation provided by the instructor.

---

## ✨ Features

- ✅ MNIST handwritten digit recognition
- ✅ CNN model using TensorFlow/Keras
- ✅ Epoch-based multithreading
- ✅ Layer Thread Pool
- ✅ Semaphore-based synchronization
- ✅ Scheduler implementation
- ✅ Thread status monitoring
- ✅ GUI for prediction
- ✅ Draw your own digit
- ✅ Predict image files
- ✅ Random MNIST sample prediction
- ✅ Model saving/loading
- ✅ Modular project architecture

---

# 🏗 Project Structure

```
OS_CNN_Project/

│
├── config/
│
├── data/
│
├── gui/
│
├── model/
│
├── os_layer/
│   ├── epoch_thread.py
│   ├── layer_thread.py
│   ├── scheduler.py
│   ├── synchronization.py
│   ├── thread_pool.py
│   └── thread_status.py
│
├── trainer/
│
├── saved_model/
│
├── main.py
└── requirements.txt
```

---

# ⚙ Thread Architecture

The project separates the Operating System layer from the Machine Learning layer.

```
GUI
 │
 ▼
Trainer
 │
 ▼
Scheduler
 │
 ▼
Epoch Threads
 │
 ▼
Synchronization Manager
 │
 ▼
Thread Pool
 │
 ▼
Layer Threads
 │
 ▼
TensorFlow Training
```

---

# 🔄 Training Workflow

```
Load Dataset
      │
      ▼
Create CNN Model
      │
      ▼
Create ThreadPool
      │
      ▼
Create Epoch Threads
      │
      ▼
Synchronization
      │
      ▼
Execute Layer Tasks
      │
      ▼
Train One Epoch
      │
      ▼
Collect Metrics
      │
      ▼
Evaluate Model
      │
      ▼
Save Model
```

---

# 🧩 Operating System Concepts

The following OS concepts were implemented:

- Multithreading
- Thread Pool
- Thread Scheduling
- Semaphores
- Critical Section
- Shared Resource Protection
- Synchronization
- Thread States

---

# 🎨 GUI

The graphical interface supports:

- Drawing handwritten digits
- Predicting custom images
- Predicting random MNIST images
- Model training
- Training log visualization

---

# 🧠 CNN Architecture

```
Input (28×28)

↓

Flatten

↓

Dense(128, ReLU)

↓

Dense(10, Softmax)
```

This architecture was intentionally kept unchanged to match the instructor's reference implementation.

---

# 📊 Dataset

MNIST

- 60,000 training images
- 10,000 testing images
- Image size: 28×28
- Classes: 10 digits (0–9)

---

# 🚀 Installation

```bash
git clone https://github.com/kheibar313/OS_CNN_Project.git

cd OS_CNN_Project

pip install -r requirements.txt
```

---

# ▶ Run

```bash
python main.py
```

---

# 📸 Screenshots

You can add screenshots here.

```
docs/gui.png
```

---

# 👨‍💻 Technologies

- Python
- TensorFlow
- Tkinter
- Pillow
- NumPy

---

# 📚 Educational Purpose

This project was developed for the Operating Systems course to demonstrate how multithreading and synchronization techniques can be integrated into a neural network training workflow.

---

# 👤 Author

**Mohammad Sadegh Ghiasvand**

Computer Engineering Student

GitHub:

https://github.com/kheibar313

---

# ⭐ If you found this project useful, consider giving it a star.

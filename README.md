# 🧠 AI-ML Projects

This repository contains small **AI and machine learning** projects focused on **digit recognition**.

---

## 🔢 Neural Network from Scratch

📁 `neural_network/`

* `number_recognition.py` — basic neural network built only with NumPy
* `number_recognition2.py` — improved version with better accuracy
* `number_rec_performance.py` — performance + scaling analysis
* `test.py` — tests predictions

### ⚡ Performance Notes

* Measured training time across different dataset sizes using `number_rec_performance.py`
* Used fewer iterations (100) in `number_rec_performance.py` to keep experiments fast
* Used more iterations (300+) in `number_recognition2.py` for better prediction accuracy

**Learning resources used:**

* https://www.youtube.com/watch?v=aircAruvnKk
* https://www.youtube.com/watch?v=w8yWXqWQYmU

---

## ✍️ Digit Recognizer

📁 `neural_network/digit_recognizer/`

A handwriting-based digit recognizer built with **TensorFlow/Keras** using the **MNIST dataset**.

* `draw.py` — draw a single digit for prediction
* `draw_multi_digits.py` — draw multiple digits
* `train_model.py` — train the CNN model
* `mnist_cnn_model.h5` — saved trained model

---

# gradient-descent-from-scratch
Rebuilding core machine learning algorithms using only NumPy — no scikit-learn, no PyTorch. Just math, code, and first principles.

🎯 Gradient Descent From Scratch
<p align="center"> <b>Rebuilding Machine Learning from First Principles</b><br> <i>No black boxes. No shortcuts. Just math + code.</i> </p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"> <img src="https://img.shields.io/badge/NumPy-Only-green?style=for-the-badge"> <img src="https://img.shields.io/badge/ML-From%20Scratch-orange?style=for-the-badge"> <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge"> </p>
🚀 Project Overview

What actually happens when you call model.fit()?

This project answers that by implementing:

Linear Regression
Logistic Regression
Gradient Descent (Batch, SGD, Mini-batch)

Using only:

NumPy
Matplotlib

❌ No scikit-learn
❌ No PyTorch

🎥 Gradient Descent in Action
<p align="center"> <img src="outputs/gradient_descent.gif" width="600"> </p> <p align="center"> <i>Gradient descent optimizing loss step-by-step</i> </p>

🔁 Replace the above path with your actual GIF file
(e.g., created from matplotlib animation or notebook)

🧠 Core Idea

Gradient Descent = iteratively minimizing error

Start with random weights
        ↓
Make predictions
        ↓
Compute loss
        ↓
Compute gradients
        ↓
Update weights
        ↓
Repeat until convergence
📐 Mathematical Foundation
Linear Regression
ŷ = wx + b

Loss (MSE):
L = (1/n) Σ(y - ŷ)²

Gradients:
∂L/∂w = -(2/n) Σ x(y - ŷ)
∂L/∂b = -(2/n) Σ (y - ŷ)
Logistic Regression
σ(z) = 1 / (1 + e^-z)

Loss:
Binary Cross Entropy

Gradients:
∂L/∂w = (1/n) Xᵀ(y_pred - y)
∂L/∂b = mean(y_pred - y)
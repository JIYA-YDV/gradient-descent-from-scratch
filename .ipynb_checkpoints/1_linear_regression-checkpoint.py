# 1_linear_regression.py
import numpy as np
import matplotlib.pyplot as plt

# ── Set random seed for reproducibility ───────────────────
np.random.seed(42)

# ── Generate synthetic data ───────────────────────────────
# Imagine: hours studied vs exam score
n_samples = 100
X = 2 * np.random.randn(n_samples)        # hours studied
y = 3 * X + 4 + np.random.randn(n_samples) # true: y = 3x + 4 + noise

print(f"Data: {n_samples} samples")
print(f"True relationship: y = 3x + 4 (+ noise)")
print(f"Our job: learn w=3, b=4 from data alone")

# ── Initialize weights ────────────────────────────────────
w = 0.0   # start at zero
b = 0.0   # start at zero

# ── Hyperparameters ───────────────────────────────────────
learning_rate = 0.01
n_epochs = 1000

# ── Track loss for plotting ───────────────────────────────
loss_history = []

print(f"\nTraining with lr={learning_rate}, epochs={n_epochs}")
print("="*50)

# ── GRADIENT DESCENT LOOP ─────────────────────────────────
for epoch in range(n_epochs):
    
    # ── FORWARD PASS ──────────────────────────────────────
    y_pred = w * X + b          # prediction
    error = y - y_pred          # residuals
    
    # ── COMPUTE LOSS (MSE) ────────────────────────────────
    loss = np.mean(error ** 2)
    loss_history.append(loss)
    
    # ── COMPUTE GRADIENTS ─────────────────────────────────
    # This IS backpropagation — manually calculated
    dL_dw = -2 * np.mean(X * error)   # ∂L/∂w
    dL_db = -2 * np.mean(error)        # ∂L/∂b
    
    # ── UPDATE WEIGHTS (gradient descent step) ────────────
    w = w - learning_rate * dL_dw
    b = b - learning_rate * dL_db
    
    # ── Log progress ──────────────────────────────────────
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | "
              f"w: {w:.4f} | b: {b:.4f}")

print(f"\nFinal results:")
print(f"Learned w = {w:.4f} (true: 3.0)")
print(f"Learned b = {b:.4f} (true: 4.0)")


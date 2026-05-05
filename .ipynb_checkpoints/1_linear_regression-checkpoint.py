# 1_linear_regression.py
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Set random seed for reproducibility ───────────────────
os.makedirs('outputs', exist_ok=True)
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

# ── VISUALIZATIONS ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Linear Regression: Gradient Descent From Scratch',
             fontsize=14, fontweight='bold')

# Plot 1: Loss curve
ax = axes[0]
ax.plot(loss_history, color='#e74c3c', linewidth=2)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('MSE Loss', fontsize=11)
ax.set_title('Loss Curve\n(Gradient Descent Working)', fontsize=11)
ax.grid(alpha=0.3)

# Annotate the big drop
ax.annotate('Steep gradient\n(far from minimum)',
            xy=(50, loss_history[50]),
            xytext=(200, loss_history[0]*0.7),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=9)

ax.annotate('Flat gradient\n(near minimum)',
            xy=(800, loss_history[800]),
            xytext=(600, loss_history[200]),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=9)

# Plot 2: Fitted line
ax = axes[1]
ax.scatter(X, y, alpha=0.5, color='#3498db', 
           label='Data points', s=30)
X_line = np.linspace(X.min(), X.max(), 100)
ax.plot(X_line, w * X_line + b, 
        color='#e74c3c', linewidth=2.5,
        label=f'Learned: y = {w:.2f}x + {b:.2f}')
ax.plot(X_line, 3 * X_line + 4, 
        color='green', linewidth=2, linestyle='--',
        label='True: y = 3.00x + 4.00')
ax.set_xlabel('X (hours studied)', fontsize=11)
ax.set_ylabel('y (exam score)', fontsize=11)
ax.set_title('Fitted Line vs True Line', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/linear_regression.png', 
            dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Saved: outputs/linear_regression.png")
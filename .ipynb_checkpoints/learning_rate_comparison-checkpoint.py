# learning_rate_comparison.py

import numpy as np
import matplotlib.pyplot as plt

# ── Set random seed for reproducibility ───────────────────
np.random.seed(42)

# ── Generate synthetic data ───────────────────────────────
n_samples = 100
X = 2 * np.random.randn(n_samples)
y = 3 * X + 4 + np.random.randn(n_samples)

# ── Training function ─────────────────────────────────────
def train_linear_regression(X, y, learning_rate, n_epochs=200):
    """
    Train linear regression using gradient descent
    Returns: loss history, final weight, final bias
    """
    w, b = 0.0, 0.0
    losses = []
    
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = w * X + b
        error = y - y_pred
        
        # Loss (MSE)
        loss = np.mean(error ** 2)
        losses.append(loss)
        
        # Gradients
        dL_dw = -2 * np.mean(X * error)
        dL_db = -2 * np.mean(error)
        
        # Update step
        w = w - learning_rate * dL_dw
        b = b - learning_rate * dL_db
        
        # Stop if diverging
        if loss > 1e10:
            print(f"lr={learning_rate} diverged at epoch {epoch}")
            losses.extend([float('inf')] * (n_epochs - epoch - 1))
            break
    
    return losses, w, b


# ── Compare different learning rates ──────────────────────
learning_rates = [0.001, 0.01, 0.1, 0.5]
colors = ['blue', 'green', 'orange', 'red']
labels = [
    'Too small (0.001)',
    'Good (0.01)',
    'Too large (0.1)',
    'Diverging (0.5)'
]

# ── Plot setup ────────────────────────────────────────────
plt.figure(figsize=(12, 6))

for lr, color, label in zip(learning_rates, colors, labels):
    losses, final_w, final_b = train_linear_regression(X, y, lr)
    
    # Clean losses for plotting
    finite_losses = [l if l < 1000 else None for l in losses[:200]]
    
    plt.plot(finite_losses, color=color, linewidth=2, label=label)

# ── Styling ───────────────────────────────────────────────
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Learning Rate Comparison\nSmall = Slow | Large = Unstable')
plt.legend()
plt.ylim(0, 100)
plt.grid(alpha=0.3)

# ── Save and show ─────────────────────────────────────────
plt.tight_layout()
plt.savefig('outputs/learning_rate_comparison.png', dpi=300)
plt.show()

print("✓ Saved: outputs/learning_rate_comparison.png")
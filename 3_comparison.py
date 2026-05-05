# 3_comparison.py
# Batch GD vs SGD vs Mini-batch comparison

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)
np.random.seed(42)

# ── Helper functions ──────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

# ── Generate same data as logistic regression ─────────────
n = 200
disorder_severity    = np.random.randn(n)
treatment_readiness  = np.random.randn(n)

z_true    = 1.5 * disorder_severity + 2.0 * treatment_readiness
prob_true = sigmoid(z_true)
y         = (prob_true > 0.5).astype(int)
X         = np.column_stack([disorder_severity,
                              treatment_readiness])

print(f"Dataset: {n} samples, 2 features")
print(f"Class balance: {y.mean()*100:.1f}% seeks treatment")

# ── Core training function ────────────────────────────────
def gradient_descent_variants(X, y, variant='batch',
                               batch_size=32,
                               learning_rate=0.1,
                               n_epochs=100):
    """
    Train logistic regression with different GD variants.
    
    variant options:
        'batch'      → use ALL data per update
        'sgd'        → use 1 sample per update
        'mini_batch' → use batch_size samples per update
    """
    w = np.zeros(X.shape[1])
    b = 0.0
    n = len(X)
    losses = []

    for epoch in range(n_epochs):

        if variant == 'batch':
            batches = [np.arange(n)]

        elif variant == 'sgd':
            indices = np.random.permutation(n)
            batches = [[i] for i in indices]

        elif variant == 'mini_batch':
            indices = np.random.permutation(n)
            batches = [
                indices[i:i + batch_size]
                for i in range(0, n, batch_size)
            ]

        epoch_loss = 0

        for batch_idx in batches:
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            # Forward pass
            z      = X_batch @ w + b
            y_pred = sigmoid(z)

            # Loss
            batch_loss  = binary_cross_entropy(y_batch, y_pred)
            epoch_loss += batch_loss

            # Gradients
            error = y_pred - y_batch
            dL_dw = (X_batch.T @ error) / len(batch_idx)
            dL_db = np.mean(error)

            # Divergence check
            if np.isnan(loss := batch_loss) or loss > 1e6:
                return losses  # return early if exploding

            # Weight update
            w = w - learning_rate * dL_dw
            b = b - learning_rate * dL_db

        losses.append(epoch_loss / len(batches))

    return losses


# ── Run all three variants ────────────────────────────────
print("\nTraining three variants...")

print("  [1/3] Batch GD...")
batch_losses = gradient_descent_variants(
    X, y,
    variant='batch',
    learning_rate=0.1,
    n_epochs=100
)

print("  [2/3] SGD...")
sgd_losses = gradient_descent_variants(
    X, y,
    variant='sgd',
    learning_rate=0.01,
    n_epochs=100
)

print("  [3/3] Mini-batch GD...")
minibatch_losses = gradient_descent_variants(
    X, y,
    variant='mini_batch',
    batch_size=32,
    learning_rate=0.05,
    n_epochs=100
)

print(f"\nFinal losses:")
print(f"  Batch GD:    {batch_losses[-1]:.4f}")
print(f"  SGD:         {sgd_losses[-1]:.4f}")
print(f"  Mini-batch:  {minibatch_losses[-1]:.4f}")

# ── Plot comparison ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    'Gradient Descent Variants Comparison\n'
    'Batch GD vs SGD vs Mini-batch GD',
    fontsize=13, fontweight='bold'
)

# Plot 1: All three on same axis
ax = axes[0]
ax.plot(batch_losses,
        linewidth=2.5, color='#3498db',
        label=f'Batch GD (lr=0.1) — smooth')
ax.plot(sgd_losses,
        linewidth=1.2, color='#e74c3c',
        alpha=0.8, label=f'SGD (lr=0.01) — noisy')
ax.plot(minibatch_losses,
        linewidth=2.0, color='#2ecc71',
        label=f'Mini-batch (lr=0.05, bs=32) — balanced')

ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.set_title('Loss Curves — All Variants', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Annotate
ax.annotate(
    'Smooth convergence\n(all data used)',
    xy=(50, batch_losses[50]),
    xytext=(60, batch_losses[10] * 0.8),
    arrowprops=dict(arrowstyle='->', color='#3498db'),
    fontsize=8, color='#3498db'
)

ax.annotate(
    'Noisy path\n(1 sample = unstable)',
    xy=(20, sgd_losses[20]),
    xytext=(30, max(sgd_losses[:50]) * 0.9),
    arrowprops=dict(arrowstyle='->', color='#e74c3c'),
    fontsize=8, color='#e74c3c'
)

# Plot 2: Smoothed SGD to show it DOES converge
ax2 = axes[1]

# Smooth SGD with moving average
window = 10
sgd_smooth = np.convolve(
    sgd_losses,
    np.ones(window)/window,
    mode='valid'
)

ax2.plot(batch_losses,
         linewidth=2.5, color='#3498db',
         label='Batch GD')
ax2.plot(range(len(sgd_smooth)), sgd_smooth,
         linewidth=2, color='#e74c3c',
         linestyle='--',
         label='SGD (smoothed — moving avg)')
ax2.plot(minibatch_losses,
         linewidth=2.0, color='#2ecc71',
         label='Mini-batch GD')

ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Smoothed Comparison\n'
              '(SGD smoothed to show true trend)',
              fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Add summary box
summary = (
    "Summary:\n"
    "Batch GD:   Smooth, slow, stable\n"
    "SGD:        Fast, noisy, unstable\n"
    "Mini-batch: Balanced — industry standard\n"
    "\nYour NAM thesis used mini-batch\n"
    "with batch_size=256 ✓"
)
ax2.text(
    0.98, 0.98, summary,
    transform=ax2.transAxes,
    fontsize=8.5,
    verticalalignment='top',
    horizontalalignment='right',
    fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='#f8f9fa',
              alpha=0.8)
)

plt.tight_layout()
plt.savefig('outputs/gd_comparison.png',
            dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Saved: outputs/gd_comparison.png")

# ── Key insight print ─────────────────────────────────────
print("""
KEY INSIGHT:
─────────────────────────────────────────────────
Your NAM thesis code used:
    batch_size = 256
    optimizer  = Adam (adaptive mini-batch GD)
    
This is mini-batch gradient descent.
3,279 samples / 256 batch = ~13 steps per epoch
78 epochs × 13 steps = ~1,014 weight updates

Each update used exactly the math in this file.
─────────────────────────────────────────────────
""")
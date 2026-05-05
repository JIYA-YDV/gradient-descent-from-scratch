# 2_logistic_regression.py
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Sigmoid function ──────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ── Binary Cross Entropy Loss ─────────────────────────────
def binary_cross_entropy(y_true, y_pred):
    # Clip to prevent log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * np.log(1 - y_pred)
    )

# ── Generate mental health-like data ─────────────────────
# Feature 1: disorder severity (0-1)
# Feature 2: treatment readiness (0-1)
# Label: seeks treatment (0 or 1)

n = 200
disorder_severity = np.random.randn(n)
treatment_readiness = np.random.randn(n)

# True relationship (what we want model to learn)
z_true = 1.5 * disorder_severity + 2.0 * treatment_readiness
prob_true = sigmoid(z_true)
y = (prob_true > 0.5).astype(int)

# Stack features
X = np.column_stack([disorder_severity, 
                     treatment_readiness])

print("Simulated mental health dataset:")
print(f"  Features: disorder_severity, treatment_readiness")
print(f"  Samples: {n}")
print(f"  Seeks treatment: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"  True weights: w1=1.5, w2=2.0")

# ── Initialize ────────────────────────────────────────────
w = np.zeros(2)    # weights for 2 features
b = 0.0            # bias
learning_rate = 0.1
n_epochs = 500

loss_history = []
accuracy_history = []

# ── GRADIENT DESCENT LOOP ─────────────────────────────────
for epoch in range(n_epochs):
    
    # ── FORWARD PASS ──────────────────────────────────────
    z = X @ w + b              # weighted sum
    y_pred = sigmoid(z)        # apply sigmoid → probability
    
    # ── COMPUTE LOSS ──────────────────────────────────────
    loss = binary_cross_entropy(y, y_pred)
    loss_history.append(loss)
    
    # ── COMPUTE ACCURACY ──────────────────────────────────
    accuracy = np.mean((y_pred > 0.5) == y)
    accuracy_history.append(accuracy)
    
    # ── COMPUTE GRADIENTS ─────────────────────────────────
    # For logistic regression:
    # ∂L/∂w = (1/n) X^T (y_pred - y)
    # ∂L/∂b = (1/n) Σ(y_pred - y)
    
    error = y_pred - y                    # prediction error
    dL_dw = (X.T @ error) / n            # gradient for weights
    dL_db = np.mean(error)               # gradient for bias
    
    # ── UPDATE WEIGHTS ────────────────────────────────────
    w = w - learning_rate * dL_dw
    b = b - learning_rate * dL_db
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | "
              f"Accuracy: {accuracy:.4f} | "
              f"w: [{w[0]:.3f}, {w[1]:.3f}]")

print(f"\nFinal Results:")
print(f"Learned weights: w1={w[0]:.4f} (true: 1.5), "
      f"w2={w[1]:.4f} (true: 2.0)")
print(f"Final accuracy: {accuracy_history[-1]:.4f}")

# ── THIS IS YOUR LOGISTIC REGRESSION ─────────────────────
print(f"\nConnection to your thesis:")
print(f"Your LR found: mh_disorder_past=+1.1396")
print(f"This is the same w — learned by gradient descent!")

# ── VISUALIZATION ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Logistic Regression From Scratch\n'
             '(Binary Cross-Entropy Loss + Gradient Descent)',
             fontsize=13, fontweight='bold')

# Plot 1: Loss curve
ax = axes[0]
ax.plot(loss_history, color='#e74c3c', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Binary Cross-Entropy Loss')
ax.set_title('Loss Curve')
ax.grid(alpha=0.3)

# Plot 2: Accuracy curve  
ax = axes[1]
ax.plot(accuracy_history, color='#2ecc71', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Over Training')
ax.set_ylim(0, 1)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3)

# Plot 3: Decision boundary
ax = axes[2]
# Plot data points
ax.scatter(X[y==1, 0], X[y==1, 1], 
           color='#3498db', label='Seeks treatment', 
           alpha=0.7, s=50)
ax.scatter(X[y==0, 0], X[y==0, 1], 
           color='#e74c3c', label='No treatment',
           alpha=0.7, s=50)

# Decision boundary: w1*x1 + w2*x2 + b = 0
x1_range = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x2_boundary = -(w[0] * x1_range + b) / w[1]
ax.plot(x1_range, x2_boundary, 
        'k-', linewidth=2, label='Decision boundary')

ax.set_xlabel('Disorder Severity')
ax.set_ylabel('Treatment Readiness')
ax.set_title('Decision Boundary\n(Learned by Gradient Descent)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/logistic_regression.png',
            dpi=300, bbox_inches='tight')
plt.show()
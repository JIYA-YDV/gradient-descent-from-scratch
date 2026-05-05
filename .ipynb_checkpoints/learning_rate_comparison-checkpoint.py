def train_linear_regression(X, y, learning_rate, n_epochs=200):
    """Train and return loss history"""
    w, b = 0.0, 0.0
    losses = []
    
    for epoch in range(n_epochs):
        y_pred = w * X + b
        error = y - y_pred
        loss = np.mean(error ** 2)
        losses.append(loss)
        
        # Gradient descent
        dL_dw = -2 * np.mean(X * error)
        dL_db = -2 * np.mean(error)
        
        w = w - learning_rate * dL_dw
        b = b - learning_rate * dL_db
        
        # Stop if diverging
        if loss > 1e10:
            print(f"  lr={learning_rate}: DIVERGED at epoch {epoch}")
            losses.extend([float('inf')] * (n_epochs - epoch - 1))
            break
    
    return losses, w, b

# Compare 4 learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]
colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
labels = ['Too small (0.001)', 'Just right (0.01)', 
          'Too large (0.1)', 'Way too large (0.5)']

fig, ax = plt.subplots(figsize=(12, 6))

for lr, color, label in zip(learning_rates, colors, labels):
    losses, final_w, final_b = train_linear_regression(X, y, lr)
    
    # Only plot finite values
    finite_losses = [l if l < 1000 else None for l in losses[:200]]
    epochs = list(range(200))
    
    ax.plot(epochs, finite_losses, 
            color=color, linewidth=2.5, label=label)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.set_title('Learning Rate Comparison\n'
             'Too small=slow, Too large=diverges, Just right=converges',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/learning_rate_comparison.png',
            dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: outputs/learning_rate_comparison.png")
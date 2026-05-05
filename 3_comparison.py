# 3_comparison.py

def gradient_descent_variants(X, y, variant='batch', 
                               batch_size=32, 
                               learning_rate=0.1,
                               n_epochs=100):
    """
    variant: 'batch', 'sgd', 'mini_batch'
    """
    w = np.zeros(X.shape[1])
    b = 0.0
    n = len(X)
    losses = []
    
    for epoch in range(n_epochs):
        
        if variant == 'batch':
            # Use ALL data
            indices = np.arange(n)
            batches = [indices]
            
        elif variant == 'sgd':
            # Use ONE sample at a time
            indices = np.random.permutation(n)
            batches = [[i] for i in indices]
            
        elif variant == 'mini_batch':
            # Use batch_size samples
            indices = np.random.permutation(n)
            batches = [indices[i:i+batch_size] 
                      for i in range(0, n, batch_size)]
        
        epoch_loss = 0
        
        for batch_idx in batches:
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            # Forward
            z = X_batch @ w + b
            y_pred = sigmoid(z)
            
            # Loss
            batch_loss = binary_cross_entropy(y_batch, y_pred)
            epoch_loss += batch_loss
            
            # Gradients
            error = y_pred - y_batch
            dL_dw = (X_batch.T @ error) / len(batch_idx)
            dL_db = np.mean(error)
            
            # Update
            w = w - learning_rate * dL_dw
            b = b - learning_rate * dL_db
        
        losses.append(epoch_loss / len(batches))
    
    return losses

# Run all three
batch_losses    = gradient_descent_variants(
    X, y, 'batch', learning_rate=0.1)
sgd_losses      = gradient_descent_variants(
    X, y, 'sgd', learning_rate=0.01)
minibatch_losses = gradient_descent_variants(
    X, y, 'mini_batch', batch_size=32, learning_rate=0.05)

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(batch_losses, linewidth=2.5, 
        color='#3498db', label='Batch GD (all data)')
ax.plot(sgd_losses, linewidth=1.5, alpha=0.7,
        color='#e74c3c', label='SGD (1 sample)')
ax.plot(minibatch_losses, linewidth=2, 
        color='#2ecc71', label='Mini-batch GD (32 samples)')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Batch GD vs SGD vs Mini-batch GD\n'
             'Smooth=Batch, Noisy=SGD, Balanced=Mini-batch',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Add annotations
ax.annotate('Smooth but slow\n(uses all data)',
            xy=(50, batch_losses[50]),
            xytext=(60, batch_losses[20]),
            arrowprops=dict(arrowstyle='->'),
            fontsize=9, color='#3498db')

ax.annotate('Fast but noisy\n(1 sample = unstable)',
            xy=(30, sgd_losses[30]),
            xytext=(40, max(sgd_losses)*0.8),
            arrowprops=dict(arrowstyle='->'),
            fontsize=9, color='#e74c3c')

plt.tight_layout()
plt.savefig('outputs/gd_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.show()
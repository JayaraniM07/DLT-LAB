import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# Load and preprocess data
data = load_breast_cancer()
X, y = data.data, data.target.reshape(-1, 1)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize
inp, hid, out = X.shape[1], 10, 1
W1, b1 = np.random.randn(inp, hid), np.zeros((1, hid))
W2, b2 = np.random.randn(hid, out), np.zeros((1, out))
sigmoid = lambda z: 1 / (1 + np.exp(-z))
losses, val_losses = [], []

# Training
for epoch in range(1, 51):
    Z1 = X_train @ W1 + b1
    A1 = np.tanh(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    loss = -np.mean(y_train * np.log(A2 + 1e-8) + (1 - y_train) * np.log(1 - A2 + 1e-8))
    losses.append(loss)

    dZ2 = A2 - y_train
    dW2 = A1.T @ dZ2 / len(y_train)
    db2 = np.mean(dZ2, axis=0, keepdims=True)
    dZ1 = (dZ2 @ W2.T) * (1 - A1 ** 2)
    dW1 = X_train.T @ dZ1 / len(y_train)
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    W1 -= 0.01 * dW1; b1 -= 0.01 * db1
    W2 -= 0.01 * dW2; b2 -= 0.01 * db2

    Z1t = X_test @ W1 + b1
    A1t = np.tanh(Z1t)
    A2t = sigmoid(A1t @ W2 + b2)
    val_loss = -np.mean(y_test * np.log(A2t + 1e-8) + (1 - y_test) * np.log(1 - A2t + 1e-8))
    val_losses.append(val_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} - loss: {loss:.4f}, val_loss: {val_loss:.4f}")

# Accuracy
y_pred = (A2t > 0.5).astype(int)
acc = np.mean(y_pred == y_test) * 100
print(f"\nFinal Test Accuracy: {acc:.2f}%")

# Plot
plt.plot(losses, label="Train loss")
plt.plot(val_losses, label="Validation loss", linestyle='--')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Over Epochs")
plt.legend(); plt.show()

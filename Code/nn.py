from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target
C = 3
print(X)
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])

def predict(X, W):
    Xb = add_bias(X)
    return np.argmax(Xb @ W.T, axis=1)

def accuracy(X, y, W):
    return np.mean(predict(X, W) == y)

def train_batch(X, y, C, lr=0.005, epochs=300):
    W = np.zeros((C, X.shape[1] + 1))
    for _ in range(epochs):
        Xb = add_bias(X)
        for c in range(C):
            yc = np.where(y == c, 1, -1)
            y_hat = np.where(Xb @ W[c] >= 0, 1, -1)
            mis = yc != y_hat
            if np.any(mis):
                W[c] += lr * (yc[mis][:, None] * Xb[mis]).sum(axis=0)
    return W

def train_minibatch(X, y, C, lr=0.01, epochs=300, batch_size=12):
    W = np.zeros((C, X.shape[1] + 1))
    n = X.shape[0]
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            xb = X[idx[i:i+batch_size]]
            yb = y[idx[i:i+batch_size]]
            Xb = add_bias(xb)
            for c in range(C):
                yc = np.where(yb == c, 1, -1)
                y_hat = np.where(Xb @ W[c] >= 0, 1, -1)
                mis = yc != y_hat
                if np.any(mis):
                    W[c] += lr * (yc[mis][:, None] * Xb[mis]).sum(axis=0)
    return W

def train_sgd(X, y, C, lr=0.05, epochs=300):
    W = np.zeros((C, X.shape[1] + 1))
    n = X.shape[0]
    for _ in range(epochs):
        idx = np.random.permutation(n)
        for i in idx:
            xi = X[i:i+1]
            yi = y[i]
            Xb = add_bias(xi)
            for c in range(C):
                yc = 1 if yi == c else -1
                if yc * (Xb @ W[c]) <= 0:
                    W[c] += lr * yc * Xb[0]
    return W

W_batch = train_batch(X_train, y_train, C)
W_mini  = train_minibatch(X_train, y_train, C)
W_sgd   = train_sgd(X_train, y_train, C)

print("TRAIN ACCURACY")
print("Batch :", accuracy(X_train, y_train, W_batch))
print("Mini  :", accuracy(X_train, y_train, W_mini))
print("SGD   :", accuracy(X_train, y_train, W_sgd))

print("\nTEST ACCURACY")
print("Batch :", accuracy(X_test, y_test, W_batch))
print("Mini  :", accuracy(X_test, y_test, W_mini))
print("SGD   :", accuracy(X_test, y_test, W_sgd))

X_mean = X_train.mean(axis=0)
Xc = X_train - X_mean
cov = np.cov(Xc, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
Wpca = eigvecs[:, idx[:2]]
X2 = Xc @ Wpca

plt.style.use("seaborn-v0_8-darkgrid")

def plot_decision(W, title):
    x_min, x_max = X2[:,0].min()-1, X2[:,0].max()+1
    y_min, y_max = X2[:,1].min()-1, X2[:,1].max()+1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_4d = grid_2d @ Wpca.T + X_mean
    Z = predict(grid_4d, W).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.35, cmap="viridis")
    colors = ["tab:red", "tab:blue", "tab:green"]

    for c in range(C):
        plt.scatter(
            X2[y_train == c, 0],
            X2[y_train == c, 1],
            c=colors[c],
            s=40,
            edgecolor="black",
            label=f"Class {c}"
        )

    plt.title(title, fontsize=12, weight="bold")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(frameon=True)
    plt.axis("equal")

plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
plot_decision(W_batch, "Batch Perceptron (lr=0.005)")
plt.subplot(1,3,2)
plot_decision(W_mini, "Mini-batch (lr=0.01, b=12)")
plt.subplot(1,3,3)
plot_decision(W_sgd, "SGD (lr=0.05)")
plt.tight_layout()
plt.show()

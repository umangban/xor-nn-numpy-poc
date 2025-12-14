import numpy as np

# -----------------------
# Activation functions
# -----------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


# -----------------------
# Derivative functions
# -----------------------

def sigmoid_deriv_from_sigmoid(s):
    return s * (1 - s)

def tanh_deriv_from_pre_activation(z):
    return 1.0 - np.tanh(z) ** 2

# -----------------------
# Training function
# -----------------------
def train_xor(
    hidden_dim=4,
    lr=0.1,
    epochs=30000,
    seed=42,
    verbose=False,
    print_every=1000
):
    """
    Train a 2-layer neural network on XOR.

    Returns:
        params (dict): trained weights and biases
        loss_history (list): loss per epoch
    """

    np.random.seed(seed)

    # XOR dataset
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    n_samples, input_dim = X.shape
    output_dim = 1

    # Xavier initialization
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1.0 / input_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1.0 / hidden_dim)
    b2 = np.zeros((1, output_dim))

    loss_history = []
    eps = 1e-9

    for epoch in range(epochs):
        # ----- Forward pass -----
        z1 = X @ W1 + b1
        a1 = tanh(z1)

        z2 = a1 @ W2 + b2
        out = sigmoid(z2)

        # ----- Loss -----
        loss = -(
            y * np.log(out + eps) +
            (1 - y) * np.log(1 - out + eps)
        )
        loss = np.mean(loss)
        loss_history.append(loss)


        if verbose and epoch % print_every == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

        # ----- Backpropagation -----
        d_out = out - y
        dW2 = a1.T @ d_out / n_samples
        db2 = np.mean(d_out, axis=0, keepdims=True)

        da1 = d_out @ W2.T
        dz1 = da1 * tanh_deriv_from_pre_activation(z1)
        dW1 = X.T @ dz1 / n_samples
        db1 = np.mean(dz1, axis=0, keepdims=True)

        # ----- Gradient descent -----
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "final_predictions": out
    }

    return params, loss_history


# ---------- Inference ----------
def predict_xor(X, params):
    """
    Run forward pass using trained parameters.

    Args:
        X (np.array): shape (n_samples, 2)
        params (dict): trained weights

    Returns:
        float: probability
    """

    z1 = X @ params["W1"] + params["b1"]
    a1 = tanh(z1)

    z2 = a1 @ params["W2"] + params["b2"]
    out = sigmoid(z2)

    return out[0, 0]
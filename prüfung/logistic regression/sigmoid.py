import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid function.
    """
    return 1 / (1 + np.exp(-z))

# Given values
theta = np.array([0, 1.5, 0.22])  # theta_0, theta_1, theta_2
x = np.array([1, 0.002, 4])  # Bias term (1), x1 (2 grams = 0.002 kg), x2 (4 microCuries)

# Compute the hypothesis
z = np.dot(theta, x)  # θ^T * x
y_hat = sigmoid(z)  # Apply the sigmoid function

# Output the result
print(f"The machine's prediction (ŷ) for the given inputs is: {y_hat:.4f}")

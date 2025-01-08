import numpy as np

# Gegebene Daten
X = np.array([
    [1, 29932.5, 4.8, 70160, 18],
    [1, 31007.8, 1, 104458, 16.4],
    [1, 32181.2, 1, 232666, 16.9]
])  # Design-Matrix

y = np.array([5.9, 5.6, 5.4])  # Zielvektor

# Normalengleichung: θ = (X^T X)^(-1) X^T y
X_transpose = X.T
theta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

# Ergebnis
print("Berechnete Parameter (Theta):")
print(theta)

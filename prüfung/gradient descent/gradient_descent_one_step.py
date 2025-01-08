import numpy as np
import matplotlib.pyplot as plt

# Datenpunkte (X, Y)
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Y = np.array([1.0, 2.0, 1.3, 3.75, 2.25])

# Initialisierung
theta_0 = 1.0  # Intercept
theta_1 = 0.5  # Slope
alpha = 0.1    # Lernrate
M = len(X)     # Anzahl der Datenpunkte

# Hypothese h_theta(x)
def hypothesis(x, theta_0, theta_1):
    return theta_0 + theta_1 * x

# Kostenfunktion
def cost_function(X, Y, theta_0, theta_1):
    return (1 / (2 * M)) * np.sum((hypothesis(X, theta_0, theta_1) - Y) ** 2)

# Gradientenberechnung
def gradient_descent_step(X, Y, theta_0, theta_1, alpha):
    error = hypothesis(X, theta_0, theta_1) - Y
    theta_0_update = theta_0 - alpha * (1 / M) * np.sum(error)
    theta_1_update = theta_1 - alpha * (1 / M) * np.sum(error * X)
    return theta_0_update, theta_1_update

# Zeichne die Datenpunkte und die Regressionslinie
def plot_regression(X, Y, theta_0, theta_1, title):
    plt.scatter(X, Y, color='blue', label='Datenpunkte')
    x_line = np.linspace(min(X), max(X), 100)
    y_line = hypothesis(x_line, theta_0, theta_1)
    plt.plot(x_line, y_line, color='red', label='Regression')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# Schritt 1: Initiale Regressionslinie zeichnen
# plot_regression(X, Y, theta_0, theta_1, 'Initiale Regressionslinie')

# Kosten vor Update
initial_cost = cost_function(X, Y, theta_0, theta_1)
print(f"Kosten vor Gradientenabstieg: {initial_cost}")

# Schritt 2: Ein Schritt des Gradientenabstiegs
theta_0, theta_1 = gradient_descent_step(X, Y, theta_0, theta_1, alpha)

# Kosten nach Update
updated_cost = cost_function(X, Y, theta_0, theta_1)
print(f"Kosten nach einem Schritt Gradientenabstieg: {updated_cost}")

# Zeichne die neue Regressionslinie
plot_regression(X, Y, theta_0, theta_1, 'Regressionslinie nach einem Schritt Gradientenabstieg')

# Ausgabe der aktualisierten Parameter
print(f"Aktualisierte Parameter: theta_0 = {theta_0}, theta_1 = {theta_1}")

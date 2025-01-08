import numpy as np
import matplotlib.pyplot as plt

# Daten
X = np.array([1.00, 2.00, 3.00, 4.00, 5.00])
Y = np.array([1.00, 2.00, 1.30, 3.75, 2.25])

# Berechnung von Mittelwerten
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Berechnung der Parameter theta_1 (Steigung) und theta_0 (y-Achsenabschnitt)
numerator = np.sum((X - mean_x) * (Y - mean_y))
denominator = np.sum((X - mean_x)**2)
theta_1 = numerator / denominator
theta_0 = mean_y - theta_1 * mean_x

# Ausgabe der Ergebnisse
print(f"Theta_0 (y-Achsenabschnitt): {theta_0}")
print(f"Theta_1 (Steigung): {theta_1}")

# Erstellung des Scatterplots
plt.scatter(X, Y, color='blue', label='Datenpunkte')

# Erstellung der Regressionslinie
x_line = np.linspace(min(X), max(X), 100)
y_line = theta_0 + theta_1 * x_line
plt.plot(x_line, y_line, color='red', label='Regressionslinie')

# Plot-Anpassungen
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Lineare Regression')
plt.grid(True)
plt.show()

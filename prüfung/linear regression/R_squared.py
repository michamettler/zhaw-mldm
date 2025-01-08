import numpy as np

# Gegebene Daten
X = np.array([1.00, 2.00, 3.00, 4.00, 5.00])
Y = np.array([1.00, 2.00, 1.30, 3.75, 2.25])

# Gegebene Parameter der Regressionslinie
theta_0 = 0.785
theta_1 = 0.425

# Vorhersage mit der Regressionslinie
Y_pred = theta_0 + theta_1 * X

# Mittelwert von Y
mean_y = np.mean(Y)

# Berechnung der Summe der quadratischen Abweichungen (Residuen)
SS_res = np.sum((Y - Y_pred)**2)

# Berechnung der Gesamtstreuung
SS_tot = np.sum((Y - mean_y)**2)

# Berechnung von R^2
R_squared = 1 - (SS_res / SS_tot)

# Ergebnisse
print(f"SS_res (Residuen): {SS_res}")
print(f"SS_tot (Gesamtstreuung): {SS_tot}")
print(f"R^2 (Determinationskoeffizient): {R_squared}")

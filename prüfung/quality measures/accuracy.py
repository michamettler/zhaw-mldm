import numpy as np

def calculate_accuracy(y_true, y_pred, threshold):
    """
    Calculate the accuracy for a given threshold.
    """
    # Convert predictions to binary labels based on the threshold
    y_pred_labels = (y_pred > threshold).astype(int)
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred_labels)
    return accuracy

# True labels (ground truth)
y_true = np.array([0, 0, 1, 1])

# Predicted probabilities
y_pred = np.array([0.2, 0.6, 0.5, 0.9])

# Define thresholds
thresholds = [0.5, 0.7]

# Calculate and print accuracy for each threshold
for tau in thresholds:
    accuracy = calculate_accuracy(y_true, y_pred, tau)
    print(f"Accuracy for threshold τ = {tau}: {accuracy:.2f}")

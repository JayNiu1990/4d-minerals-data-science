import numpy as np
from pygam import LinearGAM, s, f
import matplotlib.pyplot as plt
import pandas as pd
# Generate some example data
np.random.seed(42)
# X_data = np.random.rand(100)  # 1D array for X
# y_data = 3 * X_data + np.sin(5 * X_data) + np.random.normal(0, 0.5, size=100)


df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\multiscale.csv")
X_data = np.array(df['x']).reshape(-1,1)
y_data = np.array(df['y'])

# Reshape X_data to a 2D array since pygam expects a 2D input
X_data = X_data.reshape(-1, 1)

# Fit a GAM model
gam_model = LinearGAM().fit(X_data, y_data)

XX = np.linspace(X_data.min(), X_data.max(), 100).reshape(-1, 1)
y_pred = gam_model.predict(XX)

mra = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\multiscale_mra.csv")
x1 = np.array(mra['x'])
y1 = np.array(mra['y'])

plt.scatter(X_data, y_data, color='b', alpha=0.3, label='Data')
plt.scatter(x1, y1, color='m', alpha=0.3, label='Data')
plt.plot(XX, y_pred, color='r', label='GAM Prediction')
plt.title('GAM Regression')
plt.xlabel('log10 (Tonnage)')
plt.ylabel('log10 (Variance)')
plt.legend()
plt.show()



from pygam import LinearGAM
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(42)
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\multiscale.csv")
X_data = np.array(df['x']).reshape(-1,1)
y_data = np.array(df['y'])
# Create a LinearGAM model
gam_model = LinearGAM().fit(X_data, y_data)

# Make predictions
y_pred = gam_model.predict(X_data)

# Number of bootstrap samples
num_samples = 1000

# Bootstrap samples and predictions
bootstrap_predictions = np.zeros((num_samples, len(X_data)))

for i in range(num_samples):
    # Generate a bootstrap sample
    indices = np.random.choice(len(X_data), size=len(X_data), replace=True)
    X_bootstrap = X_data[indices]
    y_bootstrap = y_data[indices]

    # Fit a LinearGAM model on the bootstrap sample
    gam_bootstrap = LinearGAM().fit(X_bootstrap, y_bootstrap)

    # Make predictions on the original data
    bootstrap_predictions[i, :] = gam_bootstrap.predict(X_data)

# Calculate percentiles for prediction intervals
lower_bound = np.percentile(bootstrap_predictions, 2.5, axis=0)
upper_bound = np.percentile(bootstrap_predictions, 97.5, axis=0)

# Plot the data and the GAM predictions with uncertainties
plt.scatter(X_data, y_data, label='Data')
plt.plot(X_data, y_pred, color='red', label='GAM Prediction')
plt.fill_between(X_data.flatten(), lower_bound.flatten(), upper_bound.flatten(), color='orange', alpha=0.3, label='95% Prediction Intervals')
plt.legend()
plt.show()









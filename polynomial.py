import pandas as pd
import numpy as np
import math
import numpy
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\multiscale.csv")
x1 = np.array(df['x'])
y1 = np.array(df['y'])
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\Mineral sorting\\Kansanshi\\multiscale_mra.csv")
x2 = np.array(df['x']).reshape(-1,1)
y2 = np.array(df['y'])


# from scipy.optimize import curve_fit
# np.random.seed(0)
# # Define the nonlinear function
# def quadratic_func(x, a, b, c, d):
#     return a + b * x + c * x**2 +d*x**3 
# popt, pcov = curve_fit(quadratic_func, x1, y1)
# Visualize the results

plt.scatter(x1, y1, label='Data')
plt.scatter(x2, y2, label='MRA')
# plt.plot(x1, quadratic_func(x1, *popt), 'r-', label='Fit')
plt.legend()
plt.show()
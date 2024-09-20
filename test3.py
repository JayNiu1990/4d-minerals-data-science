import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd
import scipy
# Read data from CSV
fields = ['BHID', 'Fe_dh', 'As_dh', 'CuT_dh', "X", "Y", "Z", "LITH", "AL_ALT"]
df = pd.read_csv("C:\\Users\\NIU004\\OneDrive - CSIRO\\Desktop\\dhesc_ass_geol_attribs.csv", skipinitialspace=True, usecols=fields)

# Filter out invalid data
df = df[(pd.to_numeric(df["CuT_dh"], errors='coerce') > 0) & (pd.to_numeric(df["Fe_dh"], errors='coerce') > 0)]

def confidence_ellipse(x, y, ax, confidence=0.8, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    confidence : float, optional (default=0.8)
        The confidence level for the ellipse (between 0 and 1). Default is 0.8 for 80% confidence.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Calculate the scaling factor based on the desired confidence level
    scale_factor = np.sqrt(2.0) * scipy.stats.norm.ppf((1 + confidence) / 2)

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Calculate the angle of rotation for the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Create the ellipse
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=np.sqrt(eigenvalues[0]) * scale_factor * 2,
                      height=np.sqrt(eigenvalues[1]) * scale_factor * 2,
                      angle=angle,
                      facecolor=facecolor,
                      **kwargs)

    ax.add_patch(ellipse)
    return ellipse

# Plot the data and 80% confidence ellipse
fig, ax = plt.subplots(figsize=(12, 8))
x = np.log10(df['CuT_dh'].astype("float"))
y = np.log10(df['Fe_dh'].astype("float"))
ax.scatter(x, y, s=0.5)
ax.set_xlabel('log(Cu) (w.t%)')
ax.set_ylabel('log(Fe) (w.t%)')

confidence_ellipse(x, y, ax, confidence=0.8, edgecolor='fuchsia', linestyle='--')

plt.title('Scatter Plot with 80% Confidence Ellipse')
plt.show()

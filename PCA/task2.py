import numpy as np
import matplotlib
import matplotlib.pyplot as plt

scores = np.genfromtxt('X_reduced.csv', delimiter=';')
loadings = np.genfromtxt('X_loadings.csv', delimiter=';')

values = np.dot(scores, loadings.T)

plt.imshow(values, cmap='Greys_r')
plt.show()


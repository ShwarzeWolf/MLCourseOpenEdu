import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt

raw_data = pd.read_csv("dataTask1.csv", header=None)
data = pd.DataFrame(raw_data)

pca = PCA(n_components=3, svd_solver='full')
data_transformed = pca.fit_transform(data)
print(data_transformed)

plt.plot(data_transformed[:, 0], data_transformed[:, 1], 'o')
plt.show()

explained_variance = np.round(np.cumsum(pca.explained_variance_ratio_), 3)
print(explained_variance)

plt.plot(np.arange(3), explained_variance, ls = '-')
plt.show()